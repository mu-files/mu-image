# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Array handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from enum import StrEnum
from math import gcd
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.lib.array_utils import byte_bounds


def _is_direct_source(arr: np.ndarray) -> bool:
    """True when the engine can bind ``arr`` itself as a source buffer.

    Pixels in a row are adjacent, the data pointer is element-aligned, and
    the row pitch is a positive multiple of a pixel. A negative pitch is a
    flipped view, not a source buffer. The stride of an axis of length 1 is
    ignored, as NumPy and the engine binding both do: ``v[None, :]`` has a
    row stride of 0.
    """
    itemsize = int(arr.dtype.itemsize)
    if int(arr.ctypes.data) % itemsize != 0:
        return False

    def is_packed_axis(axis: int, step_bytes: int) -> bool:
        return int(arr.shape[axis]) == 1 or int(arr.strides[axis]) == step_bytes

    if arr.ndim == 2:
        pixel = itemsize
        if not is_packed_axis(1, itemsize):
            return False
    elif arr.ndim == 3:
        channels = int(arr.shape[2])
        pixel = itemsize * channels
        if not (is_packed_axis(2, itemsize) and is_packed_axis(1, pixel)):
            return False
    else:
        return False
    if int(arr.shape[0]) == 1:
        return True
    row_pitch = int(arr.strides[0])
    packed = int(arr.shape[1]) * pixel
    return row_pitch >= packed and row_pitch % pixel == 0


def _base_chain(arr: np.ndarray) -> list[np.ndarray]:
    """``arr`` then every ndarray along ``.base``."""
    chain: list[np.ndarray] = []
    seen: set[int] = set()
    current: Any = arr
    while isinstance(current, np.ndarray) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.base
    return chain


def _seal_ndarray(arr: np.ndarray) -> np.ndarray:
    """Mark ``arr`` and every ndarray along ``.base`` read-only. Do not copy."""
    arr = np.asarray(arr)
    for current in _base_chain(arr):
        if current.flags.writeable:
            current.setflags(write=False)
    return arr


class _Memory:
    """Presents ``interface`` to NumPy as an array and keeps ``base`` alive.

    ``interface`` is a NumPy array interface that points into memory
    ``base`` reads.
    """

    def __init__(self, base: np.ndarray, interface: dict[str, Any]):
        self.base = base
        self.__array_interface__ = interface


def _view_over_packed(arr: np.ndarray) -> Optional["Array"]:
    """An Array that reads ``arr`` as a view of a packed parent, or ``None`` to copy.

    ``arr`` is ``(H, W)`` or ``(H, W, C)``. The parent is built over the same
    memory from ``arr``'s strides, so it works whatever layout ``arr`` was
    sliced from. Each parent pixel is ``pixel_size`` consecutive elements
    and holds every channel ``arr`` reads at one position. Parent rows are
    one ``arr`` row stride apart. The view reads every parent row and every
    ``col_step``-th parent pixel, reverses each axis whose stride is
    negative, and picks ``arr``'s channels out of each parent pixel.

    ``None`` when a stride is not a whole number of elements, an axis longer
    than 1 has stride 0, rows overlap, or the parent would reach outside the
    memory ``arr`` belongs to.

    The result is the slice: origin ``(0, 0)`` and a canvas the size of
    ``arr``. Later crops do not reach samples outside that slice.
    """
    itemsize = int(arr.dtype.itemsize)
    if any(int(stride) % itemsize for stride in arr.strides):
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    channels = int(arr.shape[2]) if arr.ndim == 3 else 1
    # An axis of length 1 is never stepped along, so its stride is ignored.
    # 0 below means "any step".
    row_elements = abs(int(arr.strides[0])) // itemsize if height > 1 else 0
    col_elements = abs(int(arr.strides[1])) // itemsize if width > 1 else 0
    channel_elements = abs(int(arr.strides[2])) // itemsize if channels > 1 else 1
    if (
        (height > 1 and row_elements == 0)
        or (width > 1 and col_elements == 0)
        or channel_elements == 0
    ):
        return None

    # The pixel size must divide both steps, because the engine needs the
    # row pitch to be a whole number of pixels. gcd(n, 0) is n.
    channel_span = (channels - 1) * channel_elements + 1
    common = gcd(row_elements, col_elements)
    if common == 0:
        pixel_size = channel_span
    else:
        pixel_size = next(
            (size for size in range(channel_span, common + 1) if common % size == 0),
            0,
        )
        if pixel_size == 0:
            return None
    col_step = col_elements // pixel_size if col_elements else 1
    parent_width = (width - 1) * col_step + 1
    row_pitch = row_elements if row_elements else parent_width * pixel_size
    if row_pitch < parent_width * pixel_size:
        return None

    # ``lowest`` reads the same samples with every stride non-negative, so its
    # data pointer is the lowest address ``arr`` reads.
    flipped = [int(stride) < 0 for stride in arr.strides]
    lowest = arr[tuple(slice(None, None, -1) if flip else slice(None) for flip in flipped)]
    owner_low, owner_high = byte_bounds(_base_chain(arr)[-1])
    extent = ((height - 1) * row_pitch + parent_width * pixel_size) * itemsize
    # ``first_channel`` is where ``arr``'s first channel sits in a parent
    # pixel. Any position works if the parent stays inside the owner.
    for first_channel in range(pixel_size - channel_span + 1):
        parent_start = int(lowest.ctypes.data) - first_channel * itemsize
        if owner_low <= parent_start and parent_start + extent <= owner_high:
            break
    else:
        return None

    if pixel_size == 1:
        shape: Tuple[int, ...] = (height, parent_width)
        strides: Tuple[int, ...] = (row_pitch * itemsize, itemsize)
    else:
        shape = (height, parent_width, pixel_size)
        strides = (row_pitch * itemsize, pixel_size * itemsize, itemsize)
    parent = np.asarray(
        _Memory(
            arr,
            {
                "version": 3,
                "shape": shape,
                "typestr": arr.dtype.str,
                "data": (parent_start, True),
                "strides": strides,
            },
        )
    )
    if not _is_direct_source(parent):
        return None

    rows = slice(None, None, -1) if flipped[0] else slice(None)
    cols = slice(None, None, -col_step) if flipped[1] else slice(None, None, col_step)
    if pixel_size == 1:
        key: tuple[Any, ...] = (rows, cols) if arr.ndim == 2 else (rows, cols, None)
    else:
        src_channels = [first_channel + i * channel_elements for i in range(channels)]
        if flipped[2]:
            src_channels.reverse()
        if src_channels == list(range(pixel_size)):
            key = (rows, cols)
        else:
            key = (rows, cols, src_channels)
    return Array(parent).view(key, oob_valid=False, reset_origin=True)



if TYPE_CHECKING:
    from .engines.graph import OpNode


class ElementType(StrEnum):
    """Closed element-type vocabulary for Array / graph / engine IR.

    String values match the native ``MuImgDType`` names and NumPy dtype names.
    Distinct from ``np.dtype`` (buffer descriptors) and TIFF ``TiffType``
    (tag wire formats).
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    UINT8 = "uint8"
    UINT16 = "uint16"

    @classmethod
    def _missing_(cls, value: object) -> ElementType | None:
        """Map a NumPy dtype or scalar type. Strings use the StrEnum values."""
        if isinstance(value, str):
            return None
        try:
            dtype = np.dtype(value)
            return next(
                (member for member in cls if dtype == np.dtype(member.value)),
                None,
            )
        except (TypeError, ValueError):
            return None

    @property
    def numpy_dtype(self) -> type:
        """NumPy scalar type for this element type (``np.float32``, …)."""
        return np.dtype(self.value).type

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return int(np.dtype(self.value).itemsize)


type ElementTypeLike = str | ElementType | np.dtype[Any] | type[np.generic]


@dataclass(frozen=True)
class ArrayMeta:
    dtype: ElementType
    # The NumPy shape: ``(N,)``, ``(H, W)``, or ``(H, W, C)``. A mono array
    # can be ``(H, W)`` or ``(H, W, 1)``.
    shape: Tuple[int, ...]
    # Buffer top-left in the shared canvas coordinate system, as (row, col).
    origin: Tuple[int, int] = (0, 0)
    # Rect in that same system: (x0, y0, width, height). A later view or
    # crop uses it. When this array is the whole canvas, (x0, y0) is
    # (origin col, origin row).
    canvas: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.shape)
        if not 1 <= len(shape) <= 3:
            raise ValueError("array must be (N,), (H,W) or (H,W,C)")
        if len(shape) == 3 and shape[2] < 1:
            raise ValueError(f"unsupported channel count: {shape[2]}")
        if min(shape[:2]) < 1:
            raise ValueError(f"shape dimensions must be at least 1, got {shape}")
        object.__setattr__(self, "shape", shape)

    @property
    def height(self) -> int:
        """Rows in the buffer. A 1D array is one row."""
        return 1 if self.is_1d else self.shape[0]

    @property
    def width(self) -> int:
        """Columns in the buffer. A 1D array's length is its width."""
        return self.shape[0] if self.is_1d else self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[2] if self.channel_axis else 1

    @property
    def channel_axis(self) -> bool:
        """True when the shape has a channel axis, including ``(H, W, 1)``."""
        return len(self.shape) == 3

    @property
    def is_1d(self) -> bool:
        return len(self.shape) == 1

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def buffer_shape(self) -> Tuple[int, ...]:
        """The shape of the buffer the engine reads and writes.

        It equals ``shape`` except for a 1D array, whose buffer is ``(1, N)``.
        """
        return (1, self.shape[0]) if self.is_1d else self.shape

    def copy(self, **changes: Any) -> "ArrayMeta":
        return replace(self, **changes)

    def with_size(
        self,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        channels: Optional[int] = None,
        ndim: Optional[int] = None,
        **changes: Any,
    ) -> "ArrayMeta":
        """A copy with a new height, width, channel count, or rank.

        Each size that is not given keeps this array's value. ``ndim``
        defaults to this array's rank, except that more than one channel
        always needs a channel axis. A 1D result must be one row and one
        channel. ``changes`` sets other fields, as in ``copy``.
        """
        height = self.height if height is None else height
        width = self.width if width is None else width
        channels = self.channels if channels is None else channels
        ndim = self.ndim if ndim is None else ndim
        if channels > 1 and ndim == 2:
            ndim = 3
        if ndim == 1:
            if height != 1 or channels != 1:
                raise ValueError(
                    "a 1D array has one row and one channel; got "
                    f"height={height}, channels={channels}"
                )
            shape: Tuple[int, ...] = (width,)
        elif ndim == 2:
            shape = (height, width)
        else:
            shape = (height, width, channels)
        return replace(self, shape=shape, **changes)


def meta_from_array(arr: np.ndarray) -> ArrayMeta:
    return _meta_from_shape(arr.shape, ElementType(arr.dtype))


def _require_scalar(value: Any, op: str) -> float:
    if isinstance(value, Array):
        raise TypeError(f"{op}: array–array arithmetic not supported")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{op}: RHS must be a scalar") from e


def _expand_pad(
    value: Any, name: str, axes: int, *, nonneg: bool = False
) -> list[int] | list[float]:
    """NumPy ``pad_width`` / ``constant_values`` broadcast to ``axes`` pairs.

    Returns ``[before, after]`` for each axis, flattened. An int is every
    side. A pair is ``(before, after)`` on every axis.
    """
    try:
        pairs = np.broadcast_to(np.asarray(value), (axes, 2))
    except ValueError:
        if axes == 3:
            detail = "((top, bottom), (left, right), (before, after))"
        elif axes == 1:
            detail = "((before, after),)"
        else:
            detail = "((top, bottom), (left, right))"
        raise ValueError(
            f"{name}: expected an int, a pair, or {detail}; got {value!r}"
        ) from None
    sides = [pair.item() for pair in pairs.flat]
    if nonneg and any(side < 0 for side in sides):
        raise ValueError(f"{name}: values must be non-negative; got {sides}")
    return sides


def _expand_spatial_pad(value: Any, name: str, *, nonneg: bool = False) -> list[int] | list[float]:
    """``[top, bottom, left, right]``. Channel axes are never included."""
    return _expand_pad(value, name, 2, nonneg=nonneg)


def _pad_width_has_channel_axis(pad_width: Any) -> bool:
    """True when ``pad_width`` is three ``(before, after)`` pairs.

    A bare int and a single pair stay spatial even on a rank-3 array.
    """
    if isinstance(pad_width, bool) or isinstance(
        pad_width, (int, np.integer, float, np.floating)
    ):
        return False
    if (
        isinstance(pad_width, (tuple, list))
        and len(pad_width) == 2
        and all(
            isinstance(side, (int, np.integer, float, np.floating))
            and not isinstance(side, bool)
            for side in pad_width
        )
    ):
        return False
    try:
        arr = np.asarray(pad_width)
    except (ValueError, TypeError):
        return False
    return arr.shape == (3, 2)


def _slice_span(slc: slice, length: int, name: str) -> Tuple[int, int, int]:
    """Return ``(start, size, step)`` for a 1-d slice on an axis of ``length``.

    ``start`` is the first sample. ``size`` is the sample count. ``step``
    is the source step, including -1.
    """
    start, stop, step = slc.indices(length)
    n = len(range(start, stop, step))
    if n == 0:
        raise ValueError(f"{name}: slice results in an empty dimension")
    return start, n, step


def _as_axis_int(value: Any, name: str) -> int:
    """Require a real integer for a view keyword (bool is not an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _wrap_channel_index(value: Any, channels: int) -> int:
    """NumPy wrap of one last-axis index into ``0 .. channels-1``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("channel index must be an int")
    raw = int(value)
    index = raw + channels if raw < 0 else raw
    if index < 0 or index >= channels:
        raise IndexError(
            f"index {value} is out of bounds for axis 2 with size {channels}"
        )
    return index


def _src_channels_from_last_axis(
    key: Any, channels: int
) -> Tuple[Optional[list[int]], Optional[bool]]:
    """Source channel list and rank-3 flag for a last-axis key.

    The list is ``None`` when it is identity ``0..C-1``. The flag is
    ``None`` to keep the parent's channel axis, ``False`` when an integer
    index drops the axis, and ``True`` when a slice or list of one channel
    keeps ``(H, W, 1)``.
    """
    rank_drop = isinstance(key, (int, np.integer)) and not isinstance(
        key, (bool, np.bool_)
    )
    if isinstance(key, slice):
        start, stop, step = key.indices(channels)
        src_channels = list(range(start, stop, step))
        if not src_channels:
            raise ValueError("channels: slice results in an empty dimension")
    else:
        if isinstance(key, np.ndarray):
            if key.dtype == np.bool_ or key.ndim != 1:
                raise TypeError("channel index must be a 1-d sequence of ints")
            values = key.tolist()
        elif isinstance(key, (list, tuple)):
            values = list(key)
        elif rank_drop:
            values = [int(key)]
        else:
            raise TypeError(
                "channel index must be an int, a slice, or a sequence of ints"
            )
        if not values:
            raise ValueError("channels: empty index")
        src_channels = [_wrap_channel_index(value, channels) for value in values]
    if src_channels == list(range(channels)):
        return None, False if rank_drop else None
    if rank_drop:
        return src_channels, False
    if len(src_channels) == 1:
        return src_channels, True
    return src_channels, None


def _expand_index_key(key: Any, ndim: int) -> Tuple[Tuple[Any, ...], bool]:
    """Expand a slice key to one entry per axis, plus a last-axis newaxis flag.

    Mono arrays are rank 2 ``(H, W)`` unless ``channel_axis`` is set, in
    which case they are ``(H, W, 1)``. Multi-channel arrays are rank 3
    ``(H, W, C)``. At most one Ellipsis is replaced with ``slice(None)``.
    ``None`` (``np.newaxis``) counts as a new axis, not an existing one.
    A trailing Ellipsis that fills no axes is dropped, as in NumPy
    (``arr[:, :, ...]`` on a 2-d array is ``arr[:, :]``). A short key such
    as ``t[:]`` or ``t[10:90]`` is padded on the right with ``slice(None)``.

    H and W must be slices. The last axis of a rank-3 array may be an int,
    a slice, or a sequence of ints. ``newaxis`` is only a new last axis on
    a rank-2 array (``t[:, :, None]``).
    """
    if key is Ellipsis:
        items: Tuple[Any, ...] = (Ellipsis,)
    elif isinstance(key, slice):
        items = (key,)
    elif isinstance(key, tuple):
        items = key
    else:
        raise TypeError("region must be a spatial slice or a tuple of slices")

    n_ellipsis = sum(item is Ellipsis for item in items)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        ellipsis_at = items.index(Ellipsis)
        n_explicit = sum(item is not Ellipsis and item is not None for item in items)
        n_fill = ndim - n_explicit
        if n_fill < 0:
            raise IndexError(f"too many indices for an array: {n_explicit}")
        items = items[:ellipsis_at] + (slice(None),) * n_fill + items[ellipsis_at + 1 :]

    newaxis = False
    if any(item is None for item in items):
        if (
            sum(item is None for item in items) != 1
            or items[-1] is not None
            or len(items) - 1 != ndim
            or ndim != 2
        ):
            raise IndexError(
                "newaxis is only supported as a new last axis on a (H, W) array"
            )
        items = items[:-1]
        newaxis = True
    else:
        if len(items) > ndim:
            raise IndexError(f"too many indices for an array: {len(items)}")
        if len(items) < ndim:
            items = items + (slice(None),) * (ndim - len(items))
    spatial_end = len(items) - 1 if ndim == 3 else len(items)
    if any(not isinstance(items[axis], slice) for axis in range(spatial_end)):
        raise TypeError(
            "region must be slice objects; integer axes and masks are not supported"
        )
    return items, newaxis


@dataclass(frozen=True)
class _Window:
    """Spatial box plus optional last-axis gather.

    ``left`` and ``top`` are the first sample. ``row_step`` and ``col_step``
    are the source steps, including -1. ``src_channels`` is ``None`` when
    the last axis is identity. ``channel_axis`` is ``None`` to keep the
    parent's flag, ``True`` to present ``C == 1`` as ``(H, W, 1)``, and
    ``False`` to drop that axis. ``left`` and ``top`` may be negative so a
    later view can reach back into the parent canvas.
    """

    left: int
    top: int
    width: int
    height: int
    row_step: int = 1
    col_step: int = 1
    src_channels: Optional[list[int]] = None
    channel_axis: Optional[bool] = None

    def is_full(self, meta: ArrayMeta) -> bool:
        return (
            self.left == 0
            and self.top == 0
            and self.width == meta.width
            and self.height == meta.height
            and self.row_step == 1
            and self.col_step == 1
            and self.src_channels is None
        )


def _window_from_slices(
    meta: ArrayMeta, rows: Any, cols: Any, channels: Any = None
) -> _Window:
    src_channels = None
    channel_axis = None
    if channels is not None:
        if len(meta.shape) < 3:
            raise IndexError("too many indices for a (H, W) array")
        src_channels, channel_axis = _src_channels_from_last_axis(
            channels, meta.channels
        )
    top, height, row_step = _slice_span(rows, meta.height, "rows")
    left, width, col_step = _slice_span(cols, meta.width, "cols")
    return _Window(
        left=left,
        top=top,
        width=width,
        height=height,
        row_step=row_step,
        col_step=col_step,
        src_channels=src_channels,
        channel_axis=channel_axis,
    )


def _window_from_key(meta: ArrayMeta, key: Any) -> _Window:
    items, newaxis = _expand_index_key(key, len(meta.shape))
    rows, cols, *rest = items
    window = _window_from_slices(meta, rows, cols, rest[0] if rest else None)
    if newaxis:
        return replace(window, channel_axis=True)
    return window


def _index_key_1d(key: Any) -> Tuple[slice, int, int]:
    """Split a key on a 1D array into its slice and the newaxis counts around it.

    Returns ``(cols, leading, trailing)``: ``leading`` and ``trailing`` are
    how many ``None`` entries come before and after the one axis. ``v[2:5]``
    is ``(slice(2, 5), 0, 0)``, ``v[None, :]`` is ``(slice(None), 1, 0)``,
    and ``v[:, None]`` is ``(slice(None), 0, 1)``.
    """
    if key is Ellipsis or key is None or isinstance(key, slice):
        items: Tuple[Any, ...] = (key,)
    elif isinstance(key, tuple):
        items = key
    else:
        raise TypeError("region must be a slice or a tuple of slices")
    n_ellipsis = sum(item is Ellipsis for item in items)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    explicit = [item for item in items if item is not None and item is not Ellipsis]
    if len(explicit) > 1:
        raise IndexError(
            "too many indices for array: array is 1-dimensional, "
            f"but {len(explicit)} were indexed"
        )
    if explicit:
        items = tuple(item for item in items if item is not Ellipsis)
    elif n_ellipsis:
        items = tuple(slice(None) if item is Ellipsis else item for item in items)
    else:
        items = items + (slice(None),)
    position = next(i for i, item in enumerate(items) if item is not None)
    cols = items[position]
    if not isinstance(cols, slice):
        raise TypeError(
            "region must be slice objects; integer axes and masks are not supported"
        )
    return cols, position, len(items) - position - 1


def _view_1d(
    array: "Array", region: Any, *, oob_valid: bool, reset_origin: bool
) -> "Array":
    """NumPy indexing on a 1D array, done as a view of its one-row buffer."""
    import muimage as mi

    cols, leading, trailing = _index_key_1d(region)
    row = _with_meta(array, array.meta.with_size(ndim=2))
    out = row.view((slice(None), cols), oob_valid=oob_valid, reset_origin=reset_origin)
    if (leading, trailing) == (0, 0):
        return _with_meta(out, out.meta.with_size(ndim=1))
    if (leading, trailing) == (1, 0):
        return out
    if (leading, trailing) == (1, 1):
        return out.view(np.s_[:, :, None])
    if (leading, trailing) == (0, 1):
        return mi.orientation(out, orientation=5)
    raise IndexError(
        "on a 1D array, newaxis is supported as v[None, :], v[:, None], "
        "or v[None, :, None]"
    )


def _window(
    meta: ArrayMeta,
    region: slice | tuple[Any, ...] | None,
    left: int | None,
    top: int | None,
    width: int | None,
    height: int | None,
) -> _Window:
    """Normalize a slice region or a keyword rect to a spatial box plus channels.

    ``width`` and ``height`` must be at least 1.
    """
    rect = (left, top, width, height)
    has_rect = any(v is not None for v in rect)
    if region is not None:
        if has_rect:
            raise TypeError("cannot mix a slice region with left, top, width, height")
        return _window_from_key(meta, region)
    if any(v is None for v in rect):
        raise TypeError("view requires a slice region or left, top, width, and height")
    left_i = _as_axis_int(left, "left")
    top_i = _as_axis_int(top, "top")
    width_i = _as_axis_int(width, "width")
    height_i = _as_axis_int(height, "height")
    if width_i < 1 or height_i < 1:
        raise ValueError(
            f"view: invalid box top={top_i} left={left_i} "
            f"width={width_i} height={height_i}"
        )
    return _Window(left=left_i, top=top_i, width=width_i, height=height_i)


def rot90(m: "Array", k: int = 1, axes: Tuple[int, int] = (0, 1)) -> "Array":
    """Rotate in the spatial plane. Same arguments as ``numpy.rot90``.

    180° is a canvas-keeping view. Quarter turns still go through
    orientation, because a slice cannot swap height and width.
    """
    if m.meta.is_1d:
        raise ValueError(f"Axes={tuple(axes)} out of range for array of ndim=1.")
    if tuple(axes) != (0, 1):
        raise ValueError("Array only supports rot90 in the spatial plane (axes=(0, 1)).")
    turns = int(k) % 4
    if turns == 0:
        return m
    if turns == 2:
        return m.view(np.s_[::-1, ::-1], oob_valid=True)
    import muimage as mi

    # k=1 is 90° CCW (TIFF 8); k=3 is 90° CW (6).
    return mi.orientation(m, orientation={1: 8, 3: 6}[turns])


def fliplr(m: "Array") -> "Array":
    """Flip left–right. Same as ``numpy.fliplr``.

    A canvas-keeping view, so a later crop can still reach the parent.
    """
    if m.meta.is_1d:
        raise ValueError("Input must be >= 2-d.")
    return m.view(np.s_[:, ::-1], oob_valid=True)


def flipud(m: "Array") -> "Array":
    """Flip up–down. Same as ``numpy.flipud``: on a 1D array, reverse it.

    A canvas-keeping view, so a later crop can still reach the parent.
    """
    if m.meta.is_1d:
        return m.view(np.s_[::-1], oob_valid=True)
    return m.view(np.s_[::-1, :], oob_valid=True)


def _meta_from_shape(shape: Any, dtype: ElementType) -> ArrayMeta:
    """Build whole-canvas meta for ``(N,)``, ``(H, W)``, or ``(H, W, C)``."""
    dims = tuple(shape) if np.iterable(shape) else (shape,)
    meta = ArrayMeta(dtype=dtype, shape=dims)
    return meta.copy(canvas=(0, 0, meta.width, meta.height))


def _with_meta(array: "Array", meta: ArrayMeta) -> "Array":
    """The same buffer or node as ``array``, presented with ``meta``.

    ``meta`` must describe the same buffer (same ``buffer_shape`` and dtype).
    """
    out = object.__new__(Array)
    out._meta = meta
    out._data = array._data
    out._node = array._node
    return out


def _retag_channel_axis(array: "Array", channel_axis: bool) -> "Array":
    """Present the same pixels with or without a size-1 channel axis.

    A source buffer is reshaped. A lazy array shares its node; realize
    allocates ``(H, W, 1)`` or ``(H, W)`` and the engine still writes one
    channel.
    """
    if array.meta.channel_axis == channel_axis:
        return array
    meta = array.meta.with_size(ndim=3 if channel_axis else 2)
    if array._data is not None and array.meta.channels == 1:
        return Array(np.reshape(array._data, meta.shape))
    return Array(_meta=meta, _node=array._node)


def _dtype_from_fill(fill_value: Any) -> ElementType:
    """Element type of ``fill_value`` when ``full(..., dtype=None)``."""
    if isinstance(fill_value, bool):
        raise TypeError("unsupported fill_value dtype: bool")
    if isinstance(fill_value, np.generic) or (
        isinstance(fill_value, np.ndarray) and fill_value.ndim == 0
    ):
        return ElementType(np.asarray(fill_value).dtype)
    if isinstance(fill_value, np.ndarray):
        try:
            return ElementType(fill_value.dtype)
        except ValueError:
            pass
    if isinstance(fill_value, (int, float, list, tuple)):
        return ElementType.FLOAT32
    raise TypeError(f"unsupported fill_value type: {type(fill_value).__name__}")


def _fill_samples(fill_value: Any, channels: int) -> list[float]:
    """Scalar or length-``channels`` vector as a flat f32 list."""
    arr = np.asarray(fill_value)
    if arr.ndim == 0:
        return [float(arr.item())]
    if arr.ndim != 1:
        raise ValueError("fill_value must be a scalar or a length-C vector")
    if arr.size == 1:
        return [float(arr.reshape(-1)[0])]
    if arr.size == channels:
        return [float(v) for v in arr]
    raise ValueError(
        f"fill_value length {arr.size} does not match channel count {channels}"
    )


def _emit_fill(meta: ArrayMeta, fill_value: Any) -> "Array":
    import muimage as mi

    return mi.fill(
        Array(_meta=meta),
        value=_fill_samples(fill_value, meta.channels),
    )


def zeros(shape: Any, dtype: ElementTypeLike = ElementType.FLOAT32) -> "Array":
    """Lazy array filled with 0. Same arguments as ``numpy.zeros`` for image rank."""
    return _emit_fill(_meta_from_shape(shape, ElementType(dtype)), 0)


def ones(shape: Any, dtype: ElementTypeLike = ElementType.FLOAT32) -> "Array":
    """Lazy array filled with 1. Same arguments as ``numpy.ones`` for image rank."""
    return _emit_fill(_meta_from_shape(shape, ElementType(dtype)), 1)


def full(shape: Any, fill_value: Any, dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy array filled with a constant. Same arguments as ``numpy.full`` for image rank."""
    et = ElementType(dtype) if dtype is not None else _dtype_from_fill(fill_value)
    return _emit_fill(_meta_from_shape(shape, et), fill_value)


def zeros_like(array: "Array", dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy zeros with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return zeros(array.shape, dtype=et)


def ones_like(array: "Array", dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy ones with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return ones(array.shape, dtype=et)


def full_like(array: "Array", fill_value: Any, dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy constant with ``array``'s shape. ``dtype`` defaults to the reference."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return full(array.shape, fill_value, dtype=et)


def _tile_reps(shape: Tuple[int, ...], reps: Any) -> tuple[int, int, int]:
    """Right-align ``reps`` onto ``shape`` and return ``(row, col, channel)``.

    An int is the last axis. Missing leading counts are 1. A longer tuple
    would add an axis. A rank-2 array has no channel axis, so that count
    stays 1. A 1D array has only a column count.
    """
    if np.iterable(reps):
        counts = tuple(operator.index(value) for value in reps)
    else:
        counts = (operator.index(reps),)
    if len(counts) > len(shape):
        raise ValueError(f"tile reps {reps!r} would add an axis on shape {shape}")
    if any(count < 1 for count in counts):
        raise ValueError(f"tile reps must be >= 1, got {reps!r}")
    if len(shape) == 1:
        return 1, counts[0], 1
    row, col, channel = (1,) * (len(shape) - len(counts)) + counts + (1,) * (3 - len(shape))
    return row, col, channel


def tile(array: "Array | np.ndarray", reps: Any) -> "Array":
    """Repeat ``array`` like ``numpy.tile`` for an image-rank array.

    Unlike NumPy, a count of 0 and ``reps`` longer than the array rank
    (which would add a leading axis) raise ``ValueError``.
    """
    from .engines.graph import op

    array = Array(array)
    is_1d = array.meta.is_1d
    if np.iterable(reps):
        reps = tuple(reps)
    reps_count = len(reps) if isinstance(reps, tuple) else 1
    if is_1d and reps_count == 3:
        raise ValueError(
            f"tile reps {reps!r} would make a 1D array's length the channel count, "
            "which is not supported yet"
        )
    if is_1d and reps_count == 2:
        # NumPy promotes (N,) to (1, N), which is this array's buffer.
        array = _with_meta(array, array.meta.with_size(ndim=2))
    row_reps, col_reps, channel_reps = _tile_reps(array.shape, reps)
    return op(
        "tile",
        array,
        row_reps=row_reps,
        col_reps=col_reps,
        channel_reps=channel_reps,
    )


class Array:
    """Lazy array handle: a source buffer and/or an engine op result.

    Do not mutate an array after wrapping it. Ingest seals the array and its
    ndarray ``.base`` root. ``realize()`` caches pixels on this handle.
    """

    __slots__ = ("_meta", "_data", "_node")

    def __new__(
        cls,
        data: Optional[np.ndarray | Array] = None,
        *,
        origin: Optional[Tuple[int, int]] = None,
        _meta: Optional[ArrayMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        if isinstance(data, Array):
            if origin is not None:
                raise ValueError("origin= is only valid when ingesting a source buffer")
            if _meta is not None or _node is not None:
                raise ValueError("Array(Array) cannot take _meta= or _node=")
            return data
        return super().__new__(cls)

    def __init__(
        self,
        data: Optional[np.ndarray | Array] = None,
        *,
        origin: Optional[Tuple[int, int]] = None,
        _meta: Optional[ArrayMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        if isinstance(data, Array):
            return
        if data is not None:
            if _node is not None:
                raise ValueError("source Array cannot also have an op node")
            arr = np.asarray(data)
            meta = meta_from_array(arr)
            if origin is not None:
                row, col = int(origin[0]), int(origin[1])
                meta = meta.copy(
                    origin=(row, col),
                    canvas=(col, row, meta.width, meta.height),
                )
            # A 1D (N,) ndarray is bound as one row. Adding a leading axis of
            # length 1 is always a view, so this never copies. Other arrays
            # are not reshaped, because sealing must mark the caller's array.
            if meta.is_1d:
                arr = arr.reshape(meta.buffer_shape)
            data_out: np.ndarray | None
            if _is_direct_source(arr):
                data_out, node = _seal_ndarray(arr), None
            else:
                viewed = _view_over_packed(arr)
                if viewed is not None:
                    _seal_ndarray(arr)
                    data_out, node = viewed._data, viewed._node
                else:
                    # "A" copies a misaligned array too. A one-row array
                    # can be C-contiguous and still start on a byte that is
                    # not a multiple of its item size.
                    packed = np.require(arr, requirements=["C", "A"])
                    data_out, node = _seal_ndarray(packed), None
            self._meta = meta
            self._data = data_out
            self._node = node
        elif _meta is not None:
            if origin is not None:
                raise ValueError("origin= is only valid for source Arrays")
            self._meta = _meta
            self._data = None
            self._node = _node
        else:
            raise ValueError("Array requires an ndarray source or _meta=")

    @property
    def dtype(self) -> ElementType:
        return self._meta.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._meta.shape

    @property
    def meta(self) -> ArrayMeta:
        return self._meta

    def __sub__(self, other: Any) -> "Array":
        from .engines.graph import op

        value = _require_scalar(other, "sub_scalar")
        return op("sub_scalar", self, value=value)

    def __mul__(self, other: Any) -> "Array":
        from .engines.graph import op

        value = _require_scalar(other, "mul_scalar")
        return op("mul_scalar", self, value=value)

    def view(
        self,
        region: slice | tuple[Any, ...] | None = None,
        *,
        left: int | None = None,
        top: int | None = None,
        width: int | None = None,
        height: int | None = None,
        oob_valid: bool = True,
        reset_origin: bool = False,
    ) -> "Array":
        """Window into this array.

        ``oob_valid`` true keeps the parent canvas in this coordinate system.
        """
        import muimage as mi

        if self._meta.is_1d:
            if region is None:
                raise TypeError(
                    "a 1D array is viewed with a slice, not left, top, width, height"
                )
            return _view_1d(
                self, region, oob_valid=oob_valid, reset_origin=reset_origin
            )
        window = _window(self._meta, region, left, top, width, height)
        if window.is_full(self._meta):
            if (
                window.channel_axis is None
                or window.channel_axis == self._meta.channel_axis
            ):
                return self
            return _retag_channel_axis(self, window.channel_axis)

        attrs: dict[str, Any] = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
            "oob_valid": oob_valid,
            "reset_origin": reset_origin,
        }
        if window.src_channels is not None:
            attrs["src_channels"] = window.src_channels
        if window.row_step != 1:
            attrs["row_step"] = window.row_step
        if window.col_step != 1:
            attrs["col_step"] = window.col_step

        out = mi.view(self, **attrs)
        if (
            window.channel_axis is not None
            and out.meta.channel_axis != window.channel_axis
        ):
            out = Array(
                _meta=out.meta.with_size(ndim=3 if window.channel_axis else 2),
                _node=out._node,
            )
        return out

    def crop(
        self,
        region: slice | tuple[Any, ...] | None = None,
        *,
        left: int | None = None,
        top: int | None = None,
        width: int | None = None,
        height: int | None = None,
        reset_origin: bool = False,
    ) -> "Array":
        """Hard window: same as ``view(..., oob_valid=False)``."""
        return self.view(
            region,
            left=left,
            top=top,
            width=width,
            height=height,
            oob_valid=False,
            reset_origin=reset_origin,
        )

    def pad(
        self,
        pad_width: Any,
        mode: str = "constant",
        constant_values: Any = 0,
    ) -> "Array":
        """Grow the array. ``pad_width`` and ``constant_values`` follow ``numpy.pad``.

        A bare int or a two-axis width grows height and width only. On an
        array whose shape is ``(H, W, C)``, a third pair ``(before, after)``
        adds constant channels. That channel pad requires ``mode="constant"``.
        On a 1D array, ``pad_width`` is one ``(before, after)`` pair, as in NumPy.
        """
        import muimage as mi

        if self._meta.is_1d:
            left, right = (
                int(v) for v in _expand_pad(pad_width, "pad_width", 1, nonneg=True)
            )
            consts = [float(v) for v in _expand_pad(constant_values, "constant_values", 1)]
            return mi.pad(
                self,
                top=0,
                bottom=0,
                left=left,
                right=right,
                mode=mode,
                constant_values=[0.0, 0.0] + consts,
            )
        if _pad_width_has_channel_axis(pad_width):
            if len(self.shape) != 3:
                raise ValueError(
                    f"pad_width has 3 axes but array shape is {self.shape}"
                )
            sides = _expand_pad(pad_width, "pad_width", 3, nonneg=True)
            top, bottom, left, right, channel_before, channel_after = (
                int(v) for v in sides
            )
            consts = [
                float(v) for v in _expand_pad(constant_values, "constant_values", 3)
            ]
            if (channel_before or channel_after) and mode != "constant":
                raise ValueError(
                    f"pad: a channel pad requires mode 'constant', got {mode!r}"
                )
            attrs: dict[str, Any] = {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "channel_before": channel_before,
                "channel_after": channel_after,
                "mode": mode,
                "constant_values": consts,
            }
            return mi.pad(self, **attrs)

        top, bottom, left, right = (
            int(v) for v in _expand_spatial_pad(pad_width, "pad_width", nonneg=True)
        )
        consts = [float(v) for v in _expand_spatial_pad(constant_values, "constant_values")]
        attrs = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "mode": mode,
            "constant_values": consts,
        }
        return mi.pad(self, **attrs)

    def __getitem__(self, key: Any) -> "Array":
        """NumPy spatial slice: a hard crop of this array."""
        if key is None:
            # crop reads region=None as "no region given".
            key = (None,)
        return self.crop(key)

    def transpose(self, *axes: Any) -> "Array":
        """Transpose the 2D spatial dimensions of the array.

        Accepts optional axes to match NumPy, but enforces 2D spatial remapping.
        A 1D array is returned unchanged, as in NumPy.
        """
        if len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):
            axes = tuple(axes[0])
        if self._meta.is_1d:
            if axes and axes != (0,):
                raise ValueError("axes don't match array")
            return self
        if axes and axes != (1, 0) and axes != (1, 0, 2):
            raise ValueError("Array only supports 2D spatial axis transposition.")
        import muimage as mi

        return mi.orientation(self, orientation=5)

    @property
    def T(self) -> "Array":
        """Spatial transpose. Same as ``transpose()``."""
        return self.transpose()

    def astype(self, dtype: ElementTypeLike) -> "Array":
        """Numeric cast. ``255`` uint8 becomes ``255.0`` float32."""
        import muimage as mi

        dest = ElementType(dtype)
        if dest is self.dtype:
            return self
        return mi.cast_dtype(self, dest_dtype=dest.value)

    def convert_type(
        self,
        dtype: ElementTypeLike,
        src_bits: int | None = None,
        dst_bits: int | None = None,
        clip_max: float | None = None,
    ) -> "Array":
        """Pixel-range convert. ``255`` uint8, ``65535`` uint16, and ``1.0`` float16 become ``1.0`` float32.

        If the values are already float but not yet in that 0..1 range, pass
        ``src_bits`` / ``dst_bits`` to name the integer range they still use.
        """
        import muimage as mi

        dest = ElementType(dtype)
        if (
            dest is self.dtype
            and src_bits is None
            and dst_bits is None
            and clip_max is None
        ):
            return self
        attrs: dict[str, Any] = {"dest_dtype": dest.value}
        if src_bits is not None:
            attrs["src_bits"] = int(src_bits)
        if dst_bits is not None:
            attrs["dst_bits"] = int(dst_bits)
        if clip_max is not None:
            attrs["clip_max"] = float(clip_max)
        return mi.convert_dtype(self, **attrs)

    def realize(self, *, force_recompute: bool = False) -> np.ndarray:
        """Run the graph if needed and return this array's pixels (read-only)."""
        from .engines.graph import realize

        return realize(self, force_recompute=force_recompute)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol: getting the array materializes the graph."""
        arr = self.realize()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)
