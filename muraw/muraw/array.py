# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Array handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.lib.array_utils import byte_bounds


def _is_direct_source(arr: np.ndarray) -> bool:
    """True when the engine can bind ``arr`` itself as a source buffer.

    Pixels in a row are adjacent, the data pointer is element-aligned, and
    the row pitch is a whole number of elements that holds one packed row.
    It need not be a whole number of pixels: a padded RGB888 camera buffer
    can have a pitch that is not a multiple of 3 bytes. A negative pitch is a
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
    return row_pitch >= packed and row_pitch % itemsize == 0


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

    # The pixel size must divide the column step so the view reads whole
    # parent pixels. The row pitch only has to be a whole number of elements.
    channel_span = (channels - 1) * channel_elements + 1
    if col_elements == 0:
        pixel_size = channel_span
    else:
        pixel_size = next(
            (
                size
                for size in range(channel_span, col_elements + 1)
                if col_elements % size == 0
            ),
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


# Orientation code that transposes a source whose (rows, columns) were
# reversed: 5 is a plain transpose, 6 is np.rot90(m, 3), 7 is the
# transverse, and 8 is np.rot90(m).
_TRANSPOSE_ORIENTATION = {
    (False, False): 5,
    (True, False): 6,
    (True, True): 7,
    (False, True): 8,
}


def _transposed_source(arr: np.ndarray) -> Optional["Array"]:
    """``arr`` read as an orientation of its row/column swap, or ``None`` to copy.

    ``img.T``, ``img.transpose(1, 0, 2)``, ``np.rot90(img)`` and Fortran-order
    arrays swap the row and column strides. ``arr.swapaxes(0, 1)`` with its
    row and column flips undone is read as a direct source or a view, and
    one orientation op transposes it back and applies those flips, so
    ``Array(np.rot90(m))`` builds the same graph as ``mi.rot90(Array(m))``.
    An array with one row or one column is never transposed.
    """
    if int(arr.shape[0]) == 1 or int(arr.shape[1]) == 1:
        return None
    swapped = arr.swapaxes(0, 1)
    flipped = (int(swapped.strides[0]) < 0, int(swapped.strides[1]) < 0)
    unflipped = swapped[
        slice(None, None, -1) if flipped[0] else slice(None),
        slice(None, None, -1) if flipped[1] else slice(None),
    ]
    if _is_direct_source(unflipped):
        inner: Optional[Array] = Array(unflipped)
    else:
        inner = _view_over_packed(unflipped)
    if inner is None:
        return None

    import muimage as mi
    return mi.orientation(inner, orientation=_TRANSPOSE_ORIENTATION[flipped])



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
        """Map a NumPy dtype, scalar type, or dtype string such as ``"f4"``.

        An unsupported dtype raises ``ValueError`` with a message that says
        how to convert the input. No input is converted automatically.
        """
        supported = ", ".join(member.value for member in cls)
        dtype = None
        if value is not None:
            try:
                dtype = np.dtype(value)
            except TypeError:
                pass
        if dtype is None:
            raise ValueError(f"{value!r} is not a dtype. Supported dtypes: {supported}.")
        member = next((member for member in cls if dtype == np.dtype(member.value)), None)
        if member is not None:
            return member
        if dtype.kind == "f" and dtype.itemsize == 8:
            raise ValueError(
                "float64 is not supported; muimage computes in float32. "
                "Convert with arr.astype(np.float32)."
            )
        if not dtype.isnative and dtype.newbyteorder("=").name in supported.split(", "):
            raise ValueError(
                f"big-endian {dtype.str} is not supported. Convert to native byte order "
                'with arr.astype(arr.dtype.newbyteorder("=")).'
            )
        raise ValueError(f"dtype {dtype} is not supported. Supported dtypes: {supported}.")

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


def _expand_index_key(key: Any, ndim: int) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    """Split an index key into one entry per existing axis and the new axes.

    ``ndim`` is the NumPy rank: 1 for ``(N,)``, 2 for ``(H, W)``, 3 for
    ``(H, W, C)``. At most one Ellipsis is replaced with ``slice(None)``,
    and a short key such as ``t[:]`` or ``t[10:90]`` is padded on the right
    with ``slice(None)``, as in NumPy.

    Each ``None`` (``np.newaxis``) adds an axis of length 1. The second
    value is the position of each new axis in the result's shape, so
    ``(H, W)[:, None, :]`` gives ``(1,)`` and ``(N,)[None, :, None]`` gives
    ``(0, 2)``. The result may have at most 3 axes.

    The spatial axes must be slices. The last axis of a rank-3 array may be
    an int (which drops that axis), a slice, or a sequence of ints.
    """
    if key is Ellipsis or key is None or isinstance(key, slice):
        items: Tuple[Any, ...] = (key,)
    elif isinstance(key, tuple):
        items = key
    else:
        raise TypeError("region must be a spatial slice or a tuple of slices")

    n_ellipsis = sum(item is Ellipsis for item in items)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    n_explicit = sum(item is not Ellipsis and item is not None for item in items)
    if n_explicit > ndim:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, "
            f"but {n_explicit} were indexed"
        )
    fill = (slice(None),) * (ndim - n_explicit)
    if n_ellipsis == 1:
        ellipsis_at = items.index(Ellipsis)
        items = items[:ellipsis_at] + fill + items[ellipsis_at + 1 :]
    else:
        items = items + fill

    axes = []
    new_axes = []
    rank = 0
    for item in items:
        if item is None:
            new_axes.append(rank)
            rank += 1
            continue
        axes.append(item)
        if not isinstance(item, (int, np.integer)):
            rank += 1
    if rank > 3:
        raise IndexError(
            f"newaxis would give {rank} axes; an array has at most 3 (H, W, C)"
        )
    spatial = axes if ndim < 3 else axes[:2]
    if any(not isinstance(item, slice) for item in spatial):
        raise TypeError(
            "region must be slice objects; integer axes and masks are not supported"
        )
    return tuple(axes), tuple(new_axes)


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


def _window_from_rect(
    left: int | None,
    top: int | None,
    width: int | None,
    height: int | None,
) -> _Window:
    """The box given as ``left``, ``top``, ``width``, ``height`` keywords.

    ``width`` and ``height`` must be at least 1.
    """
    rect = (left, top, width, height)
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


def _as_shape(shape: Any) -> Tuple[int, ...]:
    """A ``shape`` argument as a tuple of ints. One int is a 1-tuple.

    ``ArrayMeta`` checks the rank and sizes, so every constructor rejects
    the same shapes with the same messages.
    """
    if isinstance(shape, (str, bytes, bool, np.bool_)):
        raise TypeError(f"shape must be an int or a sequence of ints, got {shape!r}")
    try:
        if isinstance(shape, (int, np.integer)):
            dims = (operator.index(shape),)
        else:
            dims = tuple(operator.index(dim) for dim in shape)
    except TypeError:
        raise TypeError(f"shape must be an int or a sequence of ints, got {shape!r}") from None
    ArrayMeta(dtype=ElementType.FLOAT32, shape=dims)
    return dims


def _meta_from_shape(shape: Tuple[int, ...], dtype: ElementType) -> ArrayMeta:
    """Build whole-canvas meta for ``(N,)``, ``(H, W)``, or ``(H, W, C)``."""
    meta = ArrayMeta(dtype=dtype, shape=tuple(shape))
    return meta.copy(canvas=(0, 0, meta.width, meta.height))


def _with_meta(array: "Array", meta: ArrayMeta) -> "Array":
    """The same buffer or node as ``array``, presented with ``meta``.

    ``meta`` must describe the same buffer (same ``buffer_shape`` and dtype).
    """
    out = object.__new__(Array)
    out._take_fields(array, meta)
    return out


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
    return _emit_fill(_meta_from_shape(_as_shape(shape), ElementType(dtype)), 0)


def ones(shape: Any, dtype: ElementTypeLike = ElementType.FLOAT32) -> "Array":
    """Lazy array filled with 1. Same arguments as ``numpy.ones`` for image rank."""
    return _emit_fill(_meta_from_shape(_as_shape(shape), ElementType(dtype)), 1)


def full(shape: Any, fill_value: Any, dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy array filled with a constant. Same arguments as ``numpy.full`` for image rank."""
    et = ElementType(dtype) if dtype is not None else _dtype_from_fill(fill_value)
    return _emit_fill(_meta_from_shape(_as_shape(shape), et), fill_value)


def zeros_like(array: "Array", dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy zeros with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return zeros(array.shape, dtype=et)


def ones_like(array: "Array", dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy ones with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return ones(array.shape, dtype=et)


def _emit_ramp(
    meta: ArrayMeta,
    start: list[float],
    step: list[float],
    stop: list[float] | None,
) -> "Array":
    """A 1D ``meta`` ramps along its columns; a 2D one down its rows."""
    from .engines.graph import op

    along = "columns" if len(meta.shape) == 1 else "rows"
    attrs: dict[str, Any] = {"along": along, "start": start, "step": step}
    if stop is not None:
        attrs["stop"] = stop
    return op("ramp", Array(_meta=meta), **attrs)


def arange(
    start: Any,
    stop: Any = None,
    step: Any = 1,
    *,
    dtype: ElementTypeLike = ElementType.FLOAT32,
) -> "Array":
    """Samples ``start + i * step`` for ``i`` below ``ceil((stop - start) / step)``.

    One argument is the stop, with start 0 and step 1, as in ``numpy.arange``.
    The samples are computed in float32, so they can differ from NumPy's
    float64 by a few float32 rounding steps. ``uint8`` and ``uint16``
    saturate where NumPy wraps. A step of 0 raises. An empty range raises:
    there are no empty images.
    """

    def require_number(value: Any, name: str) -> float:
        """A real Python or NumPy number. A sequence raises."""
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise TypeError(f"arange {name} must be a number")
        arr = np.asarray(value)
        if arr.ndim != 0 or arr.dtype.kind not in "iuf":
            raise TypeError(f"arange {name} must be a number")
        return float(arr)

    if stop is None:
        stop = start
        start = 0
    start_v = require_number(start, "start")
    stop_v = require_number(stop, "stop")
    step_v = require_number(step, "step")
    if step_v == 0.0:
        raise ZeroDivisionError("arange step must not be 0")
    count = ceil((stop_v - start_v) / step_v)
    if count < 1:
        raise ValueError(f"arange({start}, {stop}, {step}) is empty")
    element_type = ElementType(dtype)
    numpy_dtype = np.dtype(element_type.value)
    if np.issubdtype(numpy_dtype, np.integer):
        # NumPy stores start and start + step as the integer dtype, then
        # steps by their difference, so the step is an integer too.
        first, second = np.array([start_v, start_v + step_v]).astype(numpy_dtype)
        start_v = float(first)
        step_v = float(second) - start_v
    meta = _meta_from_shape((count,), element_type)
    return _emit_ramp(meta, [start_v], [step_v], None)


def _linspace_components(start: Any, stop: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Broadcast ``start`` and ``stop`` to one float64 vector each.

    Two scalars come back with shape ``(1,)`` and the caller still reports
    ``(N,)``. A length-1 sequence stays length 1 so the result is ``(N, 1)``.
    """
    if isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_)):
        raise TypeError("linspace start and stop must be numbers")
    start_arr = np.asarray(start)
    stop_arr = np.asarray(stop)
    if start_arr.dtype.kind not in "iuf" or stop_arr.dtype.kind not in "iuf":
        raise TypeError("linspace start and stop must be numbers")
    if start_arr.ndim > 1 or stop_arr.ndim > 1:
        raise ValueError("linspace start and stop must be scalars or 1D")
    try:
        start_b, stop_b = np.broadcast_arrays(start_arr, stop_arr)
    except ValueError as exc:
        raise ValueError("linspace start and stop must be the same length") from exc
    if start_b.ndim == 0:
        start_b = start_b.reshape(1)
        stop_b = stop_b.reshape(1)
    return (
        np.ascontiguousarray(start_b, dtype=np.float64),
        np.ascontiguousarray(stop_b, dtype=np.float64),
    )


def linspace(
    start: Any,
    stop: Any,
    num: Any = 50,
    *,
    endpoint: bool = True,
    dtype: ElementTypeLike = ElementType.FLOAT32,
    axis: int = 0,
) -> "Array":
    """``num`` samples from ``start`` to ``stop``, like ``numpy.linspace``.

    A scalar ``start`` and ``stop`` return ``(num,)``. A length-``C`` pair
    returns ``(num, C)``: row ``y``, column ``c`` runs from ``start[c]`` to
    ``stop[c]``. ``endpoint`` true writes ``stop`` into the last sample.
    ``num`` of 1 is ``start``. ``axis`` other than 0 raises. The samples are
    computed in float32, so they can differ from NumPy by a few float32
    rounding steps.
    """
    if isinstance(num, bool) or isinstance(axis, bool):
        raise TypeError("linspace num and axis must be ints")
    try:
        count = operator.index(num)
        axis_i = operator.index(axis)
    except TypeError as exc:
        raise TypeError("linspace num and axis must be ints") from exc
    if count < 1:
        raise ValueError(f"linspace num must be >= 1, got {count}")
    if axis_i != 0:
        raise ValueError(f"linspace axis must be 0, got {axis_i}")
    if not isinstance(endpoint, (bool, np.bool_)):
        raise TypeError("linspace endpoint must be a bool")
    start_v, stop_v = _linspace_components(start, stop)
    scalar = np.ndim(start) == 0 and np.ndim(stop) == 0
    if count == 1:
        step_v = np.zeros_like(start_v)
        stop_arg = None
    elif endpoint:
        step_v = (stop_v - start_v) / (count - 1)
        stop_arg = [float(value) for value in stop_v]
    else:
        step_v = (stop_v - start_v) / count
        stop_arg = None
    if scalar:
        meta = _meta_from_shape((count,), ElementType(dtype))
    else:
        meta = _meta_from_shape((count, int(start_v.shape[0])), ElementType(dtype))
    return _emit_ramp(
        meta,
        [float(value) for value in start_v],
        [float(value) for value in step_v],
        stop_arg,
    )


def full_like(array: "Array", fill_value: Any, dtype: ElementTypeLike | None = None) -> "Array":
    """Lazy constant with ``array``'s shape. ``dtype`` defaults to the reference."""
    et = array.dtype if dtype is None else ElementType(dtype)
    return full(array.shape, fill_value, dtype=et)


def _reshape_layout(array: "Array", shape: Tuple[int, ...]) -> "Array":
    """The same samples, read as ``shape``.

    When the height, width, and channel count stay the same, only the NumPy
    shape changes (``(N,)`` and ``(1, N)``, or ``(H, W)`` and
    ``(H, W, 1)``). The result shares this array's node and canvas, and a
    cached buffer is reshaped.

    Otherwise the engine sees new sizes. A source buffer is
    ``ndarray.reshape``, which is a view. A lazy array gets its own node, so
    the producer writes its buffer and the next op binds that buffer
    reshaped, with no copy.
    """
    from types import MappingProxyType

    from .engines.graph import OpNode

    if tuple(shape) == array.shape:
        return array
    new = _meta_from_shape(shape, array.dtype)
    if int(np.prod(array.shape)) != int(np.prod(new.shape)):
        raise ValueError(
            f"cannot reshape {array.shape} to {new.shape}: the sample count differs"
        )
    old = array.meta
    if (new.height, new.width, new.channels) == (old.height, old.width, old.channels):
        out = object.__new__(Array)
        out._take_fields(array, old.copy(shape=new.shape))
        if out._data is not None:
            out._data = np.reshape(out._data, out.meta.buffer_shape)
        return out
    row, col = old.origin
    meta = new.copy(origin=old.origin, canvas=(col, row, new.width, new.height))
    if array._node is None:
        assert array._data is not None
        return Array(np.reshape(array._data, meta.buffer_shape), origin=array.meta.origin)

    def reshape(samples: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(samples, shape)

    node = OpNode(
        op="reshape",
        inputs=(array,),
        attrs=MappingProxyType({"shape": meta.shape}),
        out_meta=meta,
        fn=reshape,
    )
    return Array(_meta=meta, _node=node)


def _repeated_axes(arr: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, ...]]]:
    """Strip axes that repeat one sample, or ``None`` when nothing does.

    An axis longer than 1 with stride 0 is a NumPy broadcast. Index 0 of
    that axis is the sample, and the axis stays so the rank does not change.
    The returned counts are the repeat of each axis, 1 where the stride
    was not 0.
    """
    if arr.ndim == 0 or not any(
        int(length) > 1 and int(stride) == 0
        for length, stride in zip(arr.shape, arr.strides)
    ):
        return None
    slices = []
    reps = []
    for length, stride in zip(arr.shape, arr.strides):
        length = int(length)
        if length > 1 and int(stride) == 0:
            slices.append(slice(0, 1))
            reps.append(length)
        else:
            slices.append(slice(None))
            reps.append(1)
    return arr[tuple(slices)], tuple(reps)


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
    """Repeat ``array`` like ``numpy.tile``.

    A count of 0 raises ``ValueError``. ``reps`` longer than the rank
    raises too, except 3 counts on a 1D array: NumPy promotes ``(N,)``
    to ``(1, 1, N)``, so the length becomes the channel count.
    """
    from .engines.graph import op

    array = Array(array)
    is_1d = array.meta.is_1d
    if isinstance(reps, (str, bytes)):
        raise TypeError(f"tile reps must be an int or a sequence of ints, got {reps!r}")
    if np.iterable(reps):
        reps = tuple(reps)
    reps_count = len(reps) if isinstance(reps, tuple) else 1
    if is_1d and reps_count == 3:
        # NumPy promotes (N,) to (1, 1, N), so the length becomes channels.
        array = _reshape_layout(array, (1, 1, array.shape[0]))
    elif is_1d and reps_count == 2:
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


def broadcast_to(array: "Array | np.ndarray", shape: Any) -> "Array":
    """Stretch size-1 axes to ``shape``, like ``numpy.broadcast_to``.

    Shapes are right-aligned. A missing leading axis counts as length 1.
    Each source axis must be 1 or already the destination length. The
    stretch is ``tile`` with those repeat counts. A source whose axes land
    on a different height, width, or channel count is reshaped first, as a
    view of the same samples. ``(8, 8, 3)`` to ``(1080, 1920, 3)`` raises;
    repeating a patch is ``tile``.
    """
    array = Array(array)
    dest = _as_shape(shape)
    src = array.shape
    if len(dest) < len(src):
        raise ValueError(
            f"cannot broadcast shape {src} to {dest}: the source has more axes"
        )
    padded = (1,) * (len(dest) - len(src)) + src
    reps = []
    for src_len, dest_len in zip(padded, dest):
        if src_len == dest_len:
            reps.append(1)
        elif src_len == 1:
            reps.append(dest_len)
        else:
            raise ValueError(f"cannot broadcast shape {src} to {dest}")
    if padded != src:
        array = _reshape_layout(array, padded)
    if all(count == 1 for count in reps):
        return array
    return tile(array, tuple(reps))


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
            repeated = _repeated_axes(arr)
            if repeated is not None:
                core, reps = repeated
                stretched = tile(Array(core, origin=origin), reps)
                self._take_fields(stretched, stretched._meta)
                return
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
            if not _is_direct_source(arr):
                viewed = _view_over_packed(arr)
                if viewed is not None:
                    _seal_ndarray(arr)
                    self._take_fields(viewed, meta)
                    return
                transposed = _transposed_source(arr)
                if transposed is not None:
                    _seal_ndarray(arr)
                    self._take_fields(transposed, meta)
                    return
                # "A" copies a misaligned array too. A one-row array can be
                # C-contiguous and still start on a byte that is not a
                # multiple of its item size.
                arr = np.require(arr, requirements=["C", "A"])
            self._meta = meta
            self._data = _seal_ndarray(arr)
            self._node = None
        elif _meta is not None:
            if origin is not None:
                raise ValueError("origin= is only valid for source Arrays")
            self._meta = _meta
            self._data = None
            self._node = _node
        else:
            raise ValueError("Array requires an ndarray source or _meta=")

    def _take_fields(self, other: "Array", meta: ArrayMeta) -> None:
        """Make this handle ``other``'s buffer and node, presented with ``meta``.

        Every place that builds one handle from another goes through here,
        so a new slot is copied in one place.
        """
        self._meta = meta
        self._data = other._data
        self._node = other._node

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
        A ``None`` in ``region`` adds an axis of length 1, as in NumPy.
        """
        if region is None:
            if self._meta.is_1d:
                raise TypeError(
                    "a 1D array is viewed with a slice, not left, top, width, height"
                )
            window = _window_from_rect(left, top, width, height)
            return self._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
        if any(v is not None for v in (left, top, width, height)):
            raise TypeError("cannot mix a slice region with left, top, width, height")
        axes, new_axes = _expand_index_key(region, self._meta.ndim)
        if self._meta.is_1d:
            row = _reshape_layout(self, (1, self._meta.width))
            window = _window_from_slices(row._meta, slice(None), axes[0])
            viewed = row._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
            shape = [viewed.meta.width]
        else:
            window = _window_from_slices(self._meta, *axes)
            viewed = self._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
            shape = list(viewed.shape)
        for position in new_axes:
            shape.insert(position, 1)
        return _reshape_layout(viewed, tuple(shape))

    def _view_window(
        self, window: _Window, *, oob_valid: bool, reset_origin: bool
    ) -> "Array":
        """The ``view`` op for ``window``, or this array when it is all of it."""
        import muimage as mi

        if window.is_full(self._meta):
            if (
                window.channel_axis is None
                or window.channel_axis == self._meta.channel_axis
            ):
                return self
            ndim = 3 if window.channel_axis else 2
            return _reshape_layout(self, self._meta.with_size(ndim=ndim).shape)

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
            ndim = 3 if window.channel_axis else 2
            out = _reshape_layout(out, out.meta.with_size(ndim=ndim).shape)
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
