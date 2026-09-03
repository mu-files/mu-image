# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Tensor handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np


def _require_image_ndarray(arr: np.ndarray) -> None:
    """Accept only (H, W) or (H, W, C) with C in 1, 3, 4."""
    if arr.ndim == 2:
        return
    if arr.ndim == 3:
        channels = int(arr.shape[2])
        if channels not in (1, 3, 4):
            raise ValueError(f"unsupported channel count: {channels}")
        return
    raise ValueError("array must be (H,W) or (H,W,C)")


def _pixels_are_packed(arr: np.ndarray) -> bool:
    """True when samples in a row are adjacent (unit column stride)."""
    itemsize = int(arr.dtype.itemsize)
    if arr.ndim == 2:
        return int(arr.strides[1]) == itemsize
    channels = int(arr.shape[2])
    return (
        int(arr.strides[2]) == itemsize
        and int(arr.strides[1]) == itemsize * channels
    )


def _seal_ndarray(arr: np.ndarray) -> np.ndarray:
    """Mark ``arr`` and every ndarray along ``.base`` read-only. Do not copy."""
    arr = np.asarray(arr)
    cur: np.ndarray | None = arr
    seen: set[int] = set()
    while cur is not None and isinstance(cur, np.ndarray):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if cur.flags.writeable:
            cur.setflags(write=False)
        base = cur.base
        cur = base if isinstance(base, np.ndarray) else None
    return arr


def _ingest_ndarray(data: Any) -> np.ndarray:
    """Wrap ``data`` as a source buffer. Copy only if pixels in a row are not packed."""
    arr = np.asarray(data)
    _require_image_ndarray(arr)
    if not _pixels_are_packed(arr):
        arr = np.ascontiguousarray(arr)
    return _seal_ndarray(arr)

if TYPE_CHECKING:
    from .engines.graph import OpNode


class ElementType(StrEnum):
    """Closed element-type vocabulary for Tensor / graph / engine IR.

    String values match the native ``MuImgDType`` names and NumPy dtype names.
    Distinct from ``np.dtype`` (buffer descriptors) and TIFF ``TiffType``
    (tag wire formats).
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    UINT8 = "uint8"
    UINT16 = "uint16"

    @classmethod
    def lookup(cls, value: str) -> "ElementType":
        """Look up enum member by string value."""
        from .common import enum_from_string

        return enum_from_string(cls, value)

    @classmethod
    def from_numpy(cls, dtype: np.dtype | type) -> "ElementType":
        """Map a NumPy dtype / scalar type onto ``ElementType``."""
        key = np.dtype(dtype)
        try:
            return _ELEMENT_TYPE_FROM_NUMPY[key]
        except KeyError as e:
            raise TypeError(f"unsupported ndarray dtype: {dtype}") from e

    @classmethod
    def coerce(cls, value: Union[str, "ElementType", np.dtype, type]) -> "ElementType":
        """Normalize a string, ``ElementType``, or NumPy dtype to ``ElementType``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.lookup(value)
        return cls.from_numpy(value)

    @property
    def numpy(self) -> type:
        """NumPy scalar type for this element type."""
        return NUMPY_FROM_ELEMENT_TYPE[self]


NUMPY_FROM_ELEMENT_TYPE: dict[ElementType, type] = {
    ElementType.FLOAT32: np.float32,
    ElementType.FLOAT16: np.float16,
    ElementType.UINT8: np.uint8,
    ElementType.UINT16: np.uint16,
}

_ELEMENT_TYPE_FROM_NUMPY = {
    np.dtype(np_t): et for et, np_t in NUMPY_FROM_ELEMENT_TYPE.items()
}


@dataclass(frozen=True)
class TensorMeta:
    dtype: ElementType
    height: int
    width: int
    channels: int
    # Buffer top-left in the shared canvas coordinate system, as (row, col).
    origin: Tuple[int, int] = (0, 0)
    # Rect in that same system: (x0, y0, width, height). A later view or
    # crop uses it. When this tensor is the whole canvas, (x0, y0) is
    # (origin col, origin row).
    canvas: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)

    def copy(self, **changes: Any) -> "TensorMeta":
        return replace(self, **changes)


def meta_from_array(arr: np.ndarray) -> TensorMeta:
    _require_image_ndarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        channels = 1
    else:
        h, w, channels = arr.shape
    return TensorMeta(
        dtype=ElementType.from_numpy(arr.dtype),
        height=h,
        width=w,
        channels=channels,
        origin=(0, 0),
        canvas=(0, 0, w, h),
    )


def _require_scalar(value: Any, op: str) -> float:
    if isinstance(value, Tensor):
        raise TypeError(f"{op}: tensor–tensor arithmetic not supported")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{op}: RHS must be a scalar") from e


def _expand_spatial_pad(value: Any, name: str, *, nonneg: bool = False) -> list[int] | list[float]:
    """NumPy ``pad_width`` / ``constant_values`` → ``[top, bottom, left, right]``.

    An int is all four sides. A pair is ``(before, after)`` on both spatial
    axes. A 2×2 is ``((top, bottom), (left, right))``. Channel axes are
    never included.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        sides = [arr.item(), arr.item(), arr.item(), arr.item()]
    elif arr.ndim == 1 and arr.size == 1:
        n = arr.reshape(-1)[0].item()
        sides = [n, n, n, n]
    elif arr.ndim == 1 and arr.size == 2:
        before, after = (arr.flat[0].item(), arr.flat[1].item())
        sides = [before, after, before, after]
    elif arr.ndim == 2 and arr.shape == (2, 2):
        sides = [
            arr[0, 0].item(),
            arr[0, 1].item(),
            arr[1, 0].item(),
            arr[1, 1].item(),
        ]
    else:
        raise ValueError(
            f"{name}: expected an int, a pair, or ((top, bottom), (left, right)); "
            f"got {value!r}"
        )
    if nonneg and any(v < 0 for v in sides):
        raise ValueError(f"{name}: values must be non-negative; got {sides}")
    return sides


def _require_slice_indices(slc: slice) -> None:
    """Reject float (and other non-index) start/stop/step, as NumPy does."""
    for part in (slc.start, slc.stop, slc.step):
        if part is None:
            continue
        try:
            operator.index(part)
        except TypeError:
            raise TypeError(
                "slice indices must be integers or None or have an __index__ method"
            ) from None


def _slice_span(slc: slice, length: int, name: str) -> Tuple[int, int, bool]:
    """Return (start, size, reversed) for a 1-d slice on an axis of ``length``."""
    _require_slice_indices(slc)
    if slc.step not in (None, 1, -1):
        raise ValueError(f"{name}: slice step must be 1 or -1, got {slc.step}")
    start, stop, step = slc.indices(length)
    n = len(range(start, stop, step))
    if n == 0:
        raise ValueError(f"{name}: slice results in an empty dimension")
    if step == 1:
        return start, n, False
    return start - n + 1, n, True


def _full_channel_slice(slc: slice, channels: int) -> bool:
    _require_slice_indices(slc)
    if slc.step not in (None, 1):
        return False
    start, stop, step = slc.indices(channels)
    return start == 0 and stop == channels and step == 1


def _orientation_from_flips(flip_rows: bool, flip_cols: bool) -> int:
    """Map reversed slice axes to a TIFF orientation code.

    Identity is 1. A reversed column axis is 2 (left–right). A reversed row
    axis is 4 (up–down). Both reversed is 3 (180°).
    """
    if flip_rows and flip_cols:
        return 3
    if flip_rows:
        return 4
    if flip_cols:
        return 2
    return 1


def _as_axis_int(value: Any, name: str) -> int:
    """Require a real integer for a view keyword (bool is not an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _expand_index_key(key: Any, ndim: int) -> Tuple[slice, ...]:
    """Expand a slice key to one slice per axis of the tensor.

    Mono tensors are rank 2 ``(H, W)``. RGB/RGBA tensors are rank 3
    ``(H, W, C)``. At most one Ellipsis is replaced with ``slice(None)``.
    A trailing Ellipsis that fills no axes is dropped, as in NumPy
    (``arr[:, :, ...]`` on a 2-d array is ``arr[:, :]``). A short key such
    as ``t[:]`` or ``t[10:90]`` is padded on the right with ``slice(None)``.
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
        i = items.index(Ellipsis)
        n_fill = max(0, ndim - (len(items) - 1))
        items = items[:i] + (slice(None),) * n_fill + items[i + 1 :]

    if any(not isinstance(item, slice) for item in items):
        raise TypeError(
            "region must be slice objects; integer axes, "
            "masks, and newaxis are not supported"
        )
    if len(items) < ndim:
        items = items + (slice(None),) * (ndim - len(items))
    return items


def _window_from_slices(
    meta: TensorMeta, rows: slice, cols: slice, channels: Optional[slice] = None
) -> Tuple[int, int, int, int, int]:
    if channels is not None:
        # Mono is rank 2 (H, W), even when channels == 1 in metadata.
        if len(meta.shape) < 3:
            raise IndexError("too many indices for a (H, W) tensor")
        if not _full_channel_slice(channels, meta.channels):
            raise ValueError("channel subsets are not supported")
    top, height, flip_rows = _slice_span(rows, meta.height, "rows")
    left, width, flip_cols = _slice_span(cols, meta.width, "cols")
    return left, top, width, height, _orientation_from_flips(flip_rows, flip_cols)


def _window_from_key(meta: TensorMeta, key: Any) -> Tuple[int, int, int, int, int]:
    items = _expand_index_key(key, len(meta.shape))
    if len(items) == 1:
        return _window_from_slices(meta, items[0], slice(None))
    if len(items) == 2:
        return _window_from_slices(meta, items[0], items[1])
    if len(items) == 3:
        return _window_from_slices(meta, items[0], items[1], items[2])
    raise IndexError(f"too many indices for a tensor: {len(items)}")


def _window(
    meta: TensorMeta,
    region: slice | tuple[Any, ...] | None,
    left: int | None,
    top: int | None,
    width: int | None,
    height: int | None,
) -> Tuple[int, int, int, int, int]:
    """Normalize a slice region or a keyword rect to (left, top, width, height, orientation).

    ``orientation`` is TIFF 1 (identity) for a keyword rect or a forward slice.
    A reversed slice axis is 2 (fliplr), 4 (flipud), or 3 (both, 180°).
    ``left`` and ``top`` may be negative so a later view can reach back into
    the parent canvas. ``width`` and ``height`` must be at least 1.
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
    return left_i, top_i, width_i, height_i, 1


def rot90(m: "Tensor", k: int = 1, axes: Tuple[int, int] = (0, 1)) -> "Tensor":
    """Rotate in the spatial plane. Same arguments as ``numpy.rot90``."""
    from . import mc

    if tuple(axes) != (0, 1):
        raise ValueError("Tensor only supports rot90 in the spatial plane (axes=(0, 1)).")
    turns = int(k) % 4
    if turns == 0:
        return m
    # k=1 is 90° CCW (TIFF 8); k=2 is 180 (3); k=3 is 90° CW (6).
    return mc.orientation(m, orientation={1: 8, 2: 3, 3: 6}[turns])


def fliplr(m: "Tensor") -> "Tensor":
    """Flip left–right. Same as ``numpy.fliplr``."""
    from . import mc

    return mc.orientation(m, orientation=2)


def flipud(m: "Tensor") -> "Tensor":
    """Flip up–down. Same as ``numpy.flipud``."""
    from . import mc

    return mc.orientation(m, orientation=4)


class Tensor:
    """Lazy tensor handle: a source buffer and/or an engine op result.

    Do not mutate an array after wrapping it. Ingest seals the array and its
    ndarray ``.base`` root. ``realize()`` caches pixels on this handle.
    """

    __slots__ = ("_meta", "_data", "_node")

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        *,
        origin: Optional[Tuple[int, int]] = None,
        _meta: Optional[TensorMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        if data is not None:
            if _node is not None:
                raise ValueError("source Tensor cannot also have an op node")
            arr = _ingest_ndarray(data)
            meta = meta_from_array(arr)
            if origin is not None:
                row, col = int(origin[0]), int(origin[1])
                meta = replace(
                    meta,
                    origin=(row, col),
                    canvas=(col, row, meta.width, meta.height),
                )
            self._meta = meta
            self._data = arr
            self._node = None
        elif _meta is not None and _node is not None:
            if origin is not None:
                raise ValueError("origin= is only valid for source Tensors")
            self._meta = _meta
            self._data = None
            self._node = _node
        else:
            raise ValueError("Tensor requires an ndarray source or an op node")

    @property
    def dtype(self) -> ElementType:
        return self._meta.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._meta.shape

    @property
    def meta(self) -> TensorMeta:
        return self._meta

    def __sub__(self, other: Any) -> "Tensor":
        from .engines.graph import op

        value = _require_scalar(other, "sub_scalar")
        return op("sub_scalar", self, value=value)

    def __mul__(self, other: Any) -> "Tensor":
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
    ) -> "Tensor":
        """Window into this tensor.

        ``oob_valid`` true keeps the parent canvas in this coordinate system.
        """
        from . import mc

        # _window resolves the geometry and checks for mutual exclusivity errors
        left_i, top_i, width_i, height_i, orientation = _window(
            self._meta, region, left, top, width, height
        )

        attrs: dict[str, Any] = {
            "left": left_i,
            "top": top_i,
            "width": width_i,
            "height": height_i,
            "oob_valid": oob_valid,
            "reset_origin": reset_origin,
        }

        out = mc.view(self, **attrs)
        if orientation != 1:
            out = mc.orientation(out, orientation=orientation)
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
    ) -> "Tensor":
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
    ) -> "Tensor":
        """Grow height and width. Same ``pad_width`` / ``constant_values`` shapes as ``numpy.pad``."""
        from . import mc

        top, bottom, left, right = (
            int(v) for v in _expand_spatial_pad(pad_width, "pad_width", nonneg=True)
        )
        consts = [float(v) for v in _expand_spatial_pad(constant_values, "constant_values")]
        attrs: dict[str, Any] = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "mode": mode,
            "constant_values": consts,
        }
        return mc.pad(self, **attrs)

    def __getitem__(self, key: Any) -> "Tensor":
        """NumPy spatial slice: a hard crop of this tensor."""
        return self.crop(key)

    def transpose(self, *axes: Any) -> "Tensor":
        """Transpose the 2D spatial dimensions of the tensor.

        Accepts optional axes to match NumPy, but enforces 2D spatial remapping.
        """
        if len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):
            axes = tuple(axes[0])
        if axes and axes != (1, 0) and axes != (1, 0, 2):
            raise ValueError("Tensor only supports 2D spatial axis transposition.")
        from . import mc

        return mc.orientation(self, orientation=5)

    @property
    def T(self) -> "Tensor":
        """Spatial transpose. Same as ``transpose()``."""
        return self.transpose()

    def realize(self, *, force_recompute: bool = False) -> np.ndarray:
        """Run the graph if needed and return this tensor's pixels (read-only)."""
        from .engines.graph import realize

        return realize(self, force_recompute=force_recompute)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol: getting the array materializes the graph."""
        arr = self.realize()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)
