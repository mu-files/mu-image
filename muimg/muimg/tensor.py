# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Tensor handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np

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
    if arr.ndim == 2:
        h, w = arr.shape
        channels = 1
    elif arr.ndim == 3:
        h, w, channels = arr.shape
        if channels not in (1, 3, 4):
            raise ValueError(f"unsupported channel count: {channels}")
    else:
        raise ValueError("array must be (H,W) or (H,W,C)")
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


def _slice_span(slc: slice, length: int, name: str) -> Tuple[int, int]:
    """Return (start, size) for a 1-d slice on an axis of ``length``."""
    if slc.step not in (None, 1):
        raise ValueError(f"{name}: slice step must be 1, got {slc.step}")
    start, stop, step = slc.indices(length)
    if step != 1:
        raise ValueError(f"{name}: slice step must be 1, got {step}")
    return start, stop - start


def _full_channel_slice(slc: slice, channels: int) -> bool:
    if slc.step not in (None, 1):
        return False
    start, stop, step = slc.indices(channels)
    return start == 0 and stop == channels and step == 1


def _window_from_slices(
    meta: TensorMeta, rows: slice, cols: slice, channels: Optional[slice] = None
) -> Tuple[int, int, int, int]:
    if channels is not None:
        if meta.channels == 1:
            raise IndexError("too many indices for a (H, W) tensor")
        if not _full_channel_slice(channels, meta.channels):
            raise ValueError("channel subsets are not supported")
    top, height = _slice_span(rows, meta.height, "rows")
    left, width = _slice_span(cols, meta.width, "cols")
    return left, top, width, height


def _window_from_key(meta: TensorMeta, key: Any) -> Tuple[int, int, int, int]:
    if isinstance(key, slice):
        return _window_from_slices(meta, key, slice(None))
    if not isinstance(key, tuple):
        raise TypeError("region must be a spatial slice or a tuple of slices")
    if any(not isinstance(item, slice) for item in key):
        raise TypeError(
            "region must be slice objects; integer axes, "
            "masks, and newaxis are not supported"
        )
    if len(key) == 1:
        return _window_from_slices(meta, key[0], slice(None))
    if len(key) == 2:
        return _window_from_slices(meta, key[0], key[1])
    if len(key) == 3:
        return _window_from_slices(meta, key[0], key[1], key[2])
    raise IndexError(f"too many indices for a tensor: {len(key)}")


def _window(
    meta: TensorMeta,
    region: slice | tuple[Any, ...] | None,
    left: int | None,
    top: int | None,
    width: int | None,
    height: int | None,
) -> Tuple[int, int, int, int]:
    """Normalize a slice region or a keyword rect to (left, top, width, height)."""
    rect = (left, top, width, height)
    has_rect = any(v is not None for v in rect)
    if region is not None:
        if has_rect:
            raise TypeError("cannot mix a slice region with left, top, width, height")
        return _window_from_key(meta, region)
    if any(v is None for v in rect):
        raise TypeError("view requires a slice region or left, top, width, and height")
    return int(left), int(top), int(width), int(height)


def rot90(m: "Tensor", k: int = 1, axes: Tuple[int, int] = (0, 1)) -> "Tensor":
    """Rotate in the spatial plane. Same arguments as ``numpy.rot90``."""
    from .engines import ops as engine_ops

    if tuple(axes) != (0, 1):
        raise ValueError("Tensor only supports rot90 in the spatial plane (axes=(0, 1)).")
    turns = int(k) % 4
    if turns == 0:
        return m
    # k=1 is 90° CCW (TIFF 8); k=2 is 180 (3); k=3 is 90° CW (6).
    return engine_ops.orientation(m, orientation={1: 8, 2: 3, 3: 6}[turns])


def fliplr(m: "Tensor") -> "Tensor":
    """Flip left–right. Same as ``numpy.fliplr``."""
    from .engines import ops as engine_ops

    return engine_ops.orientation(m, orientation=2)


def flipud(m: "Tensor") -> "Tensor":
    """Flip up–down. Same as ``numpy.flipud``."""
    from .engines import ops as engine_ops

    return engine_ops.orientation(m, orientation=4)


class Tensor:
    """Lazy tensor handle: either a concrete source buffer or an engine op result."""

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
            arr = np.asarray(data)
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
        from .engines import ops as engine_ops

        # _window resolves the geometry and checks for mutual exclusivity errors
        left_i, top_i, width_i, height_i = _window(
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

        return engine_ops.view(self, **attrs)

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
        from .engines import ops as engine_ops

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
        return engine_ops.pad(self, **attrs)

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
        from .engines import ops as engine_ops

        return engine_ops.orientation(self, orientation=5)

    @property
    def T(self) -> "Tensor":
        """Spatial transpose. Same as ``transpose()``."""
        return self.transpose()

    def compute(self) -> np.ndarray:
        """Materialize this tensor (engine graph only)."""
        from .engines.graph import compute

        return compute(self)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol: getting the array materializes the graph."""
        arr = self.compute()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)
