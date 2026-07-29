# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Tensor handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)


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
    )


def _require_scalar(value: Any, op: str) -> float:
    if isinstance(value, Tensor):
        raise TypeError(f"{op}: tensor–tensor arithmetic not supported")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{op}: RHS must be a scalar") from e


class Tensor:
    """Lazy tensor handle: either a concrete source buffer or an engine op result."""

    __slots__ = ("_meta", "_data", "_node")

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        *,
        _meta: Optional[TensorMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        if data is not None:
            if _node is not None:
                raise ValueError("source Tensor cannot also have an op node")
            arr = np.asarray(data)
            self._meta = meta_from_array(arr)
            self._data = arr
            self._node = None
        elif _meta is not None and _node is not None:
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
