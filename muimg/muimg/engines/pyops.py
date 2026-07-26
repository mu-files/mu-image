# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Python graph ops: write eager NumPy, call on Tensor to attach lazily.

Decorated with ``@graph_op`` — not in ``ops.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..tensor import NUMPY_FROM_DTYPE, Tensor, TensorMeta
from .graph import graph_op


def _crop_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    x, y, w, h = int(attrs["x"]), int(attrs["y"]), int(attrs["w"]), int(attrs["h"])
    if x < 0 or y < 0 or w < 1 or h < 1:
        raise ValueError(f"crop_op: invalid box x={x} y={y} w={w} h={h}")
    if y + h > t.meta.height or x + w > t.meta.width:
        raise ValueError(
            f"crop_op: box x={x} y={y} w={w} h={h} out of bounds for "
            f"{t.meta.height}x{t.meta.width}"
        )
    return TensorMeta(
        dtype=t.meta.dtype,
        height=h,
        width=w,
        channels=t.meta.channels,
    )


@graph_op(out_meta=_crop_out_meta)
def crop_op(arr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Packed ROI at origin ``(x, y)`` with size ``(w, h)``."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    out = np.ascontiguousarray(arr[y : y + h, x : x + w, ...])
    if out.size == 0:
        raise ValueError(f"crop_op: empty result x={x} y={y} w={w} h={h}")
    return out


def _cast_dtype_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    dest = attrs["dest_dtype"]
    if dest not in NUMPY_FROM_DTYPE:
        raise ValueError(f"cast_dtype_op: unsupported dest_dtype {dest!r}")
    return TensorMeta(
        dtype=dest,
        height=t.meta.height,
        width=t.meta.width,
        channels=t.meta.channels,
    )


@graph_op(out_meta=_cast_dtype_out_meta)
def cast_dtype_op(arr: np.ndarray, dest_dtype: str) -> np.ndarray:
    """Bit-cast / widen with no rescale (unlike engine ``convert_dtype``)."""
    np_dtype = NUMPY_FROM_DTYPE.get(dest_dtype)
    if np_dtype is None:
        raise ValueError(f"cast_dtype_op: unsupported dest_dtype {dest_dtype!r}")
    return np.ascontiguousarray(arr.astype(np_dtype, copy=False))


def _demosaic_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    if t.meta.channels != 1:
        raise ValueError("demosaic_op input must be mono / CFA (1 channel)")
    dest = attrs.get("return_dtype")
    dtype = dest if isinstance(dest, str) and dest in NUMPY_FROM_DTYPE else t.meta.dtype
    return TensorMeta(
        dtype=dtype,
        height=t.meta.height,
        width=t.meta.width,
        channels=3,
    )


@graph_op(out_meta=_demosaic_out_meta)
def demosaic_op(
    arr: np.ndarray,
    cfa_pattern: str,
    algorithm: str = "OPENCV_EA",
    clip_max: Optional[float] = None,
    return_dtype: Optional[str] = None,
) -> np.ndarray:
    """Non-bilinear demosaic. For bilinear use ``engines.ops.bilinear_demosaic``."""
    from ..raw_render import DemosaicAlgorithm, demosaic

    if algorithm == "DNGSDK_BILINEAR":
        raise ValueError(
            "demosaic_op does not run DNGSDK_BILINEAR; "
            "use engines.ops.bilinear_demosaic"
        )
    try:
        algo = DemosaicAlgorithm[algorithm]
    except KeyError as e:
        names = [a.name for a in DemosaicAlgorithm if a.name != "DNGSDK_BILINEAR"]
        raise ValueError(
            f"demosaic_op: unknown algorithm {algorithm!r}; expected one of {names}"
        ) from e

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    out_dtype = None
    if return_dtype is not None:
        out_dtype = NUMPY_FROM_DTYPE.get(return_dtype)
        if out_dtype is None:
            raise ValueError(f"demosaic_op: unsupported return_dtype {return_dtype!r}")

    out = demosaic(
        arr,
        cfa_pattern,
        algorithm=algo,
        clip_max=clip_max,
        return_dtype=out_dtype,
    )
    assert isinstance(out, np.ndarray)
    return np.ascontiguousarray(out)
