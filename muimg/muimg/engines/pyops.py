# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Python graph ops: write eager NumPy, call on Tensor to attach lazily.

Decorated with ``@graph_op`` — not in ``ops.yaml``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ..deps import cv2_proxy as cv2
from ..tensor import NUMPY_FROM_DTYPE, Tensor, TensorMeta
from .graph import graph_op

logger = logging.getLogger(__name__)


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
    dest = attrs["dst_dtype"]
    if dest not in NUMPY_FROM_DTYPE:
        raise ValueError(f"cast_dtype_op: unsupported dst_dtype {dest!r}")
    return TensorMeta(
        dtype=dest,
        height=t.meta.height,
        width=t.meta.width,
        channels=t.meta.channels,
    )


@graph_op(out_meta=_cast_dtype_out_meta)
def cast_dtype_op(arr: np.ndarray, dst_dtype: str) -> np.ndarray:
    """Bit-cast / widen with no rescale (unlike engine ``convert_dtype``)."""
    np_dtype = NUMPY_FROM_DTYPE.get(dst_dtype)
    if np_dtype is None:
        raise ValueError(f"cast_dtype_op: unsupported dst_dtype {dst_dtype!r}")
    return np.ascontiguousarray(arr.astype(np_dtype, copy=False))


def _demosaic_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    if t.meta.channels != 1:
        raise ValueError("demosaic_op input must be mono / CFA (1 channel)")
    algorithm = attrs.get("algorithm", "OPENCV_EA")
    # Working dtype is the algorithm's native output (= input after wrapper pre-convert)
    if algorithm == "RCD":
        dtype = "float32"
    elif algorithm == "VNG":
        dtype = "uint16"
    else:
        dtype = t.meta.dtype
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
) -> np.ndarray:
    """Non-bilinear demosaic kernel (ndarray in/out).

    Caller (``raw_render.demosaic``) emits pre/post ``convert_dtype`` neighbors.
    For bilinear use ``engines.ops.bilinear_demosaic``.
    """
    if algorithm == "DNGSDK_BILINEAR":
        raise ValueError(
            "demosaic_op does not run DNGSDK_BILINEAR; "
            "use engines.ops.bilinear_demosaic"
        )

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"demosaic_op: expected 2D CFA, got shape {arr.shape}")

    if algorithm == "RCD":
        try:
            from .. import _rcd
        except ImportError as e:
            raise ImportError(
                "RCD demosaicing is not available. RCD is GPL-licensed and must be "
                "enabled separately. See README.md for instructions to enable RCD, "
                "or use a different algorithm (VNG, OPENCV_EA, DNGSDK_BILINEAR)."
            ) from e
        if arr.dtype != np.float32:
            raise ValueError(f"RCD kernel requires float32 CFA, got {arr.dtype}")
        out = _rcd.rcd_demosaic(arr, cfa_pattern)

    elif algorithm == "VNG":
        from .. import _vng

        if arr.dtype != np.uint16:
            raise ValueError(f"VNG kernel requires uint16 CFA, got {arr.dtype}")
        out = _vng.vng_demosaic(arr, cfa_pattern)

    elif algorithm == "OPENCV_EA":
        bayer_map_bgr = {
            "RGGB": cv2.COLOR_BAYER_BG2RGB_EA,
            "BGGR": cv2.COLOR_BAYER_RG2RGB_EA,
            "GRBG": cv2.COLOR_BAYER_GB2RGB_EA,
            "GBRG": cv2.COLOR_BAYER_GR2RGB_EA,
        }
        if arr.dtype not in (np.uint8, np.uint16):
            raise ValueError(
                f"OPENCV_EA kernel requires uint8/uint16 CFA, got {arr.dtype}"
            )
        if cfa_pattern not in bayer_map_bgr:
            raise ValueError(f"Unsupported CFA pattern for OPENCV_EA: {cfa_pattern}")
        out = cv2.demosaicing(arr, bayer_map_bgr[cfa_pattern])

    else:
        raise ValueError(
            f"demosaic_op: unknown algorithm {algorithm!r}; "
            "expected one of ['VNG', 'RCD', 'OPENCV_EA']"
        )

    return np.ascontiguousarray(out)
