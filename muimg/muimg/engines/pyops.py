# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Python graph ops: write eager NumPy, call on Tensor to attach lazily.

Decorated with ``@graph_op`` — not in ``ops.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..deps import cv2_proxy as cv2
from ..tensor import ElementType, Tensor, TensorMeta
from .graph import graph_op


def _cast_dtype_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    dest = ElementType.coerce(attrs["dst_dtype"])
    return TensorMeta(
        dtype=dest,
        height=t.meta.height,
        width=t.meta.width,
        channels=t.meta.channels,
        origin=t.meta.origin,
    )


@graph_op(out_meta=_cast_dtype_out_meta)
def cast_dtype_op(arr: np.ndarray, dst_dtype: str | ElementType) -> np.ndarray:
    """Bit-cast / widen with no rescale (unlike engine ``convert_dtype``)."""
    dest = ElementType.coerce(dst_dtype)
    return np.ascontiguousarray(arr.astype(dest.numpy, copy=False))


def _demosaic_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    if t.meta.channels != 1:
        raise ValueError("demosaic_op input must be mono / CFA (1 channel)")
    algorithm = attrs.get("algorithm", "VNG")
    # Working dtype is the algorithm's native output (= input after wrapper pre-convert)
    if algorithm == "RCD":
        dtype = ElementType.FLOAT32
    elif algorithm == "VNG":
        dtype = ElementType.UINT16
    else:
        dtype = t.meta.dtype
    return TensorMeta(
        dtype=dtype,
        height=t.meta.height,
        width=t.meta.width,
        channels=3,
        origin=t.meta.origin,
    )


@graph_op(out_meta=_demosaic_out_meta)
def demosaic_op(
    arr: np.ndarray,
    cfa_pattern: str,
    algorithm: str = "VNG",
) -> np.ndarray:
    """Non-bilinear demosaic kernel (ndarray in/out).

    Caller (``raw_render.demosaic``) emits pre/post ``convert_dtype`` neighbors.
    For bilinear use ``engines.ops.bilinear_demosaic``.
    For Hamilton–Adams (``EA`` / ``EA_FAST``) use ``engines.ops.ea_demosaic``.
    ``OPENCV_EA`` stays here for quality comparison against the native EA path.
    """
    if algorithm == "DNGSDK_BILINEAR":
        raise ValueError(
            "demosaic_op does not run DNGSDK_BILINEAR; "
            "use engines.ops.bilinear_demosaic"
        )
    if algorithm in ("EA", "EA_FAST"):
        raise ValueError(
            "demosaic_op does not run EA / EA_FAST; "
            "use engines.ops.ea_demosaic"
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
                "or use a different algorithm (VNG, OPENCV_EA, DNGSDK_BILINEAR, EA, EA_FAST)."
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


@graph_op
def channel_luts_op(
    arr: np.ndarray,
    lut_r: np.ndarray,
    lut_g: np.ndarray,
    lut_b: np.ndarray,
) -> np.ndarray:
    """Apply independent 1D LUTs to R, G, B planes (float [0, 1] domain)."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"channel_luts_op: expected HxWx3, got {arr.shape}")
    out = np.empty_like(arr, dtype=np.float32)
    for i, lut in enumerate((lut_r, lut_g, lut_b)):
        lut = np.asarray(lut, dtype=np.float32).reshape(-1)
        x = np.linspace(0.0, 1.0, len(lut), dtype=np.float32)
        out[:, :, i] = np.interp(
            arr[:, :, i].astype(np.float32, copy=False), x, lut
        )
    return out


@graph_op
def radial_distortion_op(
    arr: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    scale_factor: float = 1.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
    focal_length_mm: float = 0.0,
    sensor_width_mm: float = 35.6,
) -> np.ndarray:
    """Adobe LCP ray-space radial distortion correction (cv2.remap)."""
    if focal_length_mm is None or float(focal_length_mm) <= 0.0:
        raise ValueError("focal_length_mm is required for radial distortion correction")

    height, width = arr.shape[:2]
    cx = float(center_x) * width
    cy = float(center_y) * height

    x_coords, y_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    focal_length_norm = float(focal_length_mm) / float(sensor_width_mm)
    norm_scale = max(width, height)
    x_norm = ((x_coords - cx) / norm_scale) / focal_length_norm
    y_norm = ((y_coords - cy) / norm_scale) / focal_length_norm

    r_squared = x_norm**2 + y_norm**2
    distortion_factor = (
        1.0
        + float(k1) * r_squared
        + float(k2) * r_squared**2
        + float(k3) * r_squared**3
    )
    scale = float(scale_factor)
    x_distorted = cx + scale * (x_coords - cx) * distortion_factor
    y_distorted = cy + scale * (y_coords - cy) * distortion_factor

    return cv2.remap(
        arr,
        x_distorted,
        y_distorted,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
