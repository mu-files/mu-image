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


def _orientation_out_meta(t: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    orientation = int(attrs["orientation"])
    # TIFF 6 / 8 (and mirror+rotate 5 / 7) swap width and height
    swap = orientation in (5, 6, 7, 8)
    return TensorMeta(
        dtype=t.meta.dtype,
        height=t.meta.width if swap else t.meta.height,
        width=t.meta.height if swap else t.meta.width,
        channels=t.meta.channels,
    )


@graph_op(out_meta=_orientation_out_meta)
def orientation_op(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply TIFF/EXIF orientation via OpenCV rotate (subset used by render)."""
    orientation = int(orientation)
    if orientation == 1:  # HORIZONTAL
        return arr
    if orientation == 6:  # ROTATE_90_CW
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 3:  # ROTATE_180
        return cv2.rotate(arr, cv2.ROTATE_180)
    if orientation == 8:  # ROTATE_270_CW
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    logger.warning(
        "Unsupported TIFF orientation code: %s; no rotation applied", orientation
    )
    return arr


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
