import cv2
import numpy as np
import logging

from typing import Dict
from . import _vng, _rcd

# DNG color C extension (provides color temp conversion, tone curves, HueSatMap, etc.)
from . import _dng_color

logger = logging.getLogger(__name__)

# Adobe DNG temperature/tint conversion (exact port of dng_temperature.cpp)
# Source: /3dparty/dng_sdk_1_7_1/dng_sdk/source/dng_temperature.cpp
_K_TINT_SCALE = -3000.0  # kTintScale

# Table from Wyszecki & Stiles, "Color Science" (same values as kTempTable in DNG SDK)
# Each tuple: (r, u, v, t)
_K_TEMP_TABLE = [
    (0.0,   0.18006, 0.26352,  -0.24341),
    (10.0,  0.18066, 0.26589,  -0.25479),
    (20.0,  0.18133, 0.26846,  -0.26876),
    (30.0,  0.18208, 0.27119,  -0.28539),
    (40.0,  0.18293, 0.27407,  -0.30470),
    (50.0,  0.18388, 0.27709,  -0.32675),
    (60.0,  0.18494, 0.28021,  -0.35156),
    (70.0,  0.18611, 0.28342,  -0.37915),
    (80.0,  0.18740, 0.28668,  -0.40955),
    (90.0,  0.18880, 0.28997,  -0.44278),
    (100.0, 0.19032, 0.29326,  -0.47888),
    (125.0, 0.19462, 0.30141,  -0.58204),
    (150.0, 0.19962, 0.30921,  -0.70471),
    (175.0, 0.20525, 0.31647,  -0.84901),
    (200.0, 0.21142, 0.32312,  -1.0182),
    (225.0, 0.21807, 0.32909,  -1.2168),
    (250.0, 0.22511, 0.33439,  -1.4512),
    (275.0, 0.23247, 0.33904,  -1.7298),
    (300.0, 0.24010, 0.34308,  -2.0637),
    (325.0, 0.24702, 0.34655,  -2.4681),
    (350.0, 0.25591, 0.34951,  -2.9641),
    (375.0, 0.26400, 0.35200,  -3.5814),
    (400.0, 0.27218, 0.35407,  -4.3633),
    (425.0, 0.28039, 0.35577,  -5.3762),
    (450.0, 0.28863, 0.35714,  -6.7262),
    (475.0, 0.29685, 0.35823,  -8.5955),
    (500.0, 0.30505, 0.35907, -11.3240),
    (525.0, 0.31320, 0.35968, -15.6280),
    (550.0, 0.32129, 0.36011, -23.3250),
    (575.0, 0.32931, 0.36038, -40.7700),
    (600.0, 0.33724, 0.36051,-116.4500),
]

def uv_to_colortemp(u: float, v: float) -> tuple[float, float]:
    """Exact port of dng_temperature::Set_xy_coord but accepting CIE 1960 (u,v).

    Given CIE 1960 UCS coordinates (u, v), compute correlated color temperature (Kelvin)
    and tint (Adobe's scale). Returns (temperature, tint).

    Note: CIE 1960 (u, v) is considered an "obsolete" color space; see:
    https://en.wikipedia.org/wiki/CIE_1960_color_space
    """

    last_dt = 0.0
    last_du = 0.0
    last_dv = 0.0

    # Using indices like C++: 1..30 inclusive; kTempTable has at least 31 entries (0..30)
    temp = 0.0
    tint = 0.0
    for index in range(1, 31):
        # Convert slope to delta-u and delta-v, with length 1.
        du = 1.0
        dv = _K_TEMP_TABLE[index][3]
        length = float(np.sqrt(1.0 + dv * dv))
        du /= length
        dv /= length

        # Delta from black body point to test coordinate
        uu = u - _K_TEMP_TABLE[index][1]
        vv = v - _K_TEMP_TABLE[index][2]

        # Distance above or below line
        dt = -uu * dv + vv * du

        # If below line, we have found line pair
        if dt <= 0.0 or index == 30:
            if dt > 0.0:
                dt = 0.0
            dt = -dt

            if index == 1:
                f = 0.0
            else:
                f = dt / (last_dt + dt) if (last_dt + dt) != 0.0 else 0.0

            # Interpolate the temperature (note: table stores r = 1e6/T)
            r_interp = (_K_TEMP_TABLE[index - 1][0] * f +
                        _K_TEMP_TABLE[index][0] * (1.0 - f))
            if r_interp == 0.0:
                temp = float('inf')
            else:
                temp = 1.0e6 / r_interp

            # Delta from interpolated black body point
            u_bb = (_K_TEMP_TABLE[index - 1][1] * f +
                    _K_TEMP_TABLE[index][1] * (1.0 - f))
            v_bb = (_K_TEMP_TABLE[index - 1][2] * f +
                    _K_TEMP_TABLE[index][2] * (1.0 - f))
            uu = u - u_bb
            vv = v - v_bb

            # Interpolate vectors along slope
            du = du * (1.0 - f) + last_du * f
            dv = dv * (1.0 - f) + last_dv * f
            length = float(np.sqrt(du * du + dv * dv))
            if length != 0.0:
                du /= length
                dv /= length

            # Distance along slope => tint
            tint = (uu * du + vv * dv) * _K_TINT_SCALE
            break

        # Try next line pair; carry forward state
        last_dt = dt
        last_du = du
        last_dv = dv

    return float(temp), float(tint)


def colortemp_to_uv(temperature: float, tint: float) -> tuple[float, float]:
    """Exact port of dng_temperature::Get_xy_coord but returning CIE 1960 (u,v).

    Given color temperature (Kelvin) and tint (Adobe's scale), compute CIE 1960 UCS (u, v).
    Returns (u, v).

    Note: CIE 1960 (u, v) is considered an "obsolete" color space; see:
    https://en.wikipedia.org/wiki/CIE_1960_color_space
    """
    # Inverse temperature to index table (r = 1e6 / T)
    r = 1.0e6 / float(temperature)

    # Convert tint to offset in uv space
    offset = float(tint) * (1.0 / _K_TINT_SCALE)

    u = 0.0
    v = 0.0
    for index in range(0, 30):
        if r < _K_TEMP_TABLE[index + 1][0] or index == 29:
            # Relative weight of first line
            denom = (_K_TEMP_TABLE[index + 1][0] - _K_TEMP_TABLE[index][0])
            f = ((_K_TEMP_TABLE[index + 1][0] - r) / denom) if denom != 0.0 else 0.0

            # Interpolate black body coordinates in uv
            u = _K_TEMP_TABLE[index][1] * f + _K_TEMP_TABLE[index + 1][1] * (1.0 - f)
            v = _K_TEMP_TABLE[index][2] * f + _K_TEMP_TABLE[index + 1][2] * (1.0 - f)

            # Find vectors along slope for each line and normalize
            uu1 = 1.0
            vv1 = _K_TEMP_TABLE[index][3]
            uu2 = 1.0
            vv2 = _K_TEMP_TABLE[index + 1][3]
            len1 = float(np.sqrt(1.0 + vv1 * vv1))
            len2 = float(np.sqrt(1.0 + vv2 * vv2))
            if len1 != 0.0:
                uu1 /= len1
                vv1 /= len1
            if len2 != 0.0:
                uu2 /= len2
                vv2 /= len2

            # Vector from black body point and normalize
            uu3 = uu1 * f + uu2 * (1.0 - f)
            vv3 = vv1 * f + vv2 * (1.0 - f)
            len3 = float(np.sqrt(uu3 * uu3 + vv3 * vv3))
            if len3 != 0.0:
                uu3 /= len3
                vv3 /= len3

            # Adjust coordinate along this vector by offset
            u += uu3 * offset
            v += vv3 * offset

            # Return u,v directly (xy conversion provided by helper if needed)
            return float(u), float(v)

    # Fallback (should not happen if table covers range)
    return float(u), float(v)


def uvUCS_to_xy(u: float, v: float) -> tuple[float, float]:
    """Convert CIE 1960 UCS (u, v) to CIE 1931 (x, y).

    Uses the exact formulas extracted from the DNG SDK method port.
    """
    denom_xy = (u - 4.0 * v + 2.0)
    x = 1.5 * u / denom_xy
    y = v / denom_xy
    return float(x), float(y)

def xy_to_uvUCS(x: float, y: float) -> tuple[float, float]:
    """Convert CIE 1931 (x, y) to CIE 1960 UCS (u, v).

    Uses the exact formulas extracted from the DNG SDK method port.
    """
    denom = (1.5 - x + 6.0 * y)
    u = 2.0 * x / denom
    v = 3.0 * y / denom
    return float(u), float(v)

# TODO: move this to muallsky_train which is the only client
def color_xfrm_image(image: np.ndarray, xfrm: np.ndarray) -> np.ndarray:
    """Apply a color transformation matrix to an image.
    """
    if image is None or xfrm is None:
        raise ValueError("image and xfrm must be non-None numpy arrays")
    if image.ndim < 1 or image.shape[-1] != 3:
        raise ValueError(f"image must have last dimension 3 (bands), got shape {image.shape}")
    xfrm = np.asarray(xfrm)
    if xfrm.shape != (3, 3):
        raise ValueError(f"xfrm must be a 3x3 matrix, got shape {xfrm.shape}")

    # Compute with float precision then cast back to original dtype
    in_dtype = image.dtype
    img_f = image.astype(np.float32, copy=False)
    # Apply transform with matrix multiplication on the last axis
    # result[..., c_out] = sum_c_in img[..., c_in] * xfrm[c_out, c_in]
    result = img_f @ xfrm.T
    return result.astype(in_dtype, copy=False)

def XYZ_to_YuvUCS(image: np.ndarray) -> np.ndarray:
    """Convert packed XYZ image/array to packed Y,u,v (CIE 1960 UCS).

    Input is a numpy array with last dimension of size 3 ordered as (X, Y, Z).
    Output has the same shape with last dimension (Y, u, v) where:
      u = 4X / (X + 15Y + 3Z)
      v = 6Y / (X + 15Y + 3Z)

    The dtype of the input is preserved in the output.
    """
    if image is None:
        raise ValueError("image must be a numpy array with last dimension 3 (X,Y,Z)")
    if image.ndim < 1 or image.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3 for (X,Y,Z), got shape {image.shape}")

    X = image[..., 0]
    Y = image[..., 1]
    Z = image[..., 2]

    denom = X + 15.0 * Y + 3.0 * Z
    # Avoid divide-by-zero; use small epsilon for denom==0
    eps = 1e-4
    safe = denom != 0
    denom_safe = np.where(safe, denom, eps)

    u = (4.0 * X) / denom_safe
    v = (6.0 * Y) / denom_safe

    # Pack as (Y, u, v)
    yuv = np.stack([Y, u, v], axis=-1)
    return yuv.astype(image.dtype, copy=False)

# =============================================================================
# DNG Illuminant Handling (CalibrationIlluminant1/2/3)
# =============================================================================

# Light source enum values from DNG SDK dng_tag_values.h
# Maps EXIF LightSource values to color temperature (Kelvin)
# SDK ref: dng_camera_profile.cpp IlluminantToTemperature()
ILLUMINANT_TO_TEMPERATURE = {
    0: None,      # lsUnknown - use default
    1: 5500.0,    # lsDaylight
    2: 4150.0,    # lsFluorescent (3800+4500)/2
    3: 2850.0,    # lsTungsten
    4: 5500.0,    # lsFlash
    9: 5500.0,    # lsFineWeather
    10: 6500.0,   # lsCloudyWeather
    11: 7500.0,   # lsShade
    12: 6400.0,   # lsDaylightFluorescent (5700+7100)/2
    13: 5050.0,   # lsDayWhiteFluorescent (4600+5500)/2
    14: 4150.0,   # lsCoolWhiteFluorescent (3800+4500)/2
    15: 3525.0,   # lsWhiteFluorescent (3250+3800)/2
    16: 2925.0,   # lsWarmWhiteFluorescent (2600+3250)/2
    17: 2850.0,   # lsStandardLightA
    18: 5500.0,   # lsStandardLightB
    19: 6500.0,   # lsStandardLightC
    20: 5500.0,   # lsD55
    21: 6500.0,   # lsD65
    22: 7500.0,   # lsD75
    23: 5000.0,   # lsD50
    24: 3200.0,   # lsISOStudioTungsten
    255: None,    # lsOther - requires IlluminantData
}


def illuminant_to_xy(illuminant: int) -> tuple[float, float] | None:
    """Convert DNG CalibrationIlluminant enum to CIE xy chromaticity.
    
    Uses the illuminant's color temperature to compute xy via the
    Planckian locus (same method as SDK).
    
    Args:
        illuminant: CalibrationIlluminant enum value (EXIF LightSource)
        
    Returns:
        Tuple of (x, y) chromaticity, or None if illuminant is unknown/other
    """
    temp = ILLUMINANT_TO_TEMPERATURE.get(illuminant)
    if temp is None:
        return None
    
    # Convert temperature to xy using our C++ implementation
    # (same as SDK's dng_temperature class)
    return _dng_color.temp_to_xy(temp, 0.0)


def illuminant_to_temperature(illuminant: int) -> float | None:
    """Convert DNG CalibrationIlluminant enum to color temperature (Kelvin).
    
    Args:
        illuminant: CalibrationIlluminant enum value (EXIF LightSource)
        
    Returns:
        Temperature in Kelvin, or None if illuminant is unknown/other
    """
    return ILLUMINANT_TO_TEMPERATURE.get(illuminant)


def interpolate_color_matrix(
    color_matrix1: np.ndarray,
    color_matrix2: np.ndarray,
    temp1: float,
    temp2: float,
    scene_temp: float
) -> np.ndarray:
    """Interpolate between two color matrices based on scene temperature.
    
    SDK ref: dng_color_spec.cpp FindXYZtoCamera_SingleOrDual() lines 301-345
    Uses inverse temperature weighting (mired space).
    
    Args:
        color_matrix1: ColorMatrix for illuminant 1 (lower temp, e.g. tungsten)
        color_matrix2: ColorMatrix for illuminant 2 (higher temp, e.g. daylight)
        temp1: Temperature of illuminant 1 in Kelvin
        temp2: Temperature of illuminant 2 in Kelvin
        scene_temp: Scene white point temperature in Kelvin
        
    Returns:
        Interpolated color matrix
    """
    # Ensure temp1 < temp2
    if temp1 > temp2:
        temp1, temp2 = temp2, temp1
        color_matrix1, color_matrix2 = color_matrix2, color_matrix1
    
    # Calculate interpolation weight using inverse temperature (mired space)
    if scene_temp <= temp1:
        g = 1.0
    elif scene_temp >= temp2:
        g = 0.0
    else:
        inv_t = 1.0 / scene_temp
        inv_t1 = 1.0 / temp1
        inv_t2 = 1.0 / temp2
        g = (inv_t - inv_t2) / (inv_t1 - inv_t2)
    
    # Interpolate matrices
    if g >= 1.0:
        return color_matrix1
    elif g <= 0.0:
        return color_matrix2
    else:
        return g * color_matrix1 + (1.0 - g) * color_matrix2


def apply_analog_balance(color_matrix: np.ndarray, analog_balance: np.ndarray) -> np.ndarray:
    """Apply AnalogBalance to color matrix.
    
    SDK ref: dng_color_spec.cpp constructor, line 179
    AnalogBalance is a diagonal matrix that scales each channel.
    
    Args:
        color_matrix: The color matrix (3x3 or 3xN)
        analog_balance: Per-channel scale factors (length 3)
        
    Returns:
        Scaled color matrix
    """
    # Create diagonal matrix from analog balance values
    ab_diag = np.diag(analog_balance)
    return ab_diag @ color_matrix


# =============================================================================
# DNG Tag Validation for process_raw()
# =============================================================================

# DNG tags that affect rendering but are NOT implemented in process_raw()
# Based on Adobe DNG SDK 1.7.1 dng_tag_codes.h
# These tags would cause our output to differ from the SDK reference
# Tag names match tifffile's tag name mapping
UNSUPPORTED_RENDERING_TAGS = {
    # Triple illuminant support (we support dual but not triple)
    "ColorMatrix3",
    "CalibrationIlluminant3",
    
    # Camera calibration matrices - CameraCalibration3 requires triple illuminant
    "CameraCalibration3",
    
    # Forward matrices - ForwardMatrix3 requires triple illuminant
    "ForwardMatrix3",
    
    # Reduction matrices
    "ReductionMatrix1",
    "ReductionMatrix2",
    "ReductionMatrix3",
    
    # Profile-based color adjustments
    "ProfileHueSatMapDims",
    "ProfileHueSatMapData1",
    "ProfileHueSatMapData2",
    "ProfileHueSatMapData3",
    "ProfileToneCurve",
    "ProfileLookTableDims",
    "ProfileLookTableData",
    "ProfileGainTableMap",
    "ProfileGainTableMap2",
    
    # Opcode lists (lens corrections, gain maps, etc.)
    "OpcodeList1",
    "OpcodeList2",
    "OpcodeList3",
    
    # Linearization (we assume linear response)
    "LinearizationTable",
    
    # Per-row/column black level variations
    "BlackLevelDeltaH",
    "BlackLevelDeltaV",
    
    # Linear response limit
    "LinearResponseLimit",
    
    # Baseline exposure offset
    "BaselineExposureOffset",
    
    # RGB Tables (DNG 1.6+)
    "RGBTables",
}

class UnsupportedDNGTagError(Exception):
    """Raised when a DNG file contains tags that process_raw() cannot handle."""
    pass


def validate_dng_tags(tags: dict, strict: bool = True) -> list[str]:
    """Check if DNG tags contain unsupported rendering features.
    
    Args:
        tags: Dictionary of DNG tags from the raw page (uses tifffile tag names)
        strict: If True, raise UnsupportedDNGTagError on finding unsupported tags.
                If False, return list of unsupported tag names.
    
    Returns:
        List of unsupported tag names found (empty if none)
        
    Raises:
        UnsupportedDNGTagError: If strict=True and unsupported tags are found
    """
    found_unsupported = []
    
    for tag_name in UNSUPPORTED_RENDERING_TAGS:
        if tag_name in tags and tags[tag_name] is not None:
            found_unsupported.append(tag_name)
    
    if found_unsupported and strict:
        raise UnsupportedDNGTagError(
            f"DNG contains unsupported rendering tags: {', '.join(found_unsupported)}. "
            f"process_raw() output will differ from SDK reference. "
            f"Use strict=False to process anyway."
        )
    
    return found_unsupported


def interp_center(img: np.ndarray) -> np.ndarray:
    """Interpolate each pixel from its 4 nearest neighbors (up, down, left, right).
    
    Args:
        img: Input image (H, W)
        
    Returns:
        Interpolated image where each pixel is the average of its 4 neighbors
    """
    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    
    # Interior pixels (not on edges)
    result[1:-1, 1:-1] = (
        img[0:-2, 1:-1].astype(np.float32) +  # up
        img[2:, 1:-1].astype(np.float32) +    # down
        img[1:-1, 0:-2].astype(np.float32) +  # left
        img[1:-1, 2:].astype(np.float32)      # right
    ) / 4.0
    
    # Handle edges and corners
    # Top edge (no up neighbor)
    result[0, 1:-1] = (
        img[1, 1:-1].astype(np.float32) +     # down
        img[0, 0:-2].astype(np.float32) +     # left
        img[0, 2:].astype(np.float32)         # right
    ) / 3.0
    
    # Bottom edge (no down neighbor)
    result[-1, 1:-1] = (
        img[-2, 1:-1].astype(np.float32) +    # up
        img[-1, 0:-2].astype(np.float32) +    # left
        img[-1, 2:].astype(np.float32)        # right
    ) / 3.0
    
    # Left edge (no left neighbor)
    result[1:-1, 0] = (
        img[0:-2, 0].astype(np.float32) +     # up
        img[2:, 0].astype(np.float32) +       # down
        img[1:-1, 1].astype(np.float32)       # right
    ) / 3.0
    
    # Right edge (no right neighbor)
    result[1:-1, -1] = (
        img[0:-2, -1].astype(np.float32) +    # up
        img[2:, -1].astype(np.float32) +      # down
        img[1:-1, -2].astype(np.float32)      # left
    ) / 3.0
    
    # Corners (2 neighbors each)
    result[0, 0] = (img[1, 0].astype(np.float32) + img[0, 1].astype(np.float32)) / 2.0  # top-left
    result[0, -1] = (img[1, -1].astype(np.float32) + img[0, -2].astype(np.float32)) / 2.0  # top-right
    result[-1, 0] = (img[-2, 0].astype(np.float32) + img[-1, 1].astype(np.float32)) / 2.0  # bottom-left
    result[-1, -1] = (img[-2, -1].astype(np.float32) + img[-1, -2].astype(np.float32)) / 2.0  # bottom-right
    
    return result.astype(img.dtype)

def interp_center_green(g: np.ndarray) -> np.ndarray:
    """Predict one green channel from the other by averaging 2x2 blocks.
    
    For Bayer pattern CFA, G1 and G2 are in a checkerboard pattern.
    To predict G1 from G2 (or vice versa), average each 2x2 block.
    
    Args:
        g: Input green channel (G2 to predict G1, or G1 to predict G2)
        
    Returns:
        Predicted green channel values
    """
    h, w = g.shape
    g_f = g.astype(np.float32)
    
    # Compute average of each 2x2 block using vectorized operations
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    
    # Average the 4 pixels in each 2x2 block
    avg_2x2 = (
        g_f[0:h_even:2, 0:w_even:2] +
        g_f[0:h_even:2, 1:w_even:2] +
        g_f[1:h_even:2, 0:w_even:2] +
        g_f[1:h_even:2, 1:w_even:2]
    ) / 4.0
    
    # Broadcast the averaged values back to the same size as input
    result = np.repeat(np.repeat(avg_2x2, 2, axis=0), 2, axis=1)
    
    # Handle odd dimensions if necessary
    if h > h_even:
        result = np.vstack([result, result[-1:, :]])
    if w > w_even:
        result = np.hstack([result, result[:, -1:]])
    
    return result.astype(g.dtype)

def interp_G1G2_plane(g1: np.ndarray, g2: np.ndarray, cfa_pattern: str = "RGGB") -> np.ndarray:
    """Assemble G1 and G2 planes into a full-resolution green image with interpolated R/B positions.
    
    Creates a 2x resolution image where:
    - G1 pixels are placed at their CFA positions
    - G2 pixels are placed at their CFA positions  
    - R/B positions are filled by averaging neighboring G pixels
    
    Args:
        g1: G1 plane (H/2, W/2) uint16
        g2: G2 plane (H/2, W/2) uint16
        cfa_pattern: Bayer pattern (default: "RGGB")
        
    Returns:
        Full resolution green image (H, W) uint16 with all positions filled
    """
    h_half, w_half = g1.shape
    h_full = h_half * 2
    w_full = w_half * 2
    
    # Create full resolution output
    green_full = np.zeros((h_full, w_full), dtype=np.float32)
    
    # Place G1 and G2 at their proper positions based on CFA pattern
    # RGGB: R at (0,0), G1 at (0,1), G2 at (1,0), B at (1,1)
    # BGGR: B at (0,0), G1 at (0,1), G2 at (1,0), R at (1,1)
    # GRBG: G1 at (0,0), R at (0,1), B at (1,0), G2 at (1,1)
    # GBRG: G2 at (0,0), B at (0,1), R at (1,0), G1 at (1,1)
    
    if cfa_pattern == "RGGB":
        green_full[0::2, 1::2] = g1  # G1 at (0,1)
        green_full[1::2, 0::2] = g2  # G2 at (1,0)
        # R at (0,0), B at (1,1) need interpolation
        r_rows, r_cols = 0, 0
        b_rows, b_cols = 1, 1
    elif cfa_pattern == "BGGR":
        green_full[0::2, 1::2] = g1  # G1 at (0,1)
        green_full[1::2, 0::2] = g2  # G2 at (1,0)
        # B at (0,0), R at (1,1) need interpolation
        r_rows, r_cols = 1, 1
        b_rows, b_cols = 0, 0
    elif cfa_pattern == "GRBG":
        green_full[0::2, 0::2] = g1  # G1 at (0,0)
        green_full[1::2, 1::2] = g2  # G2 at (1,1)
        # R at (0,1), B at (1,0) need interpolation
        r_rows, r_cols = 0, 1
        b_rows, b_cols = 1, 0
    elif cfa_pattern == "GBRG":
        green_full[0::2, 0::2] = g2  # G2 at (0,0)
        green_full[1::2, 1::2] = g1  # G1 at (1,1)
        # B at (0,1), R at (1,0) need interpolation
        r_rows, r_cols = 1, 0
        b_rows, b_cols = 0, 1
    else:
        raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")
    
    # Interpolate R and B positions: average of 4 neighboring G pixels (up, down, left, right)
    # Vectorized computation: average shifted versions of the green array
    # Create padded version to handle borders cleanly
    green_padded = np.pad(green_full, pad_width=1, mode='edge')
    
    # Compute average of 4 cardinal neighbors for all positions
    avg_neighbors = (
        green_padded[0:-2, 1:-1] +  # up
        green_padded[2:, 1:-1] +    # down
        green_padded[1:-1, 0:-2] +  # left
        green_padded[1:-1, 2:]      # right
    ) / 4.0
    
    # Place interpolated values at R positions
    green_full[r_rows::2, r_cols::2] = avg_neighbors[r_rows::2, r_cols::2]
    
    # Place interpolated values at B positions
    green_full[b_rows::2, b_cols::2] = avg_neighbors[b_rows::2, b_cols::2]
    
    return np.clip(green_full, 0, 65535).astype(np.uint16)

def fix_hot_pixels_channels(
    channels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    hot_candidates: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None
    ] = (None, None, None, None),
    threshold: float = 2.,
    min_brightness: int = 24000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fix hot pixels in individual color channels by replacing them.
    
    For each candidate hot pixel, checks if it's actually hot by comparing
    its value to interpolated neighbors. If hot, replaces it with the
    interpolated value.
    
    Args:
        channels: Tuple of (r, g1, g2, b) channel arrays (H/2, W/2) uint16
        hot_candidates: Tuple of (r, g1, g2, b) binary masks of hot pixel
            candidates (H/2, W/2)
        threshold: Multiplier threshold (pixel must be > threshold *
            interpolated)
        min_brightness: Minimum brightness to consider (default: 24000)
        
    Returns:
        Tuple of fixed (r, g1, g2, b) channels (H/2, W/2) uint16
    """
    # Unpack hot candidates
    hot_candidates_r, hot_candidates_g1, hot_candidates_g2, hot_candidates_b = (
        hot_candidates
    )
    
    # If no candidate maps provided, return original
    if all(c is None for c in [
        hot_candidates_r,
        hot_candidates_g1,
        hot_candidates_g2,
        hot_candidates_b
    ]):
        return tuple(ch.copy() for ch in channels)
    
    # Copy channels
    r, g1, g2, b = tuple(ch.copy() for ch in channels)
    
    # Compute channel medians for threshold scaling
    r_median = np.median(r)
    g1_median = np.median(g1)
    g2_median = np.median(g2)
    b_median = np.median(b)
    green_median = (g1_median + g2_median) / 2.0
    
    # Compute scale factors for R and B relative to green median
    r_scale = (
        r_median / green_median if green_median > 0 else 1.0
    )
    b_scale = (
        b_median / green_median if green_median > 0 else 1.0
    )
    
    # Helper to fix a single channel with scaled threshold
    def fix_channel(
        channel,
        channel_interp,
        hot_candidates,
        scale_factor=1.0
    ):
        if (hot_candidates is not None and
                np.count_nonzero(hot_candidates) > 0):
            scaled_min_brightness = min_brightness * scale_factor
            candidates_y, candidates_x = np.where(hot_candidates)
            for y, x in zip(candidates_y, candidates_x):
                if (channel[y, x] > scaled_min_brightness and
                        channel[y, x] > threshold * channel_interp[y, x]):
                    channel[y, x] = channel_interp[y, x]
    
    # Fix each channel with appropriate interpolation and scaling
    fix_channel(r, interp_center(r), hot_candidates_r, r_scale)
    fix_channel(g1, interp_center_green(g2), hot_candidates_g1, 1.0)
    fix_channel(g2, interp_center_green(g1), hot_candidates_g2, 1.0)
    fix_channel(b, interp_center(b), hot_candidates_b, b_scale)
    
    return (r, g1, g2, b)


def fix_hot_pixels(
    cfa: np.ndarray,
    cfa_pattern: str,
    hot_candidates: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None
    ] = (None, None, None, None),
    threshold: float = 2.,
    min_brightness: int = 24000
) -> np.ndarray:
    """Fix hot pixels in CFA by replacing them with interpolated values.
    
    For each candidate hot pixel, checks if it's actually hot by comparing
    its value to interpolated neighbors. If hot, replaces it with the
    interpolated value.
    
    Args:
        cfa: Raw CFA data (H, W) uint16
        cfa_pattern: Bayer pattern string (RGGB, BGGR, GRBG, or GBRG)
        hot_candidates: Tuple of (r, g1, g2, b) binary masks of hot pixel
            candidates (H/2, W/2)
        threshold: Multiplier threshold (pixel must be > threshold *
            interpolated)
        min_brightness: Minimum brightness to consider (default: 24000)
        
    Returns:
        Fixed CFA data (H, W) uint16
    """
    from . import dngio
    
    # If no candidate maps provided, return original
    if all(c is None for c in hot_candidates):
        return cfa.copy()
    
    # Extract channels
    channels = dngio.rgb_planes_from_cfa(cfa, cfa_pattern)
    
    # Fix hot pixels in channels
    fixed_channels = fix_hot_pixels_channels(
        channels,
        hot_candidates=hot_candidates,
        threshold=threshold,
        min_brightness=min_brightness
    )
    
    # Recompose CFA
    return dngio.cfa_from_rgb_planes(fixed_channels, cfa_pattern, cfa.shape)

def linear_raw_from_cfa(
    image_data: np.ndarray, 
    cfa_pattern: str, 
    orientation: int | None = None,
    algorithm: str = "RCD",
    hot_candidates: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None
    ] = (None, None, None, None),
    hot_pixel_threshold: float = 2.5,
    hot_pixel_min_brightness: int = 32768
) -> np.ndarray:
    """Demosaic CFA data to RGB with optional hot pixel correction.
    
    Args:
        image_data: 2D raw CFA data array (uint16)
        cfa_pattern: Bayer pattern string (RGGB, BGGR, GRBG, or GBRG)
        orientation: Optional EXIF orientation code (1, 3, 6, or 8)
        algorithm: Demosaic algorithm - "RCD" (default), "VNG", "LINEAR",
            "DCB", "AHD", or "OPENCV_EA"
        hot_candidates: Optional tuple of (r, g1, g2, b) binary masks of
            hot pixel candidates (H/2, W/2)
        hot_pixel_threshold: Multiplier threshold for hot pixel detection
            (default: 2.5)
        hot_pixel_min_brightness: Minimum brightness to consider for hot
            pixels (default: 32768)
        
    Returns:
        RGB array (uint16)
    """
    import io
    from . import dngio
    
    # Fix hot pixels before demosaicing if candidate maps provided
    if any(c is not None for c in hot_candidates):
        image_data = fix_hot_pixels(
            image_data,
            cfa_pattern,
            hot_candidates=hot_candidates,
            threshold=hot_pixel_threshold,
            min_brightness=hot_pixel_min_brightness
        )
    
    # Validate inputs
    if image_data.ndim != 2:
        raise ValueError(
            f"image_data must be a 2D array, but got {image_data.ndim} dimensions."
        )
    if image_data.size == 0:
        raise ValueError("image_data must not be empty.")
    
    if not isinstance(cfa_pattern, str):
        raise TypeError(
            f"cfa_pattern must be a string, but got {type(cfa_pattern).__name__}."
        )
    
    # Validate CFA pattern
    valid_patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]
    if cfa_pattern not in valid_patterns:
        raise ValueError(
            f"Invalid CFA pattern: '{cfa_pattern}'. "
            f"Supported patterns are: {valid_patterns}."
        )
    
    # Validate algorithm
    valid_algorithms = ["VNG", "RCD", "LINEAR", "DCB", "AHD", "OPENCV_EA"]
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"Invalid algorithm: '{algorithm}'. "
            f"Supported algorithms are: {valid_algorithms}."
        )
    
    # Demosaic using selected algorithm
    import time
    start_time = time.perf_counter()
    
    if algorithm == "OPENCV_EA":
        # OpenCV demosaic
        bayer_pattern_map = {
            "RGGB": cv2.COLOR_BAYER_RG2RGB_EA,
            "BGGR": cv2.COLOR_BAYER_BG2RGB_EA,
            "GRBG": cv2.COLOR_BAYER_GR2RGB_EA,
            "GBRG": cv2.COLOR_BAYER_GB2RGB_EA,
        }
        rgb = cv2.demosaicing(image_data, bayer_pattern_map[cfa_pattern])
        # OpenCV outputs BGR, swap to RGB
        rgb = rgb[..., [2, 1, 0]]
    elif algorithm == "VNG":
        # Native VNG implementation (thread-safe, ARM-optimized)
        rgb = _vng.vng_demosaic(image_data, cfa_pattern)
    elif algorithm == "RCD":
        # Native RCD implementation (Ratio Corrected Demosaicing)
        rgb = _rcd.rcd_demosaic(image_data, cfa_pattern)
    else:
        # Fallback to rawpy for other algorithms (LINEAR, DCB, AHD)
        # Note: rawpy is no longer a required dependency
        try:
            import rawpy
        except ImportError:
            raise ImportError(
                f"Algorithm '{algorithm}' requires rawpy. "
                "Install with: pip install rawpy\n"
                "Or use 'RCD', 'VNG' or 'OPENCV_EA' which are built-in."
            )
        
        algorithm_map = {
            "VNG": rawpy.DemosaicAlgorithm.VNG,
            "LINEAR": rawpy.DemosaicAlgorithm.LINEAR,
            "DCB": rawpy.DemosaicAlgorithm.DCB,
            "AHD": rawpy.DemosaicAlgorithm.AHD,
        }
        
        # Create minimal DNG in memory for rawpy
        dng_buffer = io.BytesIO()
        dngio.write_dng(
            raw_data=image_data,
            cfa_pattern=cfa_pattern,
            destination_file=dng_buffer,
            bits_per_pixel=16,
            camera_profile=None,  # Minimal DNG, no profile needed
            jxl_distance=None,  # No compression for rawpy compatibility
            jxl_effort=None,
        )
        dng_buffer.seek(0)
        
        raw = rawpy.RawPy()
        try:
            raw.open_buffer(dng_buffer)
            raw.unpack()
            
            rgb = raw.postprocess(
                demosaic_algorithm=algorithm_map[algorithm],
                output_color=rawpy.ColorSpace.raw,  # Camera RGB space
                gamma=(1, 1),  # Linear
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=[1, 1, 1, 1],  # Unity multipliers
                output_bps=16
            )
        finally:
            raw.close()
        # rawpy already outputs RGB (not BGR like OpenCV)
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Demosaic ({algorithm}): {elapsed_time*1000:.1f}ms for {image_data.shape[1]}x{image_data.shape[0]}")

    # Apply orientation if provided: ONLY accept EXIF codes (1,3,6,8).
    if isinstance(orientation, (int, np.integer)):
        exif_code = int(orientation)
        if exif_code == 6:       # 90° CW
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        elif exif_code == 3:     # 180°
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        elif exif_code == 8:     # 270° CW (90° CCW)
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif exif_code not in (1, 3, 6, 8):
            logger.warning(f"Unsupported EXIF orientation code: {exif_code}; no rotation applied")

    return rgb

def linear_raw_from_dng(
    dng_file: "DngFile",
    orientation: int | None = None,
    algorithm: str = "RCD",
    hot_candidates: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None
    ] = (None, None, None, None),
    hot_pixel_threshold: float = 2.5,
    hot_pixel_min_brightness: int = 32768
) -> np.ndarray:
    """Demosaic DNG file to RGB with optional hot pixel correction.
    
    Convenience wrapper that extracts CFA from DNG file and calls
    linear_raw_from_cfa.
    
    Args:
        dng_file: DngFile object
        orientation: Optional EXIF orientation code (1, 3, 6, or 8)
        algorithm: Demosaic algorithm - "RCD" (default), "VNG", "LINEAR",
            "DCB", "AHD", or "OPENCV_EA"
        hot_candidates: Optional tuple of (r, g1, g2, b) binary masks of
            hot pixel candidates (H/2, W/2)
        hot_pixel_threshold: Multiplier threshold for hot pixel detection
            (default: 2.5)
        hot_pixel_min_brightness: Minimum brightness to consider for hot
            pixels (default: 32768)
        
    Returns:
        RGB array (uint16)
        
    Raises:
        ValueError: If the DNG file format is invalid or missing required
            data.
    """
    from . import dngio
    
    # Extract CFA data and pattern from DNG
    cfa, cfa_pattern = dngio.cfa_from_dng(dng_file)
    
    # Call linear_raw_from_cfa with all parameters
    return linear_raw_from_cfa(
        cfa,
        cfa_pattern,
        orientation=orientation,
        algorithm=algorithm,
        hot_candidates=hot_candidates,
        hot_pixel_threshold=hot_pixel_threshold,
        hot_pixel_min_brightness=hot_pixel_min_brightness
    )

class ToneCurve:
    """
    Represents a tone curve with 5 control points.
    
    The curve uses 8-bit values (0-255) for compatibility with XMP tone curve format,
    but can be normalized to 0-1 range for use with Core Image filters.
    """
    
    def __init__(self, points_or_string=None):
        """
        Initialize ToneCurve.
        
        Args:
            points_or_string: Optional. Can be:
                - None: Initialize with default linear curve (0,0), (255,255)
                - str: Parse from string format "(0,0),(255,255)"
                - list: Use provided points directly
        """
        if points_or_string is None:
            # Default linear curve
            self.points = [
                (0, 0),
                (255, 255)  # Using 255 instead of 256 for proper 8-bit range
            ]
        elif isinstance(points_or_string, str):
            # Parse from string format
            self._parse_from_string(points_or_string)
        elif isinstance(points_or_string, list):
            # Use provided points directly
            self.points = points_or_string
        else:
            raise TypeError(f"ToneCurve constructor expects None, str, or list, got {type(points_or_string)}")
    
    def _parse_from_string(self, string_data: str):
        """Helper method to parse points from string format."""
        import re
        
        # Extract coordinate pairs using regex
        tuple_pattern = r'\((\d+),(\d+)\)'
        matches = re.findall(tuple_pattern, string_data)
        
        if not matches:
            raise ValueError(f"No valid coordinate pairs found in tone curve data: {string_data}")
        
        # Convert string coordinates to integer tuples
        self.points = [(int(x), int(y)) for x, y in matches]
    

    
    def to_normalized(self):
        """
        Convert 8-bit points to normalized 0-1 range for Core Image.
        
        Returns:
            List of (x, y) tuples with values normalized to 0-1 range.
        """
        return [(x / 255.0, y / 255.0) for x, y in self.points]
    
    @classmethod
    def from_scurve(cls, strength: float):
        """
        Create a new ToneCurve instance with S-curve adjustment applied.
        
        Args:
            strength: Float between 0.0 and 1.0, where 0.0 is no adjustment
                     and 1.0 is maximum S-curve effect.
        
        Returns:
            ToneCurve: New instance with S-curve applied.
        """
        curve = cls()
        
        # Clamp strength to valid range
        s = min(max(float(strength), 0.0), 1.0)
        
        # Constants matching the Core Image implementation
        SHADOW_PULL_FACTOR = 0.3
        HIGHLIGHT_PUSH_FACTOR = 0.015
        
        # Convert the Core Image normalized coordinates to 8-bit values
        # Core Image uses 0-1 range, we use 0-255 range
        curve.points = [
            (0, 0),  # Black point stays fixed
            (int(0.53 * 255), int((0.53 - s * SHADOW_PULL_FACTOR) * 255)),  # Shadow adjustment
            (int(0.73 * 255), int(0.73 * 255)),  # Midtone anchor point
            (int(0.90 * 255), int((0.90 + s * HIGHLIGHT_PUSH_FACTOR) * 255)),  # Highlight adjustment  
            (255, 255)  # White point stays fixed
        ]
        
        # Ensure output values stay within valid 8-bit range
        curve.points = [
            (x, min(max(y, 0), 255)) for x, y in curve.points
        ]
        
        return curve
    
    def __str__(self) -> str:
        """
        Convert ToneCurve to string format.
        
        Returns:
            String representation in format "(0,0),(56,30),(124,125),(188,212),(255,255)"
        """
        tuple_strings = [f"({x},{y})" for x, y in self.points]
        return ','.join(tuple_strings)
    
    def __repr__(self):
        return f"ToneCurve(points={self.points})"


# =============================================================================
# DNG SDK Port (Python + C++ Extension)
# =============================================================================
# Everything below is a port of the Adobe DNG SDK 1.7.1 color pipeline.
# C++ implementation in src/dng_color/dng_color_standalone.cpp
#
# Key SDK source files referenced:
#   - dng_color_spec.cpp: SetWhiteXY(), NeutralToXY(), FindXYZtoCamera()
#   - dng_render.cpp: dng_render_task::Start() for fCameraToRGB computation
#   - dng_color_spec.cpp: MapWhiteMatrix() for Bradford chromatic adaptation
#   - dng_reference.cpp: RefBaselineRGBTone() for hue-preserving tone mapping
#   - dng_ifd.cpp: Black/white level defaults and parsing
#   - dng_linearization_info.cpp: normalize_black_white implementation
# =============================================================================

def apply_acr3_tone_curve(image: np.ndarray) -> np.ndarray:
    """Apply the ACR3 default tone curve to a linear RGB image.
    
    This implements hue-preserving RGB tone mapping as per SDK's RefBaselineRGBTone.
    The curve is applied to max and min channels, and the middle channel is
    interpolated to preserve the original hue relationship.
    
    SDK ref: dng_reference.cpp lines 1868-1990
    """
    curve = _dng_color.get_acr3_curve(4096)
    return _dng_color.apply_rgb_tone(image.astype(np.float32), curve)


# Standard illuminant xy chromaticities (from DNG SDK)
D50_xy = (0.34567, 0.35850)  # PCS reference white
D55_xy = (0.33242, 0.34743)
D65_xy = (0.31271, 0.32902)  # sRGB reference white

# ProPhoto RGB matrices (from dng_color_space.cpp)
# ProPhoto uses D50 white point
PROPHOTO_RGB_TO_XYZ_D50 = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100]
], dtype=np.float64)

XYZ_D50_TO_PROPHOTO_RGB = np.array([
    [ 1.3459433, -0.2556075, -0.0511118],
    [-0.5445989,  1.5081673,  0.0205351],
    [ 0.0000000,  0.0000000,  1.2118128]
], dtype=np.float64)

# sRGB matrices (D65 white point)
XYZ_D65_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float64)

SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)


def _xy_to_XYZ(x: float, y: float) -> np.ndarray:
    """Convert xy chromaticity to XYZ with Y=1.
    
    Port of dng_color_spec.cpp: XYtoXYZ()
    """
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _XYZ_to_xy(XYZ: np.ndarray) -> tuple:
    """Convert XYZ to xy chromaticity.
    
    Port of dng_color_spec.cpp: XYZtoXY()
    """
    s = XYZ[0] + XYZ[1] + XYZ[2]
    if s <= 0:
        return D50_xy  # Fallback
    return (XYZ[0] / s, XYZ[1] / s)


def _neutral_to_xy(neutral: np.ndarray, color_matrix: np.ndarray, 
                   max_passes: int = 30) -> tuple:
    """Convert camera neutral to white point xy chromaticity.
    
    Port of dng_color_spec.cpp: NeutralToXY() (lines 659-706)
    Iteratively finds the white point that produces the given neutral.
    
    Args:
        neutral: Camera neutral RGB values (AsShotNeutral)
        color_matrix: XYZ to Camera matrix (ColorMatrix)
        max_passes: Maximum iterations (default 30 per SDK)
        
    Returns:
        (x, y) chromaticity of white point
    """
    last = D50_xy
    
    for pass_num in range(max_passes):
        # Invert color matrix to get camera->XYZ
        camera_to_xyz = np.linalg.inv(color_matrix)
        
        # Convert neutral through inverted matrix to get XYZ
        xyz = camera_to_xyz @ neutral
        
        # Convert XYZ to xy
        next_xy = _XYZ_to_xy(xyz)
        
        # Check convergence
        if abs(next_xy[0] - last[0]) + abs(next_xy[1] - last[1]) < 0.0000001:
            return next_xy
        
        # If we reach the limit, average last two estimates
        if pass_num == max_passes - 1:
            next_xy = ((last[0] + next_xy[0]) * 0.5,
                       (last[1] + next_xy[1]) * 0.5)
        
        last = next_xy
    
    return last


def _compute_camera_to_pcs(color_matrix: np.ndarray, white_xy: tuple) -> np.ndarray:
    """Compute Camera RGB to PCS (D50 XYZ) matrix.
    
    Port of dng_color_spec.cpp: SetWhiteXY() (lines 570-609)
    
    The SDK computes:
        fPCStoCamera = colorMatrix * MapWhiteMatrix(D50, white_xy)
        scale = MaxEntry(fPCStoCamera * PCStoXYZ())
        fPCStoCamera = (1/scale) * fPCStoCamera  # CRITICAL SCALING
        fCameraToPCS = Invert(fPCStoCamera)
    
    Args:
        color_matrix: XYZ to Camera matrix (ColorMatrix from DNG)
        white_xy: White point xy chromaticity
        
    Returns:
        3x3 Camera RGB to PCS (D50 XYZ) matrix
    """
    # Bradford adaptation from PCS (D50) to the white point
    # Port of dng_color_spec.cpp: MapWhiteMatrix() (lines 22-57)
    adaptation = _dng_color.bradford_adapt(D50_xy[0], D50_xy[1], white_xy[0], white_xy[1])
    
    # PCS to Camera = ColorMatrix * adaptation
    pcs_to_camera = color_matrix @ adaptation
    
    # SDK ref: dng_color_spec.cpp lines 570-582
    # Scale matrix so PCS white can just be reached when first camera channel saturates
    pcs_white_xyz = _xy_to_XYZ(D50_xy[0], D50_xy[1])  # PCStoXYZ()
    scale = np.max(pcs_to_camera @ pcs_white_xyz)     # MaxEntry(fPCStoCamera * PCStoXYZ())
    if scale != 0:
        pcs_to_camera = pcs_to_camera / scale         # (1.0 / scale) * fPCStoCamera
    
    # Camera to PCS = inverse
    camera_to_pcs = np.linalg.inv(pcs_to_camera)
    
    return camera_to_pcs


def _compute_camera_white(color_matrix: np.ndarray, white_xy: tuple) -> np.ndarray:
    """Compute camera white (what the white point looks like in camera space).
    
    Port of dng_color_spec.cpp: SetWhiteXY() (lines 546-568)
    
    fCameraWhite = colorMatrix * XYtoXYZ(white)
    Then normalized so max entry = 1.0
    
    Args:
        color_matrix: XYZ to Camera matrix
        white_xy: White point xy chromaticity
        
    Returns:
        Normalized camera white RGB values
    """
    white_XYZ = _xy_to_XYZ(white_xy[0], white_xy[1])
    camera_white = color_matrix @ white_XYZ
    
    # Normalize so max = 1.0
    max_val = np.max(camera_white)
    if max_val > 0:
        camera_white = camera_white / max_val
    
    # Clamp to valid range
    camera_white = np.clip(camera_white, 0.001, 1.0)
    
    return camera_white


def process_raw(
    dng_input,
    output_dtype: type = np.uint16,
    algorithm: str = "RCD",
    strict: bool = True,
) -> "np.ndarray | None":
    """Process a DNG file to RGB using the DNG SDK color pipeline.
    
    Exact port of Adobe DNG SDK 1.7.1 dng_render_task::ProcessArea()
    SDK source: dng_render.cpp lines 1780-2070
    
    Pipeline (matching SDK exactly):
        1. DoBaselineABCtoRGB - camera RGB to ProPhoto with camera_white clipping
           SDK ref: dng_reference.cpp lines 1389-1441
        2. DoBaseline1DFunction (ExposureRamp) - exposure/shadows adjustment
           SDK ref: dng_render.cpp lines 1907-1928
        3. DoBaselineRGBTone - ACR3 tone curve (ALWAYS applied)
           SDK ref: dng_render.cpp lines 1949-1970
        4. DoBaselineRGBtoRGB - ProPhoto to final space (sRGB)
           SDK ref: dng_render.cpp lines 2040-2048
        5. DoBaseline1DTable (EncodeGamma) - sRGB gamma encoding
           SDK ref: dng_render.cpp lines 2050-2068
    
    Args:
        dng_input: Path to DNG file or file-like object
        output_dtype: Output dtype (np.uint8, np.uint16, np.float16, np.float32)
        algorithm: Demosaic algorithm for CFA data
        strict: If True, raise UnsupportedDNGTagError if the DNG contains tags
                that this function cannot handle. If False, process anyway.
    
    Returns:
        RGB image array or None on failure
        
    Raises:
        UnsupportedDNGTagError: If strict=True and unsupported DNG tags are found
    """
    from . import dngio
    import time
    
    timings = {}
    
    try:
        t0 = time.perf_counter()
        dng = dngio.DngFile(dng_input)
        timings['dng_open'] = time.perf_counter() - t0
        
        raw_pages = dng.get_raw_pages_info()
        if not raw_pages:
            logger.error("No raw pages found in DNG file")
            return None
        
        page_id, shape, tags = raw_pages[0]
        
        # Validate that we support all rendering-related tags in this DNG
        unsupported = validate_dng_tags(tags, strict=strict)
        if unsupported and not strict:
            logger.warning(f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}")
        
        photometric = tags.get("PhotometricInterpretation")
        
        # =================================================================
        # Black/White level handling per SDK dng_ifd.cpp
        # =================================================================
        # SDK ref: dng_ifd.cpp:242-252 (constructor)
        #   for (j = 0; j < kMaxBlackPattern; j++)
        #       for (k = 0; k < kMaxBlackPattern; k++)
        #           for (n = 0; n < kMaxColorPlanes; n++)
        #               fBlackLevel [j] [k] [n] = 0.0;
        #   for (j = 0; j < kMaxColorPlanes; j++)
        #       fWhiteLevel [j] = -1.0;  // sentinel
        #
        # SDK ref: dng_ifd.cpp:3247-3259 (PostParse)
        #   uint32 defaultWhite = (fSampleFormat[0] == sfFloatingPoint) ?
        #                         1 : (uint32)((((uint64)1) << fBitsPerSample[0]) - 1);
        #   for (j = 0; j < kMaxColorPlanes; j++)
        #       if (fWhiteLevel[j] < 0.0)
        #           fWhiteLevel[j] = (real64) defaultWhite;
        # =================================================================
        
        samples_per_pixel = tags.get("SamplesPerPixel", 1)
        black_repeat_dim = tags.get("BlackLevelRepeatDim", (1, 1))
        black_repeat_rows = int(black_repeat_dim[0]) if hasattr(black_repeat_dim, '__len__') else 1
        black_repeat_cols = int(black_repeat_dim[1]) if hasattr(black_repeat_dim, '__len__') and len(black_repeat_dim) > 1 else 1
        expected_black_size = black_repeat_rows * black_repeat_cols * samples_per_pixel
        
        # BlackLevel: SDK initializes to 0.0 for all [row][col][plane]
        black_level_raw = tags.get("BlackLevel")
        if black_level_raw is None:
            # No tag: all zeros (SDK constructor behavior)
            black_level = np.zeros(expected_black_size, dtype=np.float32)
        else:
            black_level = np.atleast_1d(black_level_raw).astype(np.float32).ravel()
            if len(black_level) != expected_black_size:
                # SDK would fail (CheckTagCount), use zeros as fallback
                black_level = np.zeros(expected_black_size, dtype=np.float32)
        
        # WhiteLevel: SDK initializes to -1.0 (sentinel), then PostParse sets default
        # Default: (1 << BitsPerSample) - 1, or 1 for floating point
        bits_per_sample = tags.get("BitsPerSample", 16)
        if isinstance(bits_per_sample, (list, tuple)):
            bits_per_sample = int(bits_per_sample[0])
        else:
            bits_per_sample = int(bits_per_sample)
        default_white = float((1 << bits_per_sample) - 1)
        
        white_level_raw = tags.get("WhiteLevel")
        if white_level_raw is None:
            # No tag: use default for all planes (SDK PostParse behavior)
            white_level = np.full(samples_per_pixel, default_white, dtype=np.float32)
        else:
            white_level = np.atleast_1d(white_level_raw).astype(np.float32).ravel()
            # SDK sets default for any plane where value < 0
            white_level = np.where(white_level < 0, default_white, white_level)
            if len(white_level) < samples_per_pixel:
                # Extend with default if not enough values
                white_level = np.concatenate([
                    white_level,
                    np.full(samples_per_pixel - len(white_level), default_white, dtype=np.float32)
                ])
        
        black_delta_h = tags.get("BlackLevelDeltaH")
        black_delta_v = tags.get("BlackLevelDeltaV")
        
        # Convert delta arrays if present
        if black_delta_h is not None:
            black_delta_h = np.atleast_1d(black_delta_h).astype(np.float32)
        if black_delta_v is not None:
            black_delta_v = np.atleast_1d(black_delta_v).astype(np.float32)
        
        orientation = tags.get("Orientation", 1)
        
        # =====================================================================
        # Pre-processing: Demosaic / extract linear camera RGB
        # SDK applies crop BEFORE rotation, so we demosaic without rotation,
        # apply crop in sensor coords, then rotate.
        # =====================================================================
        t0 = time.perf_counter()
        
        # Get crop parameters (in original sensor coordinates)
        crop_origin = tags.get("DefaultCropOrigin")
        crop_size = tags.get("DefaultCropSize")
        
        if photometric == "LINEAR_RAW":
            rgb_data = dng.get_raw_linear_by_id(page_id)
            if rgb_data is None:
                logger.error("Failed to extract LINEAR_RAW data from DNG")
                return None
            
            # Normalize using C++ implementation per DNG spec Chapter 5
            rgb_camera = _dng_color.normalize_raw(
                data=rgb_data.astype(np.float32),
                black_level=black_level,
                black_repeat_rows=black_repeat_rows,
                black_repeat_cols=black_repeat_cols,
                samples_per_pixel=samples_per_pixel,
                white_level=white_level,
                black_delta_h=black_delta_h,
                black_delta_v=black_delta_v,
            )
        else:
            cfa_result = dng.get_raw_cfa_by_id(page_id)
            if cfa_result is None:
                logger.error("Failed to extract CFA data from DNG")
                return None
            
            cfa_data, cfa_pattern, cfa_pattern_codes = cfa_result
            if cfa_pattern is None:
                cfa_pattern = "RGGB"
            if cfa_pattern_codes is None:
                cfa_pattern_codes = (0, 1, 1, 2)  # Default RGGB
            
            # Normalize CFA data using C++ implementation per DNG spec Chapter 5
            # SDK demosaics on float32 throughout
            cfa_normalized = _dng_color.normalize_raw(
                data=cfa_data.astype(np.float32),
                black_level=black_level,
                black_repeat_rows=black_repeat_rows,
                black_repeat_cols=black_repeat_cols,
                samples_per_pixel=1,  # CFA is always 1 sample per pixel
                white_level=white_level,
                black_delta_h=black_delta_h,
                black_delta_v=black_delta_v,
            )
            
            # Demosaic without rotation - rotation applied after crop below
            if algorithm == "DNGSDK_BILINEAR":
                # DNG SDK bilinear demosaic - uses raw CFAPattern codes directly
                # SDK ref: dng_mosaic_info::fCFAPattern[row][col]
                cfa_pattern_array = np.array(cfa_pattern_codes, dtype=np.int32)
                rgb_camera = _dng_color.bilinear_demosaic(cfa_normalized, cfa_pattern_array)
            else:
                # Other algorithms require uint16 input
                cfa_uint16 = (cfa_normalized * 65535).astype(np.uint16)
                rgb_linear = linear_raw_from_cfa(
                    cfa_uint16, cfa_pattern,
                    orientation=None,
                    algorithm=algorithm
                )
                rgb_camera = rgb_linear.astype(np.float32) / 65535.0
        
        # Apply DefaultCrop BEFORE rotation
        # SDK ref: dng_negative.cpp:2535-2575 DefaultCropArea() uses H/V directly
        # SDK ref: dng_negative.cpp:3321-3327 crop values read without transformation
        if crop_origin is not None and crop_size is not None:
            crop_x = int(crop_origin[0])
            crop_y = int(crop_origin[1])
            crop_w = int(crop_size[0])
            crop_h = int(crop_size[1])
            rgb_camera = rgb_camera[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Apply orientation rotation
        if orientation == 6:
            rgb_camera = cv2.rotate(rgb_camera, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 3:
            rgb_camera = cv2.rotate(rgb_camera, cv2.ROTATE_180)
        elif orientation == 8:
            rgb_camera = cv2.rotate(rgb_camera, cv2.ROTATE_90_COUNTERCLOCKWISE)
        timings['demosaic'] = time.perf_counter() - t0
        
        # =====================================================================
        # Setup: Compute matrices (port of dng_render_task::Start)
        # SDK ref: dng_render.cpp lines 869-1070
        # =====================================================================
        
        # Get ColorMatrix1 (XYZ to Camera, 3x3)
        color_matrix1 = tags.get("ColorMatrix1")
        if color_matrix1 is None:
            logger.warning("No ColorMatrix1 found, using identity")
            color_matrix1 = np.eye(3, dtype=np.float64)
        else:
            color_matrix1 = np.asarray(color_matrix1, dtype=np.float64)
        
        # Get ColorMatrix2 for dual-illuminant interpolation
        color_matrix2 = tags.get("ColorMatrix2")
        if color_matrix2 is not None:
            color_matrix2 = np.asarray(color_matrix2, dtype=np.float64)
        
        # Get ForwardMatrix1/2 (camera to PCS, 3x3)
        # SDK ref: dng_color_spec.cpp lines 126-128, 586-596
        forward_matrix1 = tags.get("ForwardMatrix1")
        forward_matrix2 = tags.get("ForwardMatrix2")
        
        # Get CameraCalibration1/2 matrices (3x3, default to identity)
        # SDK ref: dng_color_spec.cpp lines 134-166
        camera_calib1 = tags.get("CameraCalibration1")
        camera_calib2 = tags.get("CameraCalibration2")
        if camera_calib1 is None:
            camera_calib1 = np.eye(3, dtype=np.float64)
        if camera_calib2 is None:
            camera_calib2 = np.eye(3, dtype=np.float64)
        
        # Get calibration illuminant temperatures
        illum1 = tags.get("CalibrationIlluminant1")
        illum2 = tags.get("CalibrationIlluminant2")
        temp1 = illuminant_to_temperature(illum1) if illum1 is not None else None
        temp2 = illuminant_to_temperature(illum2) if illum2 is not None else None
        
        # Get AsShotNeutral -> convert to white point XY
        # SDK ref: dng_render.cpp lines 889-908
        as_shot = tags.get("AsShotNeutral")
        as_shot_xy = tags.get("AsShotWhiteXY")
        camera_neutral = None
        white_xy_override = None
        
        if as_shot is not None and hasattr(as_shot, '__len__') and len(as_shot) >= 3:
            camera_neutral = np.array(as_shot[:3], dtype=np.float64)
        elif as_shot_xy is not None and len(as_shot_xy) >= 2:
            white_xy_override = (float(as_shot_xy[0]), float(as_shot_xy[1]))
        else:
            white_xy_override = D55_xy  # SDK default
        
        # Get AnalogBalance
        ab_diag = np.eye(3, dtype=np.float64)
        analog_balance = tags.get("AnalogBalance")
        if analog_balance is not None:
            analog_balance = np.asarray(analog_balance, dtype=np.float64)
            if analog_balance.size >= 3:
                ab_diag = np.diag(analog_balance[:3])
        
        # Apply AnalogBalance and CameraCalibration to ColorMatrix BEFORE interpolation
        # SDK ref: dng_color_spec.cpp lines 179, 215
        color_matrix1 = ab_diag @ camera_calib1 @ color_matrix1
        if color_matrix2 is not None:
            color_matrix2 = ab_diag @ camera_calib2 @ color_matrix2

        # Determine interpolation weight and interpolate matrices
        # SDK ref: dng_color_spec.cpp FindXYZtoCamera_SingleOrDual() lines 301-460
        if color_matrix2 is not None and temp1 is not None and temp2 is not None and temp1 != temp2:
            # Dual-illuminant: interpolate based on scene temperature
            white_xy_est = white_xy_override if white_xy_override else _neutral_to_xy(camera_neutral, color_matrix1)
            scene_temp = _dng_color.xy_to_temp(white_xy_est[0], white_xy_est[1])[0]
            
            # Calculate interpolation weight g
            t1, t2 = (temp1, temp2) if temp1 < temp2 else (temp2, temp1)
            if scene_temp <= t1:
                g = 1.0
            elif scene_temp >= t2:
                g = 0.0
            else:
                inv_t = 1.0 / scene_temp
                g = (inv_t - 1.0/t2) / (1.0/t1 - 1.0/t2)
            
            # Interpolate matrices (g1 weights matrix1, g2 weights matrix2)
            g1, g2 = (1.0 - g, g) if temp1 > temp2 else (g, 1.0 - g)
            color_matrix = g1 * color_matrix1 + g2 * color_matrix2
            camera_calib = g1 * camera_calib1 + g2 * camera_calib2
            if forward_matrix1 is not None and forward_matrix2 is not None:
                forward_matrix = g1 * forward_matrix1 + g2 * forward_matrix2
            else:
                forward_matrix = forward_matrix1 if forward_matrix1 is not None else forward_matrix2
        else:
            # Single illuminant
            color_matrix = color_matrix1
            camera_calib = camera_calib1
            forward_matrix = forward_matrix1 if forward_matrix1 is not None else forward_matrix2
        
        # SDK ref: dng_color_spec.cpp NeutralToXY() lines 659-706
        white_xy = white_xy_override if white_xy_override else _neutral_to_xy(camera_neutral, color_matrix)
        
        # SDK ref: dng_color_spec.cpp SetWhiteXY() lines 546-568
        camera_white = _compute_camera_white(color_matrix, white_xy)
        
        # SDK ref: dng_color_spec.cpp SetWhiteXY() lines 570-609
        # ForwardMatrix takes precedence if present
        if forward_matrix is not None:
            # fCameraToPCS = forwardMatrix * Invert(refCameraWhite.AsDiagonal()) * individualToReference
            individual_to_ref = np.linalg.inv(ab_diag @ camera_calib)
            ref_camera_white = individual_to_ref @ camera_white
            ref_camera_white = np.clip(ref_camera_white, 0.001, None)
            camera_to_pcs = forward_matrix @ np.linalg.inv(np.diag(ref_camera_white)) @ individual_to_ref
        else:
            camera_to_pcs = _compute_camera_to_pcs(color_matrix, white_xy)
        
        # SDK ref: dng_render.cpp lines 912-913
        # fCameraToRGB = ProPhoto.MatrixFromPCS() * CameraToPCS()
        camera_to_prophoto = XYZ_D50_TO_PROPHOTO_RGB @ camera_to_pcs
        
        # SDK ref: dng_render.cpp lines 1042-1043
        # fRGBtoFinal = sRGB.MatrixFromPCS() * ProPhoto.MatrixToPCS()
        prophoto_to_srgb = XYZ_D65_TO_SRGB @ _dng_color.bradford_adapt(D50_xy[0], D50_xy[1], D65_xy[0], D65_xy[1]) @ PROPHOTO_RGB_TO_XYZ_D50
        
        # =====================================================================
        # Step 1: DoBaselineABCtoRGB
        # SDK ref: dng_reference.cpp lines 1389-1441
        # Clip to camera_white, apply camera_to_prophoto matrix, pin to [0,1]
        # =====================================================================
        t0 = time.perf_counter()
        rgb_clipped = np.minimum(rgb_camera, camera_white.astype(np.float32))
        rgb_prophoto = _dng_color.matrix_transform(rgb_clipped, camera_to_prophoto.astype(np.float32))
        timings['matrix_camera_to_prophoto'] = time.perf_counter() - t0
        
        # =====================================================================
        # Step 2: DoBaseline1DFunction (ExposureRamp)
        # SDK ref: dng_render.cpp lines 975-999, 1907-1928
        # Default: exposure=0, shadows=5 -> white=1.0, black=0.005
        # =====================================================================
        t0 = time.perf_counter()
        exposure_white = 1.0
        exposure_black = 0.005  # shadows=5 * 0.001
        exposure_slope = 1.0 / (exposure_white - exposure_black)
        rgb_exposed = np.clip((rgb_prophoto - exposure_black) * exposure_slope, 0.0, 1.0)
        timings['exposure_ramp'] = time.perf_counter() - t0
        
        # =====================================================================
        # Step 3: DoBaselineRGBTone (ALWAYS applied)
        # SDK ref: dng_render.cpp lines 1949-1970
        # Uses ACR3 default tone curve
        # =====================================================================
        t0 = time.perf_counter()
        rgb_toned = apply_acr3_tone_curve(rgb_exposed)
        timings['tone_curve'] = time.perf_counter() - t0
        
        # =====================================================================
        # Step 4: DoBaselineRGBtoRGB
        # SDK ref: dng_render.cpp lines 2040-2048
        # Convert ProPhoto (D50) to sRGB (D65)
        # =====================================================================
        t0 = time.perf_counter()
        rgb_srgb = _dng_color.matrix_transform(rgb_toned.astype(np.float32), prophoto_to_srgb.astype(np.float32))
        timings['matrix_prophoto_to_srgb'] = time.perf_counter() - t0
        
        # =====================================================================
        # Step 5: DoBaseline1DTable (EncodeGamma)
        # SDK ref: dng_render.cpp lines 2050-2068
        # Apply sRGB gamma encoding
        # =====================================================================
        t0 = time.perf_counter()
        rgb_final = _dng_color.srgb_gamma(rgb_srgb.astype(np.float32))  # includes clipping
        timings['srgb_gamma'] = time.perf_counter() - t0
        
        # Convert to output dtype
        t0 = time.perf_counter()
        if output_dtype == np.uint8:
            result = (rgb_final * 255).astype(np.uint8)
        elif output_dtype == np.uint16:
            result = (rgb_final * 65535).astype(np.uint16)
        elif output_dtype == np.float16:
            result = rgb_final.astype(np.float16)
        elif output_dtype == np.float32:
            result = rgb_final.astype(np.float32)
        else:
            logger.warning(f"Unsupported output_dtype {output_dtype}, using float32")
            result = rgb_final.astype(np.float32)
        timings['dtype_convert'] = time.perf_counter() - t0
        
        # Print timing breakdown
        total = sum(timings.values())
        logger.info(f"process_raw timing breakdown (total: {total*1000:.1f}ms):")
        for name, t in timings.items():
            logger.info(f"  {name}: {t*1000:.1f}ms ({t/total*100:.1f}%)")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing DNG: {e}")
        import traceback
        traceback.print_exc()
        return None
