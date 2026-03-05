"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import io
import logging
import imagecodecs
import numpy as np

from pathlib import Path
from tifffile import COMPRESSION, PHOTOMETRIC, TiffFile, TiffPage, TiffWriter, TIFF
from typing import Optional, Union, List, Dict, Tuple, Any, Type, IO

logger = logging.getLogger(__name__)

# Import metadata classes from tiff_metadata module
from .tiff_metadata import (
    MetadataTags,
    XmpMetadata,
    BAYER_PATTERN_MAP,
    INVERSE_BAYER_PATTERN_MAP,
    translate_dng_tag,
)


# Default values for tags
ORIENTATION_HORIZONTAL = 1

# Complete illuminant mapping (from EXIF specification)
ILLUMINANTS = {
    0: 'Unknown',
    1: 'Daylight',
    2: 'Fluorescent', 
    3: 'Tungsten (incandescent light)',
    4: 'Flash',
    9: 'Fine weather',
    10: 'Cloudy weather',
    11: 'Shade',
    12: 'Daylight fluorescent (D 5700 – 7100K)',
    13: 'Day white fluorescent (N 4600 – 5400K)',
    14: 'Cool white fluorescent (W 3900 – 4500K)',
    15: 'White fluorescent (WW 3200 – 3700K)',
    17: 'Standard light A',
    18: 'Standard light B',
    19: 'Standard light C',
    20: 'D55',
    21: 'D65',
    22: 'D75',
    23: 'D50',
    24: 'ISO studio tungsten',
    255: 'Other light source',
}

PREVIEWCOLORSPACE_SRGB = 1

def swizzle_cfa_data(raw_data: np.ndarray) -> np.ndarray:
    """Swizzle RGGB CFA data into a 2x2 grid of R, G1, G2, B sub-images."""

    # Pre-allocate the swizzled array with same dtype as input
    h, w = raw_data.shape
    swizzled_data = np.empty((h, w), dtype=raw_data.dtype)
    
    # Calculate half dimensions for direct assignment
    h_half, w_half = h // 2, w // 2
    
    # Extract channels and write directly into pre-allocated array
    # R pixels: top-left quadrant
    swizzled_data[0:h_half, 0:w_half] = raw_data[0::2, 0::2]
    # G1 pixels: top-right quadrant  
    swizzled_data[0:h_half, w_half:w] = raw_data[0::2, 1::2]
    # G2 pixels: bottom-left quadrant
    swizzled_data[h_half:h, 0:w_half] = raw_data[1::2, 0::2]
    # B pixels: bottom-right quadrant
    swizzled_data[h_half:h, w_half:w] = raw_data[1::2, 1::2]

    return swizzled_data

def deswizzle_cfa_data(swizzled_data: np.ndarray) -> np.ndarray:
    """Deswizzle CFA data from a 2x2 grid of R, G1, G2, B sub-images back to RGGB."""
    h_swizzled, w_swizzled = swizzled_data.shape
    if h_swizzled % 2 != 0 or w_swizzled % 2 != 0:
        raise ValueError("Swizzled data dimensions must be even.")

    # Calculate half dimensions for quadrant extraction
    h_half, w_half = h_swizzled // 2, w_swizzled // 2

    # Extract the four channels from the swizzled data
    # R is top-left quadrant
    r_channel = swizzled_data[0:h_half, 0:w_half]
    # G1 (first green) is top-right quadrant
    g1_channel = swizzled_data[0:h_half, w_half:w_swizzled]
    # G2 (second green) is bottom-left quadrant
    g2_channel = swizzled_data[h_half:h_swizzled, 0:w_half]
    # B is bottom-right quadrant
    b_channel = swizzled_data[h_half:h_swizzled, w_half:w_swizzled]

    # Create an empty array for the original interleaved data
    # Its dimensions will be the same as the swizzled_data because each sub-image
    # was H/2 x W/2, and they are re-interleaved into a H x W image.
    original_data = np.empty_like(swizzled_data)

    # Place the channels back into the original RGGB pattern
    original_data[0::2, 0::2] = r_channel  # R pixels
    original_data[0::2, 1::2] = g1_channel  # G1 pixels (top-right G)
    original_data[1::2, 0::2] = g2_channel  # G2 pixels (bottom-left G)
    original_data[1::2, 1::2] = b_channel  # B pixels

    return original_data


def cfa_from_dng(
    dng_file: "DngFile",
) -> tuple[np.ndarray, str]:
    """
    Loads a DNG file and extracts the raw CFA data and pattern.

    Returns:
        Tuple of (raw_cfa_array, cfa_pattern_string)
        
    Raises:
        ValueError: If the DNG file format is invalid or missing required data.
        RuntimeError: If DNG processing fails.
    """
    try:
        # 1. Get info about raw pages to find the CFA data
        raw_pages_info = dng_file.get_raw_pages_info()
        if not raw_pages_info:
            raise ValueError("No raw pages found in DNG")

        # 2. Find the first page with CFA photometric interpretation
        cfa_page_id = None
        for page_id_loop, shape, tags_loop in raw_pages_info:
            if tags_loop.get("PhotometricInterpretation") == "CFA":
                cfa_page_id = page_id_loop
                break

        if cfa_page_id is None:
            raise ValueError("No page with CFA interpretation found in DNG")

        # 3. Get the CFA data array and pattern via API
        result = dng_file.get_raw_cfa_by_id(cfa_page_id)
        if result is None:
            raise RuntimeError(
                f"Failed to retrieve raw CFA data for page {cfa_page_id}"
            )
        raw_cfa, cfa_pattern_value = result

        if cfa_pattern_value is None:
            raise ValueError(f"Missing CFAPattern tag for page {cfa_page_id}")

    except (ValueError, RuntimeError):
        # Re-raise our specific errors as-is
        raise
    except Exception as e:
        raise RuntimeError(f"Error processing DNG file: {e}") from e

    return raw_cfa, cfa_pattern_value


def rgb_planes_from_cfa(
    raw_cfa: np.ndarray, 
    cfa_pattern_str: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts R, G1, G2, and B planes from CFA data using the given pattern.
    
    Args:
        raw_cfa: Raw CFA data array
        cfa_pattern_str: CFA pattern string (e.g., "RGGB", "BGGR")
        
    Returns:
        Tuple of (r_plane, g1_plane, g2_plane, b_plane)
        
    Raises:
        ValueError: If the CFA pattern is invalid.
    """
    # Parse the CFA pattern string
    cfa_pattern_flat = BAYER_PATTERN_MAP.get(cfa_pattern_str)
    if cfa_pattern_flat is None:
        raise ValueError(f"Unknown CFAPattern string '{cfa_pattern_str}'")

    cfa_pattern = np.array(cfa_pattern_flat).reshape(2, 2)
    
    # Extract R, G, B planes based on their positions in the CFA pattern
    # The BAYER_PATTERN_MAP uses integer values (0=R, 1=G, 2=B)
    r_pos_list = np.argwhere(cfa_pattern == 0)
    g_pos_list = np.argwhere(cfa_pattern == 1)
    b_pos_list = np.argwhere(cfa_pattern == 2)

    # There should be exactly one R, one B, and two G channels
    if len(r_pos_list) != 1 or len(b_pos_list) != 1 or len(g_pos_list) != 2:
        raise ValueError(f"Unexpected CFA pattern layout: {cfa_pattern}")

    # Extract planes
    r_plane = raw_cfa[r_pos_list[0][0]::2, r_pos_list[0][1]::2]
    b_plane = raw_cfa[b_pos_list[0][0]::2, b_pos_list[0][1]::2]
    
    g1_plane = raw_cfa[g_pos_list[0][0]::2, g_pos_list[0][1]::2]
    g2_plane = raw_cfa[g_pos_list[1][0]::2, g_pos_list[1][1]::2]

    return r_plane, g1_plane, g2_plane, b_plane


def cfa_from_rgb_planes(
    channels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfa_pattern_str: str,
    cfa_shape: tuple[int, int]
) -> np.ndarray:
    """Recompose CFA data from individual R, G1, G2, B planes.
    
    Args:
        channels: Tuple of (r, g1, g2, b) channel arrays (H/2, W/2) uint16
        cfa_pattern_str: CFA pattern string (e.g., "RGGB", "BGGR")
        cfa_shape: Shape of output CFA (H, W)
        
    Returns:
        Reconstructed CFA data (H, W) uint16
        
    Raises:
        ValueError: If the CFA pattern is invalid.
    """
    r, g1, g2, b = channels
    
    # Parse the CFA pattern string
    cfa_pattern_flat = BAYER_PATTERN_MAP.get(cfa_pattern_str)
    if cfa_pattern_flat is None:
        raise ValueError(f"Unknown CFAPattern string '{cfa_pattern_str}'")
    
    cfa_pattern = np.array(cfa_pattern_flat).reshape(2, 2)
    
    # Find positions of each channel in the 2x2 pattern
    r_pos = np.argwhere(cfa_pattern == 0)[0]  # 0 = R
    g_pos = np.argwhere(cfa_pattern == 1)     # 1 = G (two positions)
    b_pos = np.argwhere(cfa_pattern == 2)[0]  # 2 = B
    
    # Write channels to CFA
    cfa = np.zeros(cfa_shape, dtype=np.uint16)
    cfa[r_pos[0]::2, r_pos[1]::2] = r
    cfa[g_pos[0][0]::2, g_pos[0][1]::2] = g1
    cfa[g_pos[1][0]::2, g_pos[1][1]::2] = g2
    cfa[b_pos[0]::2, b_pos[1]::2] = b
    
    return cfa


def rgb_planes_from_dng(
    dng_file: "DngFile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a DNG file, finds the primary CFA raw image, and extracts
    R, G1, G2, and B planes.

    Raises:
        ValueError: If the DNG file format is invalid or missing required data.
        RuntimeError: If DNG processing fails.
    """
    # Extract CFA data and pattern, then process to RGB planes
    raw_cfa, cfa_pattern_value = cfa_from_dng(dng_file)
    return rgb_planes_from_cfa(raw_cfa, cfa_pattern_value)


def _write_thumbnail_ifd(writer: TiffWriter, thumbnail_image: np.ndarray, dng_tags: MetadataTags) -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags
    dng_tags.add_tag(("PreviewColorSpace", "I", 1, PREVIEWCOLORSPACE_SRGB))

    # Write Thumbnail to SubIFD 0
    thumb_ifd_args = {
        "photometric": "rgb",  # Interprets data as RGB
        "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
        "compression": "jpeg",  # JPEG compression for thumbnail
        "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
        "extratags": dng_tags,
        "subfiletype": 1,  # Reduced resolution image (standard for DNG previews)
        "subifds": 1,  # Has main image as subifd
    }
    # set datasize to max uncompressed size to avoid writing strips
    datasize = thumbnail_image.shape[0] * thumbnail_image.shape[1] * 3
    writer.write(
        thumbnail_image,  # Use the thumbnail_image directly
        **thumb_ifd_args,
        rowsperstrip=datasize,
    )


def _create_dng_tags(camera_profile: Optional[MetadataTags], has_jxl: bool) -> MetadataTags:
    """Create DNG metadata tags with defaults and version info.
    
    Tag names are from tifffile.py TiffTagRegistry.
    Tag types are from tifffile.py DATA_DTYPES.
    
    Args:
        camera_profile: Optional camera profile metadata to include
        has_jxl: Whether JXL compression is being used (affects backward version)
        
    Returns:
        MetadataTags object with complete DNG metadata
    """
    dng_tags = MetadataTags()

    # Use camera_profile if provided, otherwise create empty metadata
    if camera_profile is not None:
        dng_tags.extend(camera_profile)
    
    # Add required tags if not already set
    if "Orientation" not in dng_tags:
        dng_tags.add_tag(("Orientation", "H", 1, ORIENTATION_HORIZONTAL))
    
    if "ColorMatrix1" not in dng_tags:
        identity_matrix = np.identity(3, dtype=np.float64)
        dng_tags.add_matrix_as_rational_tag("ColorMatrix1", identity_matrix)
    
    if "CalibrationIlluminant1" not in dng_tags:
        dng_tags.add_tag(("CalibrationIlluminant1", "H", 1, 0))  # 0 = Unknown

    dng_tags.add_tag(("DNGVersion", "B", 4, (1, 7, 1, 0)))
    if not has_jxl:
        # need latest version for CFA compression but lots of old software can't handle it
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 4, 0, 0)))
    else:
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 7, 1, 0)))
        
    return dng_tags


def _write_thumbnail_ifd(writer: TiffWriter, thumbnail_image: np.ndarray, dng_tags: MetadataTags) -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags
    dng_tags.add_tag(("PreviewColorSpace", "I", 1, PREVIEWCOLORSPACE_SRGB))

    # Write Thumbnail to SubIFD 0
    thumb_ifd_args = {
        "photometric": "rgb",  # Interprets data as RGB
        "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
        "compression": "jpeg",  # JPEG compression for thumbnail
        "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
        "extratags": dng_tags,
        "subfiletype": 1,  # Reduced resolution image (standard for DNG previews)
        "subifds": 1,  # Has main image as subifd
    }
    # set datasize to max uncompressed size to avoid writing strips
    datasize = thumbnail_image.shape[0] * thumbnail_image.shape[1] * 3
    writer.write(
        thumbnail_image,  # Use the thumbnail_image directly
        **thumb_ifd_args,
        rowsperstrip=datasize,
    )

def write_dng(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    cfa_pattern: str = "RGGB",
    camera_profile: Optional[MetadataTags] = None,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    color_data: Optional[np.ndarray] = None
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        raw_data: Raw image data as numpy array (H, W)
        destination_file: Path or io.BytesIO object where to save the DNG file.
        bits_per_pixel: Number of bits per pixel (e.g. 12, 14, 16)
        cfa_pattern: CFA pattern string, e.g., 'RGGB'
        jxl_distance: JPEG XL Butteraugli distance. Lower is higher quality.
                     Default: None (no JXL compression).
        jxl_effort: JPEG XL compression effort (1-9). Higher is more compression/slower.
                    Only used if jxl_distance is also specified. Default: None (codec default).
        color_data: Optional color data for preview
        camera_profile: Optional MetadataTags instance to override default 
                        color matrix and illuminant
    """

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG to {destination_file}")
    else:
        logger.debug("Writing DNG to in-memory buffer")

    # TODO: implement param validation here - raw_data not none, W/H even, cfa pattern valid, bpp <= 16

    if raw_data.ndim != 2:
        raise ValueError(f"Expected 2D raw_data (height, width), got shape {raw_data.shape}")

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        bits_per_pixel = 16
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        bits_per_pixel = 8
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    dng_tags = _create_dng_tags(camera_profile, has_jxl=jxl_distance is not None)

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:
            if color_data is not None:
                _write_thumbnail_ifd(tif, color_data, dng_tags)
    
            # prepare main image
            dng_cfa_tags = MetadataTags()
            dng_cfa_tags.add_cfa_pattern_tag(cfa_pattern)
            dng_cfa_tags.add_tag(("CFARepeatPatternDim", "H", 2, (2, 2)))
            dng_cfa_tags.add_tag(("CFAPlaneColor", "B", 3, (0, 1, 2)))

            if jxl_distance is not None:
                if not (0.0 <= jxl_distance <= 15.0):
                    logger.warning(
                        f"JXL distance {jxl_distance} is outside the "
                        f"typical range [0.0, 15.0]. "
                    )
                # For JXL, distance 0.0 is mathematically lossless.
                # A distance of 1.0 is visually lossless for most people.
                # Distances around 1.5-2.0 are good for high quality lossy.
                # tifffile uses imagecodecs, which uses libjxl.
                # libjxl effort values are typically 1 (fastest) to 9 (most effort).
                # Default effort in libjxl is 7 ('falcon') if not specified.
                compression_type = "JPEGXL_DNG"  # Ensure DNG-specific JXL type
                actual_effort = jxl_effort if jxl_effort is not None else 5
                compressionargs = {
                    "distance": jxl_distance,
                    "effort": actual_effort
                }
                logger.debug(
                    f"Attempting to write DNG with JXL compression, distance: {jxl_distance}, effort: {actual_effort}"
                )

                # if compressing, need to swizzle the CFA data and indicate
                # this via tags
                processed_raw_data = swizzle_cfa_data(processed_raw_data)
                dng_cfa_tags.add_tag(("ColumnInterleaveFactor", "H", 1, 2))
                dng_cfa_tags.add_tag(("RowInterleaveFactor", "H", 1, 2))
                dng_cfa_tags.add_tag(("JXLDistance", "f", 1, jxl_distance))
                dng_cfa_tags.add_tag(("JXLEffort", "I", 1, actual_effort))
            else:
                compression_type = COMPRESSION.NONE
                compressionargs = {}

            if color_data is None:
                dng_cfa_tags.extend(dng_tags)
            
            # Prepare main image arguments
            main_image_ifd_args = {
                "subfiletype": 0,
                "photometric": "cfa",
                "subifds": 0,
                "compression": compression_type,
                "compressionargs": compressionargs,
                "extratags": dng_cfa_tags
            }

            # Write Main Raw Image to IFD
            raw_datasize = int(
                processed_raw_data.shape[0] * processed_raw_data.shape[1] * bits_per_pixel / 8
            )
            tif.write(processed_raw_data, **main_image_ifd_args, rowsperstrip=raw_datasize)

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise

def write_dng_linearraw(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    camera_profile: Optional[MetadataTags] = None,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    color_data: Optional[np.ndarray] = None,
) -> None:
    """Write a demosaiced, linear 3-band image as a LinearRaw DNG IFD.

    Differences vs write_dng:
    - Input is a demosaiced linear image with shape (H, W, 3).
    - PhotometricInterpretation is LinearRaw (34892).
    - No CFA tags; minimal extratags, only JXL-related ones when used.
    - Optional thumbnail SubIFD is supported via color_data.
    """

    if isinstance(destination_file, Path):
        logger.debug(f"Writing LinearRaw DNG to {destination_file}")
    else:
        logger.debug("Writing LinearRaw DNG to in-memory buffer")

    # Validate input raw_data
    if raw_data is None or raw_data.ndim != 3 or raw_data.shape[-1] != 3:
        raise ValueError(f"Expected 3-band image with shape (H, W, 3), got {None if raw_data is None else raw_data.shape}")

    # Prepare dtype according to bits_per_pixel; allow float16/32 as-is
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        bits_per_pixel = 16
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        bits_per_pixel = 8
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:
            # Prepare base DNG tags once (same as write_dng)
            dng_tags = _create_dng_tags(camera_profile=camera_profile, has_jxl=jxl_distance is not None)
            # Optional thumbnail SubIFD
            if color_data is not None:
                _write_thumbnail_ifd(tif, color_data, dng_tags)

            # Minimal extratags and compression settings (JXL optional)
            linearraw_tags = MetadataTags()
            compression_type = COMPRESSION.NONE
            compressionargs = {}
            if jxl_distance is not None:
                if not (0.0 <= jxl_distance <= 15.0):
                    logger.warning(
                        f"JXL distance {jxl_distance} is outside the typical range [0.0, 15.0]."
                    )
                actual_effort = jxl_effort if jxl_effort is not None else 5
                linearraw_tags.add_tag(("JXLDistance", "f", 1, jxl_distance))
                linearraw_tags.add_tag(("JXLEffort", "I", 1, actual_effort))
                compression_type = "JPEGXL_DNG"
                compressionargs = {"distance": jxl_distance, "effort": actual_effort}

            # Prepare main image IFD args
            # Include general DNG tags similar to write_dng when no thumbnail
            if color_data is None:
                linearraw_tags.extend(dng_tags)
            main_ifd_args = {
                "subfiletype": 0,
                "photometric": "linear_raw",
                "planarconfig": 1,  # CONTIG
                "compression": compression_type,
                "compressionargs": compressionargs,
                "extratags": linearraw_tags,
            }

            # Calculate rowsperstrip to avoid many strips (mirror write_dng)
            samples_per_pixel = 3
            raw_datasize = int(
                processed_raw_data.shape[0] * processed_raw_data.shape[1] * samples_per_pixel * (bits_per_pixel / 8)
            )

            tif.write(processed_raw_data, **main_ifd_args, rowsperstrip=raw_datasize)

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote LinearRaw DNG to {destination_file}")
        else:
            logger.debug("Successfully wrote LinearRaw DNG to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving LinearRaw DNG file with TiffWriter: {e}")
        raise

def write_dng_from_page(
    page: TiffPage,
    destination_file: Union[Path, io.BytesIO],
    camera_profile: Optional[MetadataTags] = None,
    color_data: Optional[np.ndarray] = None,
    skip_tags: Optional[set[str]] = None
) -> None:
    """Write DNG file by copying compressed raw data from an existing TiffPage.
    
    This function efficiently copies compressed raw data without decompression/recompression,
    preserving the original compression while updating metadata and optionally adding thumbnails.
    Works with CFA, LINEAR_RAW, and other raw photometric types.
    
    Args:
        page: TiffPage containing the raw data to copy
        destination_file: Path or io.BytesIO object where to save the DNG file
        camera_profile: Optional camera profile metadata to include
        color_data: Optional thumbnail/preview image data
        skip_tags: Optional set of tag names to skip when copying (e.g., for debugging)
    """
    # the tag names are from tifffile.py TiffTagRegistry
    # the tag types are from tifffile.py DATA_DTYPES
    has_jxl = page.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG)
    dng_tags = _create_dng_tags(camera_profile, has_jxl)

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG from TiffPage to {destination_file}")
    else:
        logger.debug("Writing DNG from TiffPage to in-memory buffer")

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder=page.parent.byteorder) as tif:

            if color_data is not None:
                _write_thumbnail_ifd(tif, color_data, dng_tags)

            dng_cfa_tags = MetadataTags()
            
            # Tags that TiffWriter handles automatically - don't copy as extratags
            dont_copy_tags = {
                'NewSubfileType',      # 254 - handled by subfiletype parameter
                'ImageWidth',          # 256 - handled by shape parameter
                'ImageLength',         # 257 - handled by shape parameter
                'Compression',         # 259 - handled by compression parameter
                'PhotometricInterpretation', # 262 - handled by photometric parameter
                'ImageDescription',    # 270 - handled by description parameter
                'StripOffsets',        # 273 - handled automatically by TiffWriter
                'SamplesPerPixel',     # 277 - handled by dtype/shape parameters
                'RowsPerStrip',        # 278 - handled by rowsperstrip parameter
                'StripByteCounts',     # 279 - handled automatically by TiffWriter
                'XResolution',         # 282 - handled by resolution parameter
                'YResolution',         # 283 - handled by resolution parameter
                'TileOffsets',         # 284 - handled internally by tifffile
                'ResolutionUnit',      # 296 - handled by resolution parameter
                'Software',            # 305 - handled by software parameter
                'SubIFDs'              # 330 - handled internally by tifffile
            }
            
            # Combine default skip tags with user-provided skip tags
            all_skip_tags = dont_copy_tags.copy()
            if skip_tags:
                all_skip_tags.update(skip_tags)
            
            # Copy page-specific tags (like the test pattern)
            if hasattr(page, 'tags'):
                for tag in page.tags.values():
                    # Skip tags that TiffWriter handles automatically or user wants to skip
                    if tag.name not in all_skip_tags:
                        # Use the same pattern as the original test: tag.code, tag.dtype, tag.count, tag.value
                        dng_cfa_tags.add_tag((tag.name, tag.dtype, tag.count, tag.value))

            if color_data is None:
                dng_cfa_tags.extend(dng_tags)
            
            # Read compressed raw data and create an iterator
            fh = page.parent.filehandle
            compressed_segments = list(fh.read_segments(
                page.dataoffsets,
                page.databytecounts,
                sort=True
            ))

            def compressed_data_iterator():
                try:
                    for segment_data, index in compressed_segments:
                        yield segment_data
                except GeneratorExit:
                    # Handle graceful shutdown
                    return

            logger.debug(f"Read {len(compressed_segments)} compressed segments from CFA page")

            # Prepare main image arguments
            main_image_ifd_args = {
                "subfiletype": 0,
                "photometric": page.photometric,
                "subifds": 0,
                "compression": page.compression,
                "extratags": dng_cfa_tags
            }

            # Determine shape based on samples per pixel (1 for CFA, 3 for LINEAR_RAW)
            samples_per_pixel = page.samplesperpixel if hasattr(page, 'samplesperpixel') else 1
            if samples_per_pixel > 1:
                write_shape = (page.imagelength, page.imagewidth, samples_per_pixel)
            else:
                write_shape = (page.imagelength, page.imagewidth)
            
            # Handle tiled vs stripped images
            if page.is_tiled:
                # Tiled image - use tile parameter
                tile_shape = (page.tilelength, page.tilewidth)
                tif.write(
                    data=compressed_data_iterator(),
                    shape=write_shape,
                    dtype=page.dtype,
                    bitspersample=page.bitspersample,
                    tile=tile_shape,
                    **main_image_ifd_args,
                )
            else:
                # Stripped image - use rowsperstrip
                raw_datasize = page.imagelength * page.imagewidth * samples_per_pixel * (page.bitspersample // 8)
                tif.write(
                    data=compressed_data_iterator(),
                    shape=write_shape,
                    dtype=page.dtype,
                    bitspersample=page.bitspersample,
                    rowsperstrip=raw_datasize,
                    **main_image_ifd_args,
                )
            
            logger.debug(f"Successfully copied compressed raw data ({sum(page.databytecounts)} bytes)")
        
        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")
        
    except Exception as e:
        logger.error(f"Error writing DNG from TiffPage: {e}")
        raise


class DngFile(TiffFile):

    """A TIFF file with DNG-specific extensions and helper methods."""

    def __init__(self, file, *args, **kwargs):
        import io
        from pathlib import Path
        
        # Normalize file input to BytesIO for consistent in-memory operation
        if isinstance(file, (str, Path)):
            # Read file into memory
            with open(file, 'rb') as f:
                file_data = f.read()
            file = io.BytesIO(file_data)
        elif isinstance(file, io.IOBase):
            # Ensure we're at the beginning for consistent behavior
            file.seek(0)
        # For any other type, let TiffFile handle it and potentially fail with a clear error
        
        super().__init__(file, *args, **kwargs)
        self._xmp_metadata = None  # Lazy-loaded XmpMetadata instance

    def _iter_all_pages_recursive(self, pages_list: Optional[List[TiffPage]]):
        """Recursively iterates through all TIFF pages, including nested ones."""
        if pages_list is None:
            return
        for page in pages_list:
            yield page
            if page.pages:  # Check if there are sub-pages
                yield from self._iter_all_pages_recursive(page.pages)

    def get_raw_pages_info(self) -> List[Tuple[int, tuple, Dict[str, Any]]]:
        """
        Returns info for pages with 'CFA' or 'LINEAR_RAW' interpretation.
        Each item in the list is a tuple: (page_id, shape, tags_dict).
        page_id is the 0-based index of the page in the flattened list of all TIFF pages.
        tags_dict contains relevant DNG tags from global and page-specific IFDs.
        """

        test_val = (TIFF.TAGS["Orientation"], "Orientation")

        global_tags = [
            "Make",
            "Model",
            "DNGVersion",
            "DNGBackwardVersion",
            "Orientation",
            # Color matrices
            "ColorMatrix1",
            "ColorMatrix2",
            "ColorMatrix3",
            "CalibrationIlluminant1",
            "CalibrationIlluminant2",
            "CalibrationIlluminant3",
            "AnalogBalance",
            "AsShotNeutral",
            "AsShotWhiteXY",
            # Forward matrices
            "ForwardMatrix1",
            "ForwardMatrix2",
            "ForwardMatrix3",
            # Camera calibration
            "CameraCalibration1",
            "CameraCalibration2",
            "CameraCalibration3",
            "CameraCalibrationSignature",
            "ProfileCalibrationSignature",
            # Exposure and rendering
            "BaselineExposure",
            "BaselineExposureOffset",
            "ShadowScale",
            "LinearResponseLimit",
            # Profile HueSatMap
            "ProfileHueSatMapDims",
            "ProfileHueSatMapData1",
            "ProfileHueSatMapData2",
            "ProfileHueSatMapData3",
            "ProfileHueSatMapEncoding",
            # Profile LookTable
            "ProfileLookTableDims",
            "ProfileLookTableData",
            "ProfileLookTableEncoding",
            # Profile tone and gain
            "ProfileToneCurve",
            "ProfileGainTableMap",
            "ProfileGainTableMap2",
            "DefaultBlackRender",
            # Opcode lists
            "OpcodeList1",
            "OpcodeList2",
            "OpcodeList3",
            # Reduction matrices
            "ReductionMatrix1",
            "ReductionMatrix2",
            "ReductionMatrix3",
            # Linearization
            "LinearizationTable",
            # RGB Tables
            "RGBTables",
        ]

        cfa_subifd_tags = [
            "PhotometricInterpretation",
            "Compression",
            "BitsPerSample",
            "SamplesPerPixel",
            "PlanarConfiguration",
            "CFARepeatPatternDim",
            "CFAPattern",
            "CFAPlaneColor",
            "BlackLevel",
            "BlackLevelRepeatDim",
            "BlackLevelDeltaH",
            "BlackLevelDeltaV",
            "WhiteLevel",
            "ColumnInterleaveFactor",
            "RowInterleaveFactor",
            "JXLDistance",
            "JXLEffort",
            "ActiveArea",
            "DefaultCropOrigin",
            "DefaultCropSize",
        ]

        info_list: List[Tuple[int, tuple, Dict[str, Any]]] = []
        global_tags_data: Dict[str, Any] = {}

        # 1. Read Global Tags from the first page (IFD0)
        if self.pages and len(self.pages) > 0:
            first_page = self.pages[0]
            for tag_name in global_tags:
                tag_id = TIFF.TAGS[tag_name]
                if tag_id in first_page.tags:
                    page_tag = first_page.tags[tag_id]
                    global_tags_data[tag_name] = translate_dng_tag(page_tag)

        # 2. Iterate through all pages
        for current_page_id, page in enumerate(self._iter_all_pages_recursive(self.pages)):
            if page.photometric and page.photometric.name in ("CFA", "LINEAR_RAW",):
                current_page_tags: Dict[str, Any] = {}
                # Start with a copy of global tags
                current_page_tags.update(global_tags_data)

                # Iterate through all tags on this page's IFD.
                # If the tag's name is in cfa_globaltags OR cfa_subifd_tags,
                # use its page-specific value, overwriting any global default.
                for lut in [global_tags, cfa_subifd_tags]:
                    for tag_name in lut:
                        tag_id = TIFF.TAGS[tag_name]
                        if tag_id in page.tags:
                            page_tag = page.tags[tag_id]
                            current_page_tags[tag_name] = translate_dng_tag(page_tag)

                info_list.append((current_page_id, page.shape, current_page_tags))

        return info_list

    def get_page_by_id(self, target_page_id: int) -> Optional[TiffPage]:
        """Retrieve a specific TiffPage by its flattened, 0-based ID."""
        for i, page in enumerate(self._iter_all_pages_recursive(self.pages)):
            if i == target_page_id:
                return page
        return None

    def get_raw_cfa_by_id(self, target_page_id: int) -> Optional[tuple[np.ndarray, str]]:
        """Retrieves the raw CFA array and CFAPattern for a specific 'CFA' page.
        
        Returns a tuple (raw_cfa_array, cfa_pattern_str) or None if page not found/invalid.
        """
        p = self.get_page_by_id(target_page_id)
        return self.get_raw_cfa_from_page(p)

    def get_raw_cfa_from_page(self, page: Optional[TiffPage]) -> Optional[tuple[np.ndarray, str]]:
        """Retrieves the raw CFA array and CFAPattern from a given TiffPage.
        
        Returns a tuple (raw_cfa_array, cfa_pattern_str) or None if page not valid/unsupported.
        """
        p = page
        if p is None or p.photometric is None or p.photometric.name != "CFA":
            return None

        # Determine if data is interleaved and needs deswizzling
        col_interleave_tag = p.tags.get(TIFF.TAGS["ColumnInterleaveFactor"])
        row_interleave_tag = p.tags.get(TIFF.TAGS["RowInterleaveFactor"])
        if (
            col_interleave_tag is not None
            and col_interleave_tag.value == 2
            and row_interleave_tag is not None
            and row_interleave_tag.value == 2
        ):
            raw_cfa = deswizzle_cfa_data(p.asarray())
        else:
            raw_cfa = p.asarray()

        # Fetch CFAPattern string representation
        cfa_tag = p.tags.get(TIFF.TAGS.get("CFAPattern"))
        cfa_str = None
        if cfa_tag is not None:
            v = cfa_tag.value
            if isinstance(v, (bytes, bytearray)) and len(v) == 4:
                cfa_codes = tuple(int(b) for b in v)
                cfa_str = INVERSE_BAYER_PATTERN_MAP.get(cfa_codes)

        return raw_cfa, cfa_str

    def get_raw_linear_by_id(self, target_page_id: int) -> Optional[np.ndarray]:
        """Retrieves the raw data array for a specific 'LINEAR_RAW' page by its ID.
        
        For single-tile JPEG XL images, uses imagecodecs directly to avoid
        tifffile's chroma subsampling limitation. Per tifffile documentation:
        "chroma subsampling without JPEG compression" is not implemented,
        meaning JPEG XL with 4:2:0 or 4:2:2 chroma subsampling may not be
        correctly upsampled by tifffile.asarray().
        
        For multi-tile JPEG XL, we use tifffile.asarray() which correctly
        assembles tiles - each tile is a separate JPEG XL bitstream that
        must be decoded and positioned individually.
        """

        p = self.get_page_by_id(target_page_id)

        if p is None or p.photometric is None or p.photometric.name != "LINEAR_RAW":
            return None

        # Check if this is JPEG XL compression
        if p.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
            # Use imagecodecs directly to handle chroma subsampling correctly
            fh = p.parent.filehandle
            
            if len(p.dataoffsets) == 1:
                # Single tile/strip
                compressed_segments = list(fh.read_segments(
                    p.dataoffsets,
                    p.databytecounts,
                    sort=True
                ))
                compressed_data = b''.join(segment_data for segment_data, index in compressed_segments)
                return imagecodecs.jpegxl_decode(compressed_data)
            else:
                # Multiple tiles - decode each and assemble
                tile_width = p.tilewidth
                tile_height = p.tilelength
                img_width = p.imagewidth
                img_height = p.imagelength
                samples = p.samplesperpixel or 3
                
                # Determine output dtype from first tile
                fh.seek(p.dataoffsets[0])
                first_tile_data = fh.read(p.databytecounts[0])
                first_tile = imagecodecs.jpegxl_decode(first_tile_data)
                dtype = first_tile.dtype
                
                # Create output array
                output = np.zeros((img_height, img_width, samples), dtype=dtype)
                
                # Decode and place each tile
                tiles_x = (img_width + tile_width - 1) // tile_width
                for i, (offset, bytecount) in enumerate(zip(p.dataoffsets, p.databytecounts)):
                    fh.seek(offset)
                    tile_data = fh.read(bytecount)
                    tile = imagecodecs.jpegxl_decode(tile_data)
                    
                    ty = (i // tiles_x) * tile_height
                    tx = (i % tiles_x) * tile_width
                    
                    # Handle edge tiles that may be smaller
                    th = min(tile_height, img_height - ty)
                    tw = min(tile_width, img_width - tx)
                    output[ty:ty+th, tx:tx+tw] = tile[:th, :tw]
                
                return output
        else:
            # Use asarray() for other compressions
            return p.asarray()

    def get_tag(
        self,
        tag_name: str,
        ifd: Optional[int] = None,
        return_type: Optional[Type] = None,
    ) -> Union[Any, None]:
        """Retrieves a metadata tag's value from the DNG file.

        Args:
            tag_name: The name of the tag to retrieve (e.g., "ExposureTime").
            ifd: Optional integer specifying the IFD to search. If None, all IFDs are searched.
            return_type: The desired type for the return value (e.g., float, int, str).

        Returns:
            The tag's value, converted to `return_type` if possible. Returns None if the tag
            is not found or if type conversion fails.
        """
        try:
            tag_id = TIFF.TAGS[tag_name]
        except KeyError:
            logger.warning(f"Tag '{tag_name}' not found in TIFF tag registry.")
            return None

        # If specific IFD requested, search only that IFD; otherwise search all pages recursively
        if ifd is not None and ifd < len(self.pages):
            pages_to_search = [self.pages[ifd]]
        else:
            pages_to_search = list(self._iter_all_pages_recursive(self.pages))

        for page in pages_to_search:
            if tag_id in page.tags:
                tag = page.tags[tag_id]
                value = tag.value

                if return_type is None:
                    return value

                try:
                    if return_type is float and isinstance(value, tuple) and len(value) == 2:
                        # Handle rational to float conversion
                        num, den = value
                        return float(num / den) if den != 0 else 0.0
                    else:
                        return return_type(value)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Could not convert tag '{tag_name}' value '{value}' to type {return_type}: {e}"
                    )
                    return None

        # Fallback: search ExifIFD if available
        exif_value = self._get_exif_tag(tag_name, return_type)
        if exif_value is not None:
            return exif_value

        return None

    def _get_exif_tag(self, tag_name: str, return_type: Optional[Type] = None) -> Union[Any, None]:
        """Search for a tag in the ExifIFD using tifffile's parsed structure.
        
        Args:
            tag_name: The name of the tag to retrieve (e.g., "ExposureTime").
            return_type: The desired type for the return value (e.g., float, int, str).
            
        Returns:
            The tag's value from ExifIFD, converted to return_type if possible. 
            Returns None if not found in ExifIFD.
        """
        # Look for ExifIFDPointer (tag 34665) on the first page (IFD0)
        if not self.pages:
            return None
            
        first_page = self.pages[0]
        exif_ptr_tag_id = TIFF.TAGS.get("ExifTag")  # ExifIFDPointer
        if exif_ptr_tag_id is None:
            return None
        exif_ptr_tag = first_page.tags.get(exif_ptr_tag_id)
        
        if exif_ptr_tag is None:
            return None
            
        # tifffile parses ExifIFD into a dict accessible via tag.value
        exif_ifd_dict = exif_ptr_tag.value
        if not isinstance(exif_ifd_dict, dict):
            return None
            
        # Search for our tag by name in the ExifIFD dict
        if tag_name not in exif_ifd_dict:
            return None
            
        value = exif_ifd_dict[tag_name]
        
        if return_type is None:
            return value
            
        try:
            if return_type is float and isinstance(value, tuple) and len(value) == 2:
                # Handle rational to float conversion
                num, den = value
                return float(num / den) if den != 0 else 0.0
            else:
                return return_type(value)
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Could not convert ExifIFD tag '{tag_name}' value '{value}' to type {return_type}: {e}"
            )
            return None

    @property
    def xmp_metadata(self) -> XmpMetadata:
        """Lazy-loaded XmpMetadata instance.
        
        Returns:
            XmpMetadata instance for querying XMP attributes. Returns an empty instance if no XMP data found.
        """
        if self._xmp_metadata is None:
            xmp_string = self.get_tag('XMP', return_type=str) or ''
            self._xmp_metadata = XmpMetadata(xmp_string)
        return self._xmp_metadata

    def xmp_has(self, prop: str) -> bool:
        """Check if an XMP property exists.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
        
        Returns:
            True if the property exists in XMP metadata
        """
        return self.xmp_metadata.has_prop(prop)

    def xmp(self, prop: str, return_type: Optional[Type] = None) -> Optional[Any]:
        """Get an XMP property value with optional type conversion.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
            return_type: Optional type to convert the value to (e.g., float, int)
        
        Returns:
            The property value, optionally converted to return_type. None if not found.
        """
        return self.xmp_metadata.get_prop(prop, return_type)


def decode_raw(
    file: Union[str, Path, IO[bytes], DngFile],
    use_xmp: bool = True,
    output_dtype: type = np.uint16,
    **processing_params
) -> np.ndarray:
    """
    Decode a DNG file to a numpy array using Core Image processing.
    
    Args:
        file: Path to DNG file, file-like object containing DNG data, or DngFile instance
        use_xmp: Whether to read XMP metadata for default values
        output_dtype: Output numpy data type (np.uint8, np.uint16, np.float16, np.float32)
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
    
    Returns:
        RGB image array with shape (height, width, 3) and specified dtype
    """
    try:
        # Import here to avoid importing heavy dependencies at module load time
        from . import color_mac
        
        # Create or use DngFile - DngFile.__init__ handles file normalization and seeking
        dng_file = file if isinstance(file, DngFile) else DngFile(file)
        dng_input = dng_file.filehandle
        
        # Build processing options using configuration mapping
        options = {}
        
        # Only process parameters if we have XMP to read or explicit parameters to apply
        if use_xmp or processing_params:
            from .color import SplineCurve as SC
            
            # Define mapping: param_name -> (option_name, xmp_name, value_type)
            # Use None for xmp_name to indicate CLI-only parameters (no XMP fallback)
            xmp_mappings = {
                "temperature": ("neutralTemperature", "Temperature", float),
                "tint": ("neutralTint", "Tint", float),
                "exposure": ("exposure", "Exposure2012", float),
                "tone_curve": ("toneCurve", "ToneCurvePV2012", SC),
                "orientation": ("imageOrientation", None, int),
                "noise_reduction": ("luminanceNoiseReductionAmount", None, float),
            }
            
            # Process each mapping: CLI params first, then XMP fallback
            for param_name, (option_name, xmp_name, value_type) in xmp_mappings.items():
                param_value = processing_params.get(param_name)
                # Use CLI parameter if provided (highest priority)
                if param_value is not None:
                    options[option_name] = param_value
                # Otherwise use XMP default if available, requested, and xmp_name is specified
                elif xmp_name is not None and use_xmp and dng_file.xmp_has(xmp_name):
                    options[option_name] = dng_file.xmp(xmp_name, value_type)

            # Add noise reduction to both luminance and color (convention)
            if ("noise_reduction" in processing_params and 
                processing_params["noise_reduction"] is not None):
                options["colorNoiseReductionAmount"] = processing_params["noise_reduction"]
        
        # Format file and options for logging
        import io
        if isinstance(file, io.BytesIO):
            file_desc = "BytesIO buffer"
        elif isinstance(file, (str, Path)):
            file_desc = str(file)
        else:
            file_desc = type(file).__name__
        
        formatted_opts = {}
        for key, value in options.items():
            if isinstance(value, (float, np.floating)):
                formatted_opts[key] = f"{float(value):.3f}"
            else:
                formatted_opts[key] = value
        logger.debug(f"Processing {file_desc} with options: {formatted_opts}")
        
        # Special case: apply lin2srgb transformation to tone curve points
        # TODO: since CI is an opaque RAW pipeline it is not possible to sequence tone curve 
        #       in the same way in Photoshop and CI so we apply the transformation here.
        if "toneCurve" in options and options["toneCurve"] is not None:

            def lin2srgb(lin):
                if lin > 0.0031308:
                    s = 1.055 * (pow(lin, (1.0 / 2.4))) - 0.055
                else:
                    s = 12.92 * lin
                return s

            spline_curve = options["toneCurve"]
            # Points are already normalized 0-1, apply lin2srgb transformation
            transformed_points = [
                (lin2srgb(x), lin2srgb(y)) for x, y in spline_curve.points
            ]
            # Create new SplineCurve with transformed points
            options["toneCurve"] = SC(transformed_points)

        # Ensure file pointer is at beginning for Core Image processing
        # (XMP reading above may have moved the pointer)
        dng_input.seek(0) 

        # Process with Core Image
        color_data = color_mac.process_raw_core_image(
            dng_input=dng_input,
            raw_filter_options=options,
            use_gpu=True,
            output_dtype=output_dtype,
        )
        
        if color_data is None:
            raise RuntimeError(f"Failed to process DNG file: {file}")
        
        logger.debug(f"Successfully decoded DNG to array with shape {color_data.shape} and dtype {color_data.dtype}")
        return color_data
                
    except Exception as e:
        logger.error(f"Error decoding {file}: {e}", exc_info=True)
        raise


def convert_raw(
    file: Union[str, Path, IO[bytes]],
    output_path: Union[str, Path],
    use_xmp: bool = True,
    output_dtype: type = np.uint8,
    **processing_params
) -> bool:
    """
    Convert a DNG file to an image file.
    
    Args:
        file: Path to DNG file or file-like object containing DNG data
        output_path: Output file path (format determined by extension)
        use_xmp: Whether to read XMP metadata for default values
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import cv2
        
        # Decode DNG to numpy array
        color_data = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        
        # Save to file
        output_path = Path(output_path)
        
        # Save to file
        success = cv2.imwrite(str(output_path), bgr_image)
        
        if success:
            logger.info(f"Successfully converted {file} to {output_path}")
            return True
        else:
            logger.error(f"Failed to save file: {output_path}")
            return False
                
    except Exception as e:
        logger.error(f"Error converting {file}: {e}", exc_info=True)
        raise


def convert_raw_to_stream(
    file: Union[str, Path, IO[bytes]],
    output_format: str = "jpg",
    use_xmp: bool = True,
    output_dtype: type = np.uint8,
    **processing_params
) -> bytes:
    """
    Convert a DNG file to encoded image bytes.
    
    Args:
        file: Path to DNG file or file-like object containing DNG data
        output_format: Output format ("jpg", "png", "tiff", etc.)
        use_xmp: Whether to read XMP metadata for default values
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
        
    Returns:
        bytes: Encoded image data
    """
    try:
        import cv2
        
        # Decode DNG to numpy array
        color_data = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        
        # Ensure format has leading dot
        format_ext = f".{output_format.lstrip('.')}"
        
        # Encode image
        success, encoded_buffer = cv2.imencode(format_ext, bgr_image)
        
        if success:
            encoded_bytes = encoded_buffer.tobytes()
            logger.info(f"Encoded {file} to {len(encoded_bytes)} bytes as {output_format}")
            return encoded_bytes
        else:
            logger.error(f"Failed to encode image as {output_format}")
            raise RuntimeError(f"Failed to encode image as {output_format}")
                
    except Exception as e:
        logger.error(f"Error converting {file} to stream: {e}", exc_info=True)
        raise


