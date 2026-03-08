"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import io
import logging
import imagecodecs
import numpy as np
from datetime import datetime

from pathlib import Path
from tifffile import COMPRESSION, PHOTOMETRIC, TiffFile, TiffPage, TiffWriter, TIFF
from typing import Optional, Union, List, Dict, Tuple, Any, Type, IO

logger = logging.getLogger(__name__)

# Import metadata classes from tiff_metadata module
from .tiff_metadata import (
    MetadataTags,
    TagSpec,
    TIFF_DTYPE_TO_STR,
    XmpMetadata,
    get_cfa_pattern_codes,
    get_native_type,
    decode_tag_value,
    resolve_tag,
    LOCAL_TIFF_TAGS,
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
        # Get the main image page (should be CFA)
        cfa_page = dng_file.get_main_page()
        if cfa_page is None or not cfa_page.is_cfa:
            raise ValueError("No CFA main page found in DNG")

        # Get the CFA data array and pattern
        result = cfa_page.get_raw_cfa()
        if result is None:
            raise RuntimeError("Failed to retrieve raw CFA data")
        raw_cfa, cfa_pattern_value = result

        if cfa_pattern_value is None:
            raise ValueError("Missing CFAPattern tag")

    except (ValueError, RuntimeError):
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
    cfa_pattern_flat = get_cfa_pattern_codes(cfa_pattern_str)
    cfa_pattern = np.array(cfa_pattern_flat).reshape(2, 2)
    
    # Extract R, G, B planes based on their positions in the CFA pattern
    # CFA codes are: 0=R, 1=G, 2=B
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
    cfa_pattern_flat = get_cfa_pattern_codes(cfa_pattern_str)
    
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
    dng_tags.add_tag("PreviewColorSpace", PREVIEWCOLORSPACE_SRGB)

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
        dng_tags.add_tag("Orientation", ORIENTATION_HORIZONTAL)
    
    if "ColorMatrix1" not in dng_tags:
        identity_matrix = np.identity(3, dtype=np.float64)
        dng_tags.add_tag("ColorMatrix1", identity_matrix)
    
    if "CalibrationIlluminant1" not in dng_tags:
        dng_tags.add_tag("CalibrationIlluminant1", 0)  # 0 = Unknown

    dng_tags.add_tag("DNGVersion", bytes([1, 7, 1, 0]))
    if not has_jxl:
        # need latest version for CFA compression but lots of old software can't handle it
        dng_tags.add_tag("DNGBackwardVersion", bytes([1, 4, 0, 0]))
    else:
        dng_tags.add_tag("DNGBackwardVersion", bytes([1, 7, 1, 0]))
        
    return dng_tags


def _write_thumbnail_ifd(writer: TiffWriter, thumbnail_image: np.ndarray, dng_tags: MetadataTags) -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags
    dng_tags.add_tag("PreviewColorSpace", PREVIEWCOLORSPACE_SRGB)

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
            dng_cfa_tags.add_tag("CFAPattern", cfa_pattern)
            dng_cfa_tags.add_tag("CFARepeatPatternDim", (2, 2))
            dng_cfa_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))

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
                dng_cfa_tags.add_tag("ColumnInterleaveFactor", 2)
                dng_cfa_tags.add_tag("RowInterleaveFactor", 2)
                dng_cfa_tags.add_tag("JXLDistance", jxl_distance)
                dng_cfa_tags.add_tag("JXLEffort", actual_effort)
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
                linearraw_tags.add_tag("JXLDistance", jxl_distance)
                linearraw_tags.add_tag("JXLEffort", actual_effort)
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
    
    # Tags that TiffWriter handles automatically - don't copy as extratags
    dont_copy_tags = {
        'NewSubfileType',
        'ImageWidth',
        'ImageLength',
        'BitsPerSample',
        'Compression',
        'PhotometricInterpretation',
        'ImageDescription',
        'StripOffsets',
        'SamplesPerPixel',
        'RowsPerStrip',
        'StripByteCounts',
        'XResolution',
        'YResolution',
        'PlanarConfiguration',
        'ResolutionUnit',
        'Software',
        'TileWidth',
        'TileLength',
        'TileOffsets',
        'TileByteCounts',
        'SubIFDs',
        'ExifTag',
    }
    if skip_tags:
        dont_copy_tags.update(skip_tags)
    
    # Build dng_tags from parent IFD0 (color calibration, camera info, etc.)
    dng_tags = MetadataTags()
    parent_ifd0 = page.parent.pages[0]
    for tag in parent_ifd0.tags.values():
        if tag.name not in dont_copy_tags:
            dng_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)
    
    # Apply camera_profile overrides on top (if provided)
    if camera_profile is not None:
        for code, stored_tag in camera_profile._tags.items():
            dng_tags.add_raw_tag(code, stored_tag.dtype, stored_tag.count, stored_tag.value)

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG from TiffPage to {destination_file}")
    else:
        logger.debug("Writing DNG from TiffPage to in-memory buffer")

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder=page.parent.byteorder) as tif:

            if color_data is not None:
                _write_thumbnail_ifd(tif, color_data, dng_tags)

            dng_cfa_tags = MetadataTags()
            
            # Copy SubIFD page tags (CFAPattern, etc.)
            for tag in page.tags.values():
                if tag.name not in dont_copy_tags:
                    dng_cfa_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)

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
            # Forward tag values via tif.write parameters (not extratags)
            main_image_ifd_args = {
                "subfiletype": 0,
                "photometric": page.photometric,
                "subifds": 0,
                "compression": page.compression,
                "extratags": dng_cfa_tags,
            }
            
            # Add optional parameters from parent IFD0 tags (these are global tags)
            parent_tags = parent_ifd0.tags
            if 'Software' in parent_tags:
                main_image_ifd_args["software"] = parent_tags['Software'].value
            if 'XResolution' in parent_tags and 'YResolution' in parent_tags:
                x_res = parent_tags['XResolution'].value
                y_res = parent_tags['YResolution'].value
                # tifffile expects (x, y) tuple or single value
                if isinstance(x_res, tuple):
                    x_res = x_res[0] / x_res[1] if x_res[1] else x_res[0]
                if isinstance(y_res, tuple):
                    y_res = y_res[0] / y_res[1] if y_res[1] else y_res[0]
                main_image_ifd_args["resolution"] = (x_res, y_res)
            if 'ResolutionUnit' in parent_tags:
                main_image_ifd_args["resolutionunit"] = parent_tags['ResolutionUnit'].value

            # Determine shape based on samples per pixel (1 for CFA, 3 for LINEAR_RAW)
            samples_per_pixel = page.samplesperpixel if hasattr(page, 'samplesperpixel') else 1
            if samples_per_pixel > 1:
                write_shape = (page.imagelength, page.imagewidth, samples_per_pixel)
            else:
                write_shape = (page.imagelength, page.imagewidth)
            
            # Handle tiled vs stripped images
            if page.is_tiled:
                tile_shape = (page.tilelength, page.tilewidth)
                # tifffile requires: tile <= image AND tile dimensions multiple of 16.
                # Some cameras use non-standard tiles for small previews.
                # We cannot reinterpret tile-compressed data as strips, so raise an error.
                tile_valid = (
                    tile_shape[0] <= page.imagelength and 
                    tile_shape[1] <= page.imagewidth and
                    tile_shape[0] % 16 == 0 and 
                    tile_shape[1] % 16 == 0
                )
                if not tile_valid:
                    raise ValueError(
                        f"Cannot copy tiled page: tile {tile_shape} not supported by tifffile "
                        f"(must be <= image size and multiple of 16)"
                    )
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


class DngPage:
    """Wrapper around a tifffile TiffPage with DNG-specific functionality.
    
    Provides convenient access to DNG tags with automatic translation,
    parent IFD tag inheritance, and raw data extraction methods.
    """
    
    # NewSubFileType values from DNG spec
    SF_MAIN_IMAGE = 0
    SF_PREVIEW_IMAGE = 1
    SF_TRANSPARENCY_MASK = 2
    SF_PREVIEW_MASK = 8
    SF_ALT_PREVIEW_IMAGE = 65537
    
    def __init__(self, tiff_page: TiffPage, page_id: int, parent: Optional['DngPage'] = None):
        """Initialize DngPage wrapper.
        
        Args:
            tiff_page: The underlying tifffile TiffPage object
            page_id: The flattened 0-based page index
            parent: Optional parent DngPage for tag inheritance
        """
        self._page = tiff_page
        self.page_id = page_id
        self.parent = parent
    
    @property
    def shape(self) -> tuple:
        """Image dimensions as (height, width) or (height, width, samples)."""
        return self._page.shape
    
    @property
    def byteorder(self) -> str:
        """File byte order ('>' for big-endian, '<' for little-endian)."""
        return self._page.parent.byteorder
    
    @property
    def tags(self):
        """Direct access to underlying TiffPage tags dict."""
        return self._page.tags
    
    @property
    def photometric(self) -> Optional[str]:
        """Photometric interpretation string (e.g., 'CFA', 'LINEAR_RAW', 'RGB')."""
        if self._page.photometric is not None:
            return self._page.photometric.name
        return None
    
    @property
    def is_cfa(self) -> bool:
        """True if this page contains CFA (Bayer) raw data."""
        return self.photometric == "CFA"
    
    @property
    def is_linear_raw(self) -> bool:
        """True if this page contains LINEAR_RAW (demosaiced) data."""
        return self.photometric == "LINEAR_RAW"
    
    @property
    def is_main_image(self) -> bool:
        """True if this page is the main image (NewSubfileType == 0)."""
        tag = self._page.tags.get(254)  # NewSubfileType tag ID
        if tag is None:
            return False
        return tag.value == self.SF_MAIN_IMAGE
    
    @property
    def is_preview(self) -> bool:
        """True if this page is a preview image."""
        tag = self._page.tags.get(254)  # NewSubfileType tag ID
        if tag is None:
            return False
        return tag.value in (self.SF_PREVIEW_IMAGE, self.SF_ALT_PREVIEW_IMAGE)
    
    @property
    def xmp(self) -> XmpMetadata:
        """XMP metadata from this page (or parent if not found locally)."""
        xmp_string = self.get_tag('XMP', str)
        if xmp_string is None:
            xmp_string = ''
        return XmpMetadata(xmp_string)
    
    def get_tag(
        self, 
        tag: Union[str, int], 
        return_type: Optional[type] = None
    ) -> Optional[Any]:
        """Get a tag value with type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string 
                 (e.g., 'ColorMatrix1', 'CFAPattern')
            return_type: Controls type conversion:
                - None (default): auto-convert to native type from TagSpec
                - specific type (str, float, list, etc.): convert to that type
        
        Returns:
            Tag value (converted based on return_type), or None if not found.
            
        See also:
            get_raw_tag: Returns raw tag value without any conversion.
        """
        # Resolve tag to id/name and get registry spec (for shape info)
        if isinstance(tag, int):
            tag_id = tag
            _, tag_name, registry_spec = resolve_tag(tag)
            if tag_name is None:
                tag_name = str(tag)
        else:
            tag_id, tag_name, registry_spec = resolve_tag(tag)
            if tag_id is None:
                logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
                return None
        
        # Get value from page or parent
        if tag_id in self._page.tags:
            raw_tag = self._page.tags[tag_id]
            effective_type = return_type or get_native_type(raw_tag.dtype, raw_tag.count)
            
            # Use registry spec shape only if count matches tag
            shape_spec = None
            if registry_spec and registry_spec.shape and registry_spec.count == raw_tag.count:
                shape_spec = TagSpec(TIFF_DTYPE_TO_STR.get(raw_tag.dtype, 'B'), raw_tag.count, registry_spec.shape)
            
            return decode_tag_value(tag_name, raw_tag.value, raw_tag.dtype, shape_spec, effective_type)
        elif self.parent is not None:
            return self.parent.get_tag(tag, return_type)
        
        return None

    def get_raw_tag(self, tag: Union[str, int]) -> Optional[Any]:
        """Get raw tag value without any type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Raw tag value as stored, or None if not found.
            
        See also:
            get_tag: Returns tag value with automatic or specified type conversion.
        """
        # For raw access, we can work with tags not in the registry
        if isinstance(tag, int):
            tag_id = tag
        else:
            tag_id, _, _ = resolve_tag(tag)
            if tag_id is None:
                logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
                return None
        
        if tag_id in self._page.tags:
            return self._page.tags[tag_id].value
        elif self.parent is not None:
            return self.parent.get_raw_tag(tag)
        
        return None
    
    def get_time_from_tags(self, time_type: str = "original") -> Optional[datetime]:
        """Extract datetime from EXIF time tags with subseconds and timezone.
        
        This is the inverse of MetadataTags.add_time_tags(). Reads the datetime,
        subsecond, and timezone offset tags and combines them into a single
        datetime object.
        
        Args:
            time_type: Which set of tags to read:
                - "original" (default): DateTimeOriginal, SubsecTimeOriginal, OffsetTimeOriginal
                - "digitized": DateTimeDigitized, SubsecTimeDigitized, OffsetTimeDigitized
                - "modified": DateTime, SubsecTime, OffsetTime
                - "preview": PreviewDateTime (no subsec/offset tags)
                
        Returns:
            datetime object with microseconds and timezone if available, or None if
            the datetime tag is not present or cannot be parsed.
            
        Example:
            capture_time = page.get_time("original")
            if capture_time:
                print(f"Captured at {capture_time}")
        """
        from .tiff_metadata import _get_time_impl
        return _get_time_impl(self, time_type)
    
    def get_raw_cfa(self) -> Optional[tuple[np.ndarray, str]]:
        """Extract raw CFA data and pattern from this page.
        
        Returns:
            Tuple of (cfa_array, cfa_pattern_str) or None if not a CFA page.
            cfa_pattern_str is e.g., 'RGGB', 'BGGR'.
        """
        if not self.is_cfa:
            return None
        
        p = self._page
        
        # Handle interleaved data that needs deswizzling
        col_interleave_tag = p.tags.get(LOCAL_TIFF_TAGS["ColumnInterleaveFactor"])
        row_interleave_tag = p.tags.get(LOCAL_TIFF_TAGS["RowInterleaveFactor"])
        if (
            col_interleave_tag is not None
            and col_interleave_tag.value == 2
            and row_interleave_tag is not None
            and row_interleave_tag.value == 2
        ):
            raw_cfa = deswizzle_cfa_data(p.asarray())
        else:
            raw_cfa = p.asarray()
        
        # Get CFA pattern string
        cfa_str = self.get_tag("CFAPattern", str)
        
        return raw_cfa, cfa_str
    
    def get_raw_linear(self) -> Optional[np.ndarray]:
        """Extract raw LINEAR_RAW data from this page.
        
        For JPEG XL compressed images, handles tile assembly and chroma subsampling.
        
        Returns:
            Raw linear data array or None if not a LINEAR_RAW page.
        """
        if not self.is_linear_raw:
            return None
        
        p = self._page
        
        # Check if this is JPEG XL compression
        if p.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
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

    def _iter_all_pages_recursive(self, pages_list: Optional[List[TiffPage]]):
        """Recursively iterates through all TIFF pages, including nested ones."""
        if pages_list is None:
            return
        for page in pages_list:
            yield page
            if page.pages:  # Check if there are sub-pages
                yield from self._iter_all_pages_recursive(page.pages)

    def _build_dng_pages_recursive(
        self, 
        pages_list: Optional[List[TiffPage]], 
        parent: Optional[DngPage],
        page_id_counter: List[int]
    ) -> List[DngPage]:
        """Build DngPage wrappers with parent references for all pages."""
        result = []
        if pages_list is None:
            return result
        
        for tiff_page in pages_list:
            page_id = page_id_counter[0]
            page_id_counter[0] += 1
            
            dng_page = DngPage(tiff_page, page_id, parent)
            result.append(dng_page)
            
            # Recursively process sub-pages with this page as parent
            if tiff_page.pages:
                result.extend(self._build_dng_pages_recursive(
                    tiff_page.pages, dng_page, page_id_counter
                ))
        
        return result

    def get_flattened_pages(self) -> List[DngPage]:
        """Get all pages as DngPage wrappers with parent references.
        
        Returns:
            List of DngPage objects in flattened order, each with a parent
            reference for tag inheritance.
        """
        return self._build_dng_pages_recursive(self.pages, None, [0])
    
    def get_main_page(self) -> Optional[DngPage]:
        """Get the main image page.
        
        First looks for a page with NewSubFileType == 0 (explicit main image).
        If not found, falls back to the first CFA or LINEAR_RAW page.
        
        Returns:
            The main image DngPage, or None if not found.
        """
        pages = self.get_flattened_pages()
        
        # First try to find explicit main image (NewSubFileType == 0)
        for page in pages:
            if page.is_main_image:
                return page
        
        # Fall back to first CFA or LINEAR_RAW page
        for page in pages:
            if page.is_cfa or page.is_linear_raw:
                return page
        
        return None


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
                elif xmp_name is not None and use_xmp:
                    main_page = dng_file.get_main_page()
                    if main_page is not None and main_page.xmp.has_prop(xmp_name):
                        options[option_name] = main_page.xmp.get_prop(xmp_name, value_type)

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


