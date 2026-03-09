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
    TIFF_TAG_TYPE_REGISTRY,
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


def _ensure_float_tags_be(tags: MetadataTags) -> MetadataTags:
    """Convert float arrays to big-endian for TiffWriter.
    
    TiffWriter passes numpy array bytes unchanged, so we must ensure
    float arrays are in big-endian byte order to match our BE file header.
    """
    for code, dtype, count, value, _ in tags:
        if dtype == 11:  # FLOAT
            if isinstance(value, (np.ndarray, np.floating)) and value.dtype.byteorder != '>':
                tags._tags[code] = tags.StoredTag(code, dtype, count, value.astype('>f4'))
    return tags


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


def _prepare_ifd_args(metadata: MetadataTags, **base_args) -> dict:
    """Prepare TiffWriter IFD args, extracting tags that have kwargs equivalents.
    
    Extracts Software, XResolution, YResolution, ResolutionUnit from metadata,
    converts to TiffWriter kwargs format, removes them from metadata, and
    merges with the provided base args.
    
    Args:
        metadata: MetadataTags instance to use as extratags (modified in-place)
        **base_args: Base IFD args (photometric, compression, etc.)
        
    Returns:
        Dict with complete IFD args including extratags and extracted kwargs
    """
    # Start with base args
    ifd_args = dict(base_args)
    
    # Extract Software
    if 'Software' in metadata:
        ifd_args['software'] = metadata.get_tag('Software')
        metadata.remove_tag('Software')
    
    # Extract Resolution (need both X and Y)
    x_res = metadata.get_tag('XResolution') if 'XResolution' in metadata else None
    y_res = metadata.get_tag('YResolution') if 'YResolution' in metadata else None
    
    if x_res is not None and y_res is not None:
        # Convert rational tuples to float if needed
        if isinstance(x_res, tuple):
            x_res = x_res[0] / x_res[1] if x_res[1] else x_res[0]
        if isinstance(y_res, tuple):
            y_res = y_res[0] / y_res[1] if y_res[1] else y_res[0]
        ifd_args['resolution'] = (x_res, y_res)
        metadata.remove_tag('XResolution')
        metadata.remove_tag('YResolution')
    
    # Extract ResolutionUnit
    if 'ResolutionUnit' in metadata:
        ifd_args['resolutionunit'] = metadata.get_tag('ResolutionUnit')
        metadata.remove_tag('ResolutionUnit')
    
    # Set remaining metadata as extratags (after extracting kwargs equivalents)
    ifd_args['extratags'] = metadata
    
    return ifd_args


def _create_dng_tags(metadata: Optional[MetadataTags], has_jxl: bool) -> MetadataTags:
    """Create DNG metadata tags with defaults and version info.
    
    Tag names are from tifffile.py TiffTagRegistry.
    Tag types are from tifffile.py DATA_DTYPES.
    
    Args:
        metadata: Optional user-supplied metadata to include
        has_jxl: Whether JXL compression is being used (affects backward version)
        
    Returns:
        MetadataTags object with complete DNG metadata
    """
    dng_tags = MetadataTags()

    # Use metadata if provided, otherwise create empty metadata
    if metadata is not None:
        dng_tags.extend(metadata)
    
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


def write_dng(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    photometric: str = "cfa",
    cfa_pattern: str = "RGGB",
    metadata: Optional[MetadataTags] = None,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    preview_image: Optional[np.ndarray] = None
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        raw_data: Raw image data as numpy array. Shape depends on photometric:
                  - "cfa": 2D array (H, W)
                  - "linear_raw": 3D array (H, W, 3)
        destination_file: Path or io.BytesIO object where to save the DNG file.
        bits_per_pixel: Number of bits per pixel (e.g. 12, 14, 16)
        photometric: Photometric interpretation, either "cfa" or "linear_raw"
        cfa_pattern: CFA pattern string, e.g., 'RGGB'. Only used when
                     photometric="cfa".
        jxl_distance: JPEG XL Butteraugli distance. Lower is higher quality.
                     Default: None (no JXL compression).
        jxl_effort: JPEG XL compression effort (1-9). Higher is more
                    compression/slower. Only used if jxl_distance is also
                    specified. Default: None (codec default).
        preview_image: Optional preview/thumbnail image
        metadata: Optional user-supplied MetadataTags to override defaults
    """
    if photometric not in ("cfa", "linear_raw"):
        raise ValueError(
            f"Unsupported photometric: {photometric}. Must be 'cfa' or 'linear_raw'"
        )

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG ({photometric}) to {destination_file}")
    else:
        logger.debug(f"Writing DNG ({photometric}) to in-memory buffer")

    # Validate input shape based on photometric
    if photometric == "cfa":
        if raw_data.ndim != 2:
            raise ValueError(
                f"Expected 2D raw_data (H, W) for photometric='cfa', "
                f"got shape {raw_data.shape}"
            )
        samples_per_pixel = 1
    else:  # linear_raw
        if raw_data.ndim != 3 or raw_data.shape[-1] != 3:
            raise ValueError(
                f"Expected 3D raw_data (H, W, 3) for photometric='linear_raw', "
                f"got shape {raw_data.shape}"
            )
        samples_per_pixel = 3

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        bits_per_pixel = 16
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        bits_per_pixel = 8
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    # IFD structure:
    # - If preview_image exists: IFD0 = preview (JPEG thumbnail), SubIFD0 = raw image
    # - If no preview_image: IFD0 = raw image (main image)
    ifd0_tags = _create_dng_tags(metadata, has_jxl=jxl_distance is not None)
    _ensure_float_tags_be(ifd0_tags)

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:
            if preview_image is not None:
                _write_thumbnail_ifd(tif, preview_image, ifd0_tags)

            # Prepare raw image IFD tags
            raw_ifd_tags = MetadataTags()

            # Add CFA-specific tags
            if photometric == "cfa":
                raw_ifd_tags.add_tag("CFAPattern", cfa_pattern)
                raw_ifd_tags.add_tag("CFARepeatPatternDim", (2, 2))
                raw_ifd_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))

            # Handle JXL compression
            if jxl_distance is not None:
                if not (0.0 <= jxl_distance <= 15.0):
                    logger.warning(
                        f"JXL distance {jxl_distance} is outside the "
                        f"typical range [0.0, 15.0]."
                    )
                compression_type = "JPEGXL_DNG"
                actual_effort = jxl_effort if jxl_effort is not None else 5
                compressionargs = {
                    "distance": jxl_distance,
                    "effort": actual_effort
                }
                logger.debug(
                    f"Writing DNG with JXL compression, "
                    f"distance: {jxl_distance}, effort: {actual_effort}"
                )

                raw_ifd_tags.add_tag("JXLDistance", jxl_distance)
                raw_ifd_tags.add_tag("JXLEffort", actual_effort)

                # CFA data needs swizzling for JXL compression
                if photometric == "cfa":
                    processed_raw_data = swizzle_cfa_data(processed_raw_data)
                    raw_ifd_tags.add_tag("ColumnInterleaveFactor", 2)
                    raw_ifd_tags.add_tag("RowInterleaveFactor", 2)
            else:
                compression_type = COMPRESSION.NONE
                compressionargs = {}

            _ensure_float_tags_be(raw_ifd_tags)

            # If no preview, raw IFD becomes IFD0 and needs ifd0_tags
            if preview_image is None:
                raw_ifd_tags.extend(ifd0_tags)

            # Prepare raw image IFD arguments (extracts Software/Resolution from metadata)
            raw_ifd_args = _prepare_ifd_args(
                raw_ifd_tags,
                subfiletype=0,
                photometric=photometric,
                compression=compression_type,
                compressionargs=compressionargs,
            )

            # Add photometric-specific IFD args
            if photometric == "cfa":
                raw_ifd_args["subifds"] = 0
            else:  # linear_raw
                raw_ifd_args["planarconfig"] = 1  # CONTIG

            # Calculate rowsperstrip
            raw_datasize = int(
                processed_raw_data.shape[0]
                * processed_raw_data.shape[1]
                * samples_per_pixel
                * bits_per_pixel
                / 8
            )
            tif.write(processed_raw_data, **raw_ifd_args, rowsperstrip=raw_datasize)

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise

# =============================================================================
# DNG Tag Copy Allowlist
# =============================================================================
# Used when copying tags from source DNG files. User-provided tags bypass this.

# Tags TiffWriter manages automatically - never copy as extratags
_TIFFWRITER_MANAGED_TAGS = {
    'NewSubfileType', 'SubfileType', 'ImageWidth', 'ImageLength',
    'BitsPerSample', 'Compression', 'PhotometricInterpretation',
    'ImageDescription', 'StripOffsets', 'SamplesPerPixel',
    'RowsPerStrip', 'StripByteCounts', 'XResolution', 'YResolution',
    'PlanarConfiguration', 'ResolutionUnit', 'Software',
    'TileWidth', 'TileLength', 'TileOffsets', 'TileByteCounts',
    'SubIFDs', 'ExifTag', 'GPSTag', 'InteroperabilityTag',
}

# JPEG thumbnail tags - not valid for raw data IFDs
_JPEG_THUMBNAIL_TAGS = {
    'YCbCrCoefficients', 'YCbCrSubSampling', 'YCbCrPositioning',
    'ReferenceBlackWhite', 'JPEGInterchangeFormat',
    'JPEGInterchangeFormatLength', 'JPEGTables', 'JPEGRestartInterval',
    'JPEGLosslessPredictors', 'JPEGPointTransforms',
    'JPEGQTables', 'JPEGDCTables', 'JPEGACTables',
}

# Tags invalidated when decompressing (digests become stale)
_COMPRESSION_INVALIDATED_TAGS = {
    'NewRawImageDigest', 'RawImageDigest', 'OriginalRawFileDigest',
    'CacheVersion', 'JXLDistance', 'JXLEffort', 'JXLDecodeSpeed',
}

# Tags only valid for preview IFDs, not standalone DNGs
_PREVIEW_ONLY_TAGS = {
    'PreviewApplicationName', 'PreviewApplicationVersion',
    'PreviewSettingsName', 'PreviewSettingsDigest',
    'PreviewColorSpace', 'PreviewDateTime', 'RawToPreviewGain',
}

# Base allowlist = registry keys minus structural/thumbnail/preview tags
_DNG_METADATA_ALLOWLIST = (
    set(TIFF_TAG_TYPE_REGISTRY.keys())
    - _TIFFWRITER_MANAGED_TAGS
    - _JPEG_THUMBNAIL_TAGS
    - _PREVIEW_ONLY_TAGS
)

def _get_dng_copy_tags(exclude_compression: bool = False) -> set:
    """Return set of tag names allowed when copying from source files.
    
    This filters tags read from source DNG files. User-provided tags
    (e.g., metadata) should bypass this filter entirely.
    
    Args:
        exclude_compression: True when decompressing (invalidates digests)
        
    Returns:
        Set of allowed tag names
    """
    tags = _DNG_METADATA_ALLOWLIST.copy()
    if exclude_compression:
        tags -= _COMPRESSION_INVALIDATED_TAGS
    return tags


def write_dng_from_page(
    page: "DngPage",
    destination_file: Union[Path, io.BytesIO],
    metadata: Optional[MetadataTags] = None,
    preview_image: Optional[np.ndarray] = None,
    skip_tags: Optional[set[str]] = None,
    decompress: bool = False,
) -> None:
    """Write DNG file from an existing DngPage.
    
    By default, copies compressed raw data directly without decompression,
    preserving the original compression. Set decompress=True to decode and
    re-encode the data (slower but works with any tile configuration).
    
    Args:
        page: DngPage containing the raw data to copy
        destination_file: Path or io.BytesIO object where to save the DNG file
        metadata: Optional user-supplied MetadataTags to override source tags
        preview_image: Optional preview/thumbnail image
        skip_tags: Optional set of tag names to skip when copying (e.g., for debugging)
        decompress: If True, decode raw data and write uncompressed. Use when
            direct copy fails due to unsupported tile configurations.
    
    Raises:
        ValueError: If decompress=True and page is not CFA or LINEAR_RAW
        ValueError: If decompress=False and tile configuration is unsupported
    """
    # IFD structure (same as write_dng):
    # - If preview_image exists: IFD0 = preview (JPEG thumbnail), SubIFD0 = raw image
    # - If no preview_image: IFD0 = raw image (main image)
    
    # Get allowlist of tags to copy from source files (bounded to registry)
    # Exclude compression tags when decompressing; preview tags always excluded
    allowed_tags = _get_dng_copy_tags(exclude_compression=decompress)
    if skip_tags:
        allowed_tags -= skip_tags
    
    # Build ifd0_tags from source IFD0 (color calibration, camera info, etc.)
    ifd0_tags = MetadataTags()
    if page.ifd0 is not None:
        for tag in page.ifd0.tags.values():
            if tag.name in allowed_tags:
                ifd0_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)
    
    # Apply user-supplied metadata overrides to ifd0_tags
    ifd0_tags.extend(metadata)
    _ensure_float_tags_be(ifd0_tags)
    
    # Build raw_ifd_tags from page-specific tags (CFAPattern, compression, etc.)
    raw_ifd_tags = MetadataTags()
    for tag in page.tags.values():
        if tag.name in allowed_tags:
            raw_ifd_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)
    _ensure_float_tags_be(raw_ifd_tags)

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG from DngPage to {destination_file}")
    else:
        logger.debug("Writing DNG from DngPage to in-memory buffer")

    if decompress:
        # Decode and re-encode path - works with any tile configuration
        bits_per_pixel = page.bitspersample
        
        if page.is_cfa:
            cfa_result = page.get_raw_cfa()
            if cfa_result is None:
                raise ValueError("Failed to extract CFA data from page")
            raw_data, cfa_pattern = cfa_result
            photometric = "cfa"
        elif page.is_linear_raw:
            raw_data = page.get_raw_linear()
            if raw_data is None:
                raise ValueError("Failed to extract LINEAR_RAW data from page")
            cfa_pattern = None
            photometric = "linear_raw"
        else:
            raise ValueError(
                f"Page must be CFA or LINEAR_RAW, got photometric={page.photometric_name}"
            )
        
        # For decompress path, merge tags for write_dng (it handles IFD structure internally)
        merged_tags = ifd0_tags.copy()
        merged_tags.extend(raw_ifd_tags)
        
        write_dng(
            raw_data=raw_data,
            destination_file=destination_file,
            bits_per_pixel=bits_per_pixel,
            cfa_pattern=cfa_pattern,
            photometric=photometric,
            metadata=merged_tags,
            preview_image=preview_image,
        )
    else:
        # Direct compressed copy path - preserves original compression
        try:
            with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:

                if preview_image is not None:
                    _write_thumbnail_ifd(tif, preview_image, ifd0_tags)
                else:
                    # No preview: raw IFD becomes IFD0 and needs ifd0_tags
                    raw_ifd_tags.extend(ifd0_tags)

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
                        return

                logger.debug(f"Read {len(compressed_segments)} compressed segments from page")

                # Prepare raw image IFD arguments (extracts Software/Resolution)
                raw_ifd_args = _prepare_ifd_args(
                    raw_ifd_tags,
                    subfiletype=0,
                    photometric=page.photometric,
                    subifds=0,
                    compression=page.compression,
                )

                # Determine shape based on samples per pixel
                samples_per_pixel = page.samplesperpixel if hasattr(page, 'samplesperpixel') else 1
                if samples_per_pixel > 1:
                    write_shape = (page.imagelength, page.imagewidth, samples_per_pixel)
                else:
                    write_shape = (page.imagelength, page.imagewidth)
                
                # Handle tiled vs stripped images
                if page.is_tiled:
                    tile_shape = (page.tilelength, page.tilewidth)
                    tile_valid = (
                        tile_shape[0] <= page.imagelength and 
                        tile_shape[1] <= page.imagewidth and
                        tile_shape[0] % 16 == 0 and 
                        tile_shape[1] % 16 == 0
                    )
                    if not tile_valid:
                        raise ValueError(
                            f"Cannot copy tiled page: tile {tile_shape} not supported by tifffile "
                            f"(must be <= image size and multiple of 16). Use decompress=True."
                        )
                    tif.write(
                        data=compressed_data_iterator(),
                        shape=write_shape,
                        dtype=page.dtype,
                        bitspersample=page.bitspersample,
                        tile=tile_shape,
                        **raw_ifd_args,
                    )
                else:
                    raw_datasize = page.imagelength * page.imagewidth * samples_per_pixel * (page.bitspersample // 8)
                    tif.write(
                        data=compressed_data_iterator(),
                        shape=write_shape,
                        dtype=page.dtype,
                        bitspersample=page.bitspersample,
                        rowsperstrip=raw_datasize,
                        **raw_ifd_args,
                    )
                
                logger.debug(f"Successfully copied compressed raw data ({sum(page.databytecounts)} bytes)")
            
            if isinstance(destination_file, Path):
                logger.debug(f"Successfully wrote DNG file to {destination_file}")
            else:
                logger.debug("Successfully wrote DNG file to in-memory buffer")
            
        except Exception as e:
            logger.error(f"Error writing DNG from DngPage: {e}")
            raise


class DngPage(TiffPage):
    """TiffPage subclass with DNG-specific functionality.
    
    Provides convenient access to DNG tags with automatic translation,
    parent IFD tag inheritance, and raw data extraction methods.
    
    Inherits all TiffPage attributes and methods. Created by "upgrading"
    an existing TiffPage instance.
    """
    
    # NewSubFileType values from DNG spec
    SF_MAIN_IMAGE = 0
    SF_PREVIEW_IMAGE = 1
    SF_TRANSPARENCY_MASK = 2
    SF_PREVIEW_MASK = 8
    SF_ALT_PREVIEW_IMAGE = 65537
    
    def __new__(cls, *args, **kwargs):
        """Create DngPage instance without calling TiffPage.__init__."""
        return object.__new__(cls)
    
    def __init__(self, tiff_page: TiffPage):
        """Initialize DngPage by copying TiffPage state.
        
        Args:
            tiff_page: The TiffPage to upgrade
        """
        self.__dict__.update(tiff_page.__dict__)
    
    @property
    def ifd0(self) -> Optional[TiffPage]:
        """Return IFD0, or None if this page IS IFD0."""
        page0 = self.parent.pages[0]
        return None if page0 is self else page0
    
    @property
    def photometric_name(self) -> Optional[str]:
        """Photometric interpretation string (e.g., 'CFA', 'LINEAR_RAW', 'RGB')."""
        if self.photometric is not None:
            return self.photometric.name
        return None
    
    @property
    def is_cfa(self) -> bool:
        """True if this page contains CFA (Bayer) raw data."""
        return self.photometric_name == "CFA"
    
    @property
    def is_linear_raw(self) -> bool:
        """True if this page contains LINEAR_RAW (demosaiced) data."""
        return self.photometric_name == "LINEAR_RAW"
    
    @property
    def is_main_image(self) -> bool:
        """True if this page is the main image (NewSubfileType == 0)."""
        tag = self.tags.get(254)  # NewSubfileType tag ID
        if tag is None:
            return False
        return tag.value == self.SF_MAIN_IMAGE
    
    @property
    def is_preview(self) -> bool:
        """True if this page is a preview image."""
        tag = self.tags.get(254)  # NewSubfileType tag ID
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
        
        # Get value from page, or fall back to IFD0 for global tags
        raw_tag = None
        if tag_id in self.tags:
            raw_tag = self.tags[tag_id]
        elif self.ifd0 is not None and tag_id in self.ifd0.tags:
            raw_tag = self.ifd0.tags[tag_id]
        
        if raw_tag is None:
            return None
        
        effective_type = return_type or get_native_type(raw_tag.dtype, raw_tag.count)
        shape_spec = None
        if registry_spec and registry_spec.shape and registry_spec.count == raw_tag.count:
            shape_spec = TagSpec(TIFF_DTYPE_TO_STR.get(raw_tag.dtype, 'B'), raw_tag.count, registry_spec.shape)
        
        return decode_tag_value(tag_name, raw_tag.value, raw_tag.dtype, shape_spec, effective_type)

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
        
        if tag_id in self.tags:
            return self.tags[tag_id].value
        
        # Fall back to IFD0 for global tags
        if self.ifd0 is not None and tag_id in self.ifd0.tags:
            return self.ifd0.tags[tag_id].value
        
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
    
    def _decode_jpegxl(self) -> np.ndarray:
        """Decode JPEG XL compressed image data, handling tiled images.
        
        Returns:
            Decoded image array.
        """
        fh = self.parent.filehandle
        
        # Read all segments using tifffile's read_segments API
        segments = list(fh.read_segments(
            self.dataoffsets,
            self.databytecounts,
            sort=True
        ))
        
        if len(segments) == 1:
            # Single tile/strip
            compressed_data, _ = segments[0]
            return imagecodecs.jpegxl_decode(compressed_data)
        else:
            # Multiple tiles - decode each and assemble
            tile_width = self.tilewidth
            tile_height = self.tilelength
            img_width = self.imagewidth
            img_height = self.imagelength
            samples = self.samplesperpixel or 1
            
            # Determine output dtype from first tile
            first_tile = imagecodecs.jpegxl_decode(segments[0][0])
            dtype = first_tile.dtype
            
            # Create output array - handle both 2D (single sample) and 3D (multi-sample)
            if samples == 1:
                output = np.zeros((img_height, img_width), dtype=dtype)
            else:
                output = np.zeros((img_height, img_width, samples), dtype=dtype)
            
            # Decode and place each tile
            tiles_x = (img_width + tile_width - 1) // tile_width
            for i, (tile_data, _) in enumerate(segments):
                tile = imagecodecs.jpegxl_decode(tile_data)
                
                ty = (i // tiles_x) * tile_height
                tx = (i % tiles_x) * tile_width
                
                # Handle edge tiles that may be smaller
                th = min(tile_height, img_height - ty)
                tw = min(tile_width, img_width - tx)
                output[ty:ty+th, tx:tx+tw] = tile[:th, :tw]
            
            return output

    def get_raw_cfa(self) -> Optional[tuple[np.ndarray, str]]:
        """Extract raw CFA data and pattern from this page.
        
        Returns:
            Tuple of (cfa_array, cfa_pattern_str) or None if not a CFA page.
            cfa_pattern_str is e.g., 'RGGB', 'BGGR'.
        """
        if not self.is_cfa:
            return None
        
        # Decode image data
        if self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
            raw_cfa = self._decode_jpegxl()
        else:
            raw_cfa = self.asarray()
        
        # Handle interleaved data that needs deswizzling
        col_interleave_tag = self.tags.get(LOCAL_TIFF_TAGS["ColumnInterleaveFactor"])
        row_interleave_tag = self.tags.get(LOCAL_TIFF_TAGS["RowInterleaveFactor"])
        if (
            col_interleave_tag is not None
            and col_interleave_tag.value == 2
            and row_interleave_tag is not None
            and row_interleave_tag.value == 2
        ):
            raw_cfa = deswizzle_cfa_data(raw_cfa)
        
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
        
        if self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
            return self._decode_jpegxl()
        else:
            return self.asarray()


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

    def _build_dng_pages_recursive(self, pages_list: Optional[List[TiffPage]]) -> List[DngPage]:
        """Build DngPage instances from TiffPages."""
        result = []
        if pages_list is None:
            return result
        
        for tiff_page in pages_list:
            dng_page = DngPage(tiff_page)
            result.append(dng_page)
            
            # Recursively process sub-pages
            if tiff_page.pages:
                result.extend(self._build_dng_pages_recursive(tiff_page.pages))
        
        return result

    def get_flattened_pages(self) -> List[DngPage]:
        """Get all pages as DngPage instances.
        
        Returns:
            List of DngPage objects in flattened order. Tag inheritance
            falls back to IFD0 via TiffPage.parent.
        """
        return self._build_dng_pages_recursive(self.pages)
    
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
        preview_image = color_mac.process_raw_core_image(
            dng_input=dng_input,
            raw_filter_options=options,
            use_gpu=True,
            output_dtype=output_dtype,
        )
        
        if preview_image is None:
            raise RuntimeError(f"Failed to process DNG file: {file}")
        
        logger.debug(f"Successfully decoded DNG to array with shape {preview_image.shape} and dtype {preview_image.dtype}")
        return preview_image
                
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
        preview_image = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(preview_image, cv2.COLOR_RGB2BGR)
        
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
        preview_image = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(preview_image, cv2.COLOR_RGB2BGR)
        
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


