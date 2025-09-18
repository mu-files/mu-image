"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import io
import logging
import numpy as np

from pathlib import Path
from tifffile import COMPRESSION, PHOTOMETRIC, TiffFile, TiffPage, TiffWriter, TIFF
from typing import Optional, Union, List, Dict, Tuple, Any, Type, IO

logger = logging.getLogger(__name__)

BAYER_PATTERN_MAP = {
    "RGGB": (0, 1, 1, 2),  # R G / G B
    "BGGR": (2, 1, 1, 0),  # B G / G R
    "GRBG": (1, 0, 2, 1),  # G R / B G
    "GBRG": (1, 2, 0, 1),  # G B / R G
}

# Inverse mapping from 2x2 CFA codes to string key
INVERSE_BAYER_PATTERN_MAP = {v: k for k, v in BAYER_PATTERN_MAP.items()}

# helper class to convert create a list of tags for tifffile.TiffWriter
class MetadataTags:

    '''
    TIFF.DATA_DTYPES (2nd argument to add_tag) used in tifffile.py below have following mapping:
        'B': unsigned byte
        's': ascii string
        'H': unsigned short
        'I': unsigned long
        '2I': unsigned rational
        'b': signed byte
        'h': signed short
        'i': signed long
        '2i': signed rational
        'f': float
        'd': double
    '''

    @staticmethod
    def _matrix_to_rational_tuple(matrix: np.ndarray, denominator: int = 10000) -> tuple:
        """Converts a NumPy float matrix to a flat tuple of (numerator, denominator) pairs."""
        # Flatten the matrix and use the common rational conversion helper
        flat_array = matrix.flatten()
        rational_list = MetadataTags.float_array_to_rationals(flat_array, max_denominator=denominator)
        return tuple(rational_list)

    @staticmethod
    def float_array_to_rationals(float_array, max_denominator: int = 10000):
        """Convert a list/array of floats to TIFF rational format using Fraction for precision."""
        from fractions import Fraction
        
        rationals = []
        for val in float_array:
            frac = Fraction(val).limit_denominator(max_denominator)
            rationals.extend([frac.numerator, frac.denominator])
        return rationals

    def __init__(self):
        self._tags = []

    def add_tag(self, tag):
        tag_code = -1

        if isinstance(tag[0], str):
            tag_code = TIFF.TAGS[tag[0]]
        elif isinstance(tag[0], int):
            tag_code = tag[0]

        # Handle dtype parameter - can be string key or DATATYPE enum value
        if isinstance(tag[1], str):
            tag_dtype = TIFF.DATA_DTYPES[tag[1]]
        else:
            # Assume it's already a DATATYPE enum value
            tag_dtype = tag[1]

        tag_formatted_contents = (tag_code, tag_dtype, tag[2], tag[3], False)
        
        # Check for duplicates and overwrite if one is found, else append
        for i, existing_tag in enumerate(self._tags):
            if existing_tag[0] == tag_code:
                self._tags[i] = tag_formatted_contents
                return

        self._tags.append(tag_formatted_contents)

    def add_string_tag(self, tag_name_str, string_value):
        """Helper to add a standard ASCII string tag with null termination."""
        string_value_with_null = string_value + "\x00"
        length = len(string_value_with_null)
        self.add_tag((tag_name_str, "s", length, string_value_with_null))

    def extend(self, other: "MetadataTags") -> None:
        """Add all tags from another MetadataTags instance."""
        if not isinstance(other, MetadataTags):
            raise TypeError(f"Expected MetadataTags instance, got {type(other).__name__}")
        # Use add_tag to ensure proper duplicate handling instead of direct list extension
        for tag_tuple in other._tags:
            # tag_tuple format: (tag_code, tag_dtype, count, value, writeonce)
            # Convert to add_tag format: (tag_name_or_code, dtype, count, value)
            self.add_tag((tag_tuple[0], tag_tuple[1], tag_tuple[2], tag_tuple[3]))

    def add_cfa_pattern_tag(self, cfa_pattern_key: str):
        """Helper to add the CFAPattern tag using the class's Bayer pattern map."""
        pattern_tuple = BAYER_PATTERN_MAP.get(cfa_pattern_key, BAYER_PATTERN_MAP["RGGB"])
        pattern_bytes = bytes(pattern_tuple)
        self.add_tag(("CFAPattern", "B", 4, pattern_bytes))

    def add_matrix_as_rational_tag(
        self,
        tag_name_str: str,
        float_matrix_np: np.ndarray,
        denominator: int = 10000,
    ):
        """Converts a float matrix to rationals and adds it as a tag."""
        flat_tuple_values = MetadataTags._matrix_to_rational_tuple(float_matrix_np, denominator)
        self.add_tag((tag_name_str, "2i", 9, flat_tuple_values))

    def add_float_array_as_rational_tag(
        self,
        tag_name_str: str,
        float_array,
        max_denominator: int = 10000,
    ):
        """Converts a float array to rationals and adds it as a tag."""
        rational_list = MetadataTags.float_array_to_rationals(float_array, max_denominator)
        count = len(float_array)
        self.add_tag((tag_name_str, "2I", count, tuple(rational_list)))

    def get_tags(self):
        self._tags.sort(key=lambda x: x[0])
        return self._tags

    def add_xmp(self, xmp_data: Dict[str, Union[str, int, float]]) -> None:
        """
        Add XMP metadata to this MetadataTags instance.
        
        Args:
            xmp_data: Dictionary of XMP properties to add. Keys can include namespace 
                     prefixes (e.g., 'crs:Temperature', 'tiff:Orientation') or will 
                     default to 'crs:' namespace if no prefix is provided.
                     
        Example:
            camera_metadata.add_xmp({
                'Temperature': 5500,
                'Tint': 0,
                'Exposure2012': -0.5,
                'tiff:Orientation': 1
            })
        """
        from datetime import datetime
        
        # Generate timestamp in ISO format with timezone
        now = datetime.now()
        iso_date = now.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Try to format timezone as -07:00 instead of -0700
        try:
            iso_date_tz = now.astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
            # Insert colon in timezone offset: -0700 -> -07:00
            if len(iso_date_tz) >= 5 and iso_date_tz[-5] in '+-':
                iso_date_tz = iso_date_tz[:-2] + ':' + iso_date_tz[-2:]
        except:
            iso_date_tz = None
        
        # Build XMP properties with namespace handling
        xmp_props = []
        
        # Add standard metadata - minimal required set
        standard_props = {
            'tiff:Orientation': '1',
            'dc:format': 'image/dng',
            'xmp:CreatorTool': 'muimg',
            'xmp:ModifyDate': iso_date,
            'crs:Version': '17.4',
            'crs:ProcessVersion': '15.4',
        }
        
        # Add user-provided data
        sequence_props = {}
        for key, value in xmp_data.items():
            # Auto-prepend 'crs:' if no namespace specified
            if ':' not in key:
                key = f'crs:{key}'
            
            # Check if value has coordinate pairs (list of 2-tuples) - needs <rdf:Seq> structure
            if hasattr(value, 'points') and isinstance(getattr(value, 'points'), list):
                # ToneCurve object with points attribute
                sequence_props[key] = getattr(value, 'points')
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (tuple, list)) and len(value[0]) == 2:
                # Direct list of 2-tuples
                sequence_props[key] = value
            else:
                standard_props[key] = str(value)
        
        # Format as XMP attributes
        for prop, value in standard_props.items():
            xmp_props.append(f'    {prop}="{value}"')
        
        # Build sequence structures for coordinate pairs
        sequence_xml = ""
        if sequence_props:
            sequence_elements = []
            for prop_name, points in sequence_props.items():
                # Extract namespace and property name for XML element
                if ':' in prop_name:
                    namespace, prop = prop_name.split(':', 1)
                    element_name = f'{namespace}:{prop}'
                else:
                    element_name = prop_name
                
                # Build rdf:li items from coordinate pairs
                sequence_items = []
                for x, y in points:
                    sequence_items.append(f'      <rdf:li>{x}, {y}</rdf:li>')
                
                sequence_elements.append(f'''    <{element_name}>
     <rdf:Seq>
{chr(10).join(sequence_items)}
     </rdf:Seq>
    </{element_name}>''')
            
            sequence_xml = chr(10).join(sequence_elements)

        # Create minimal XMP structure based on Lightroom format
        xmp_content = f'''<?xpacket begin="\\357\\273\\277" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="muimg XMP Core">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{chr(10).join(xmp_props)}>{sequence_xml}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
        
        xmp_bytes = xmp_content.encode('utf-8')
        self.add_tag(("XMP", "B", len(xmp_bytes), xmp_bytes))
        logger.debug(f"Added XMP metadata with {len(xmp_data)} user properties")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts averaged R, G, and B planes from CFA data using the given pattern.
    
    Args:
        raw_cfa: Raw CFA data array
        cfa_pattern_str: CFA pattern string (e.g., "RGGB", "BGGR")
        
    Returns:
        Tuple of (r_plane, g_plane, b_plane)
        
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

    g_plane = ((
        g1_plane.astype(np.uint32) + g2_plane.astype(np.uint32)) // 2).astype(g1_plane.dtype)

    return r_plane, g_plane, b_plane


def rgb_planes_from_dng(
    dng_file: "DngFile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a DNG file, finds the primary CFA raw image, and extracts averaged
    R, G, and B planes.

    Raises:
        ValueError: If the DNG file format is invalid or missing required data.
        RuntimeError: If DNG processing fails.
    """
    # Extract CFA data and pattern, then process to RGB planes
    raw_cfa, cfa_pattern_value = cfa_from_dng(dng_file)
    return rgb_from_cfa(raw_cfa, cfa_pattern_value)


def _write_thumbnail_ifd(writer: "TiffWriter", thumbnail_image: np.ndarray, dng_tags: "MetadataTags") -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags
    dng_tags.add_tag(("PreviewColorSpace", "I", 1, PREVIEWCOLORSPACE_SRGB))

    # Write Thumbnail to SubIFD 0
    thumb_ifd_args = {
        "photometric": "rgb",  # Interprets data as RGB
        "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
        "compression": "jpeg",  # JPEG compression for thumbnail
        "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
        "extratags": dng_tags.get_tags(),
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


def _create_dng_tags(camera_profile: Optional["MetadataTags"], has_jxl: bool) -> "MetadataTags":
    """Create DNG metadata tags with defaults and version info.
    
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
    
    # Check for required tags and add defaults if missing
    existing_tags = {tag[0] for tag in dng_tags.get_tags()}
    
    # Add Orientation if not set (default to horizontal)
    if TIFF.TAGS["Orientation"] not in existing_tags:
        dng_tags.add_tag(("Orientation", "H", 1, ORIENTATION_HORIZONTAL))
    
    # Add ColorMatrix1 if not set (default to 3x3 identity)
    if TIFF.TAGS["ColorMatrix1"] not in existing_tags:
        identity_matrix = np.identity(3, dtype=np.float64)
        dng_tags.add_matrix_as_rational_tag("ColorMatrix1", identity_matrix)
    
    # Add CalibrationIlluminant1 if not set (default to unknown)
    if TIFF.TAGS["CalibrationIlluminant1"] not in existing_tags:
        dng_tags.add_tag(("CalibrationIlluminant1", "H", 1, 0))  # 0 = Unknown

    dng_tags.add_tag(("DNGVersion", "B", 4, (1, 7, 1, 0)))
    if not has_jxl:
        # need latest version for CFA compression but lots of old software can't handle it
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 4, 0, 0)))
    else:
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 7, 1, 0)))
        
    return dng_tags


def _write_thumbnail_ifd(writer: "TiffWriter", thumbnail_image: np.ndarray, dng_tags: "MetadataTags") -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags
    dng_tags.add_tag(("PreviewColorSpace", "I", 1, PREVIEWCOLORSPACE_SRGB))

    # Write Thumbnail to SubIFD 0
    thumb_ifd_args = {
        "photometric": "rgb",  # Interprets data as RGB
        "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
        "compression": "jpeg",  # JPEG compression for thumbnail
        "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
        "extratags": dng_tags.get_tags(),
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
    camera_profile: Optional["MetadataTags"] = None,
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

    # the tag names are from tifffile.py TiffTagRegistry
    # the tag types are from tifffile.py DATA_DTYPES
    dng_tags = _create_dng_tags(camera_profile, has_jxl=jxl_distance is not None)

    try:
        logger.debug(f"Writing DNG from raw data buffer to {destination_file}")

        with TiffWriter(destination_file, bigtiff=False) as tif:
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
                "extratags": dng_cfa_tags.get_tags()
            }

            # Write Main Raw Image to IFD
            raw_datasize = int(
                processed_raw_data.shape[0] * processed_raw_data.shape[1] * bits_per_pixel / 8
            )
            tif.write(processed_raw_data, **main_image_ifd_args, rowsperstrip=raw_datasize)

        logger.debug(f"Successfully wrote DNG file to {destination_file}")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise

def write_dng_linearraw(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    camera_profile: Optional["MetadataTags"] = None,
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
        with TiffWriter(destination_file, bigtiff=False) as tif:
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
                "extratags": linearraw_tags.get_tags(),
            }

            # Calculate rowsperstrip to avoid many strips (mirror write_dng)
            samples_per_pixel = 3
            raw_datasize = int(
                processed_raw_data.shape[0] * processed_raw_data.shape[1] * samples_per_pixel * (bits_per_pixel / 8)
            )

            tif.write(processed_raw_data, **main_ifd_args, rowsperstrip=raw_datasize)

        logger.debug(f"Successfully wrote LinearRaw DNG to {destination_file}")

    except Exception as e:
        logger.error(f"Error saving LinearRaw DNG file with TiffWriter: {e}")
        raise

def write_dng_from_page(
    page: "TiffPage",
    destination_file: Union[Path, io.BytesIO],
    camera_profile: Optional["MetadataTags"] = None,
    color_data: Optional[np.ndarray] = None
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
    """
    # the tag names are from tifffile.py TiffTagRegistry
    # the tag types are from tifffile.py DATA_DTYPES
    has_jxl = page.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG)
    dng_tags = _create_dng_tags(camera_profile, has_jxl)

    try:
        logger.debug(f"Writing DNG from TiffPage to {destination_file}")
        
        with TiffWriter(destination_file, bigtiff=False) as tif:

            if color_data is not None:
                _write_thumbnail_ifd(tif, color_data, dng_tags)

            dng_cfa_tags = MetadataTags()
            
            # Tags that TiffWriter handles automatically - don't copy as extratags
            dont_copy_tags = {
                'NewSubfileType',      # 254 - handled by subfiletype parameter
                'ImageWidth',          # 256 - handled by shape parameter
                'ImageLength',         # 257 - handled by shape parameter  
                'BitsPerSample',       # 258 - handled by dtype parameter
                'Compression',         # 259 - handled by compression parameter
                'PhotometricInterpretation', # 262 - handled by photometric parameter
                'ImageDescription',    # 270 - handled by description parameter
                'StripOffsets',        # 273 - handled automatically by TiffWriter
                'SamplesPerPixel',     # 277 - handled by dtype/shape parameters
                'RowsPerStrip',        # 278 - handled by rowsperstrip parameter
                'StripByteCounts',     # 279 - handled automatically by TiffWriter
                'XResolution',         # 282 - handled by resolution parameter
                'YResolution',         # 283 - handled by resolution parameter
                'ResolutionUnit',      # 296 - handled by resolution parameter
                'Software'             # 305 - handled by software parameter
            }
            
            # Copy page-specific tags (like the test pattern)
            if hasattr(page, 'tags'):
                for tag in page.tags.values():
                    # Skip tags that TiffWriter handles automatically
                    if tag.name not in dont_copy_tags:
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
                "extratags": dng_cfa_tags.get_tags()
            }

            # Calculate raw data size for rowsperstrip
            raw_datasize = page.imagelength * page.imagewidth * (page.bitspersample // 8)
            
            # Write the compressed raw data
            tif.write(
                data=compressed_data_iterator(),
                shape=(page.imagelength, page.imagewidth),
                dtype=page.dtype,
                **main_image_ifd_args,
                rowsperstrip=raw_datasize
            )
            
            logger.debug(f"Successfully copied compressed raw data ({sum(page.databytecounts)} bytes)")
        
        logger.debug(f"Successfully wrote DNG file to {destination_file}")
        
    except Exception as e:
        logger.error(f"Error writing DNG from TiffPage: {e}")
        raise


class DngFile(TiffFile):

    """A TIFF file with DNG-specific extensions and helper methods."""

    @staticmethod
    def _rational_tuple_to_matrix(tag_value: tuple) -> np.ndarray:
        """
        Converts a flat tuple of 18 signed integers, representing 9 rational
        numbers (n1, d1, n2, d2, ..., n9, d9), into a 3x3 float NumPy array.
        """
        if not isinstance(tag_value, tuple) or len(tag_value) != 18:
            raise ValueError(
                f"Input must be a flat tuple of 18 integers for a 3x3 matrix. "
                f"Expected 18 elements, got {len(tag_value) if isinstance(tag_value, tuple) else type(tag_value)}."
            )

        float_values = []
        for i in range(0, 18, 2): # Iterate 9 times, taking 2 elements each time
            num = tag_value[i]
            den = tag_value[i+1]
            float_values.append(float(num) / float(den))

        return np.array(float_values).reshape((3, 3))

    @staticmethod
    def _translate_dng_tag_value(tag_name: str, tag_value) -> Any:
    
        _bayer_pattern_bytes_to_str_map = {
            bytes(v): k for k, v in BAYER_PATTERN_MAP.items()
        }

        MATRIX_TAG_NAMES = {"ColorMatrix1", "ColorMatrix2", "ColorMatrix3"}

        # TODO: complete list of photometric interpretation values
        if tag_name == "CFAPattern":
            cfa_bytes = tag_value
            if isinstance(cfa_bytes, bytes):
                tag_value = _bayer_pattern_bytes_to_str_map.get(cfa_bytes, tag_value)
        elif tag_name == "PhotometricInterpretation":
            if tag_value == PHOTOMETRIC.CFA:
                tag_value = "CFA"
            elif tag_value == PHOTOMETRIC.LINEAR_RAW:
                tag_value = "LINEAR_RAW"
        elif tag_name in MATRIX_TAG_NAMES:
            try:
                tag_value = DngFile._rational_tuple_to_matrix(tag_value)
            except (ValueError, ZeroDivisionError, TypeError, AttributeError) as e:
                raise ValueError(
                    f"Error converting DNG tag '{tag_name}' to matrix. Original error: {e}. "
                    f"Value type: {type(tag_value)}, Value: {str(tag_value)[:100]}"
                ) from e

        return tag_value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xmp_metadata = None  # Lazy-loaded XMP metadata dictionary

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
            "ColorMatrix1",
            "ColorMatrix2",
            "ColorMatrix3",
            "CalibrationIlluminant1",
            "CalibrationIlluminant2",
            "CalibrationIlluminant3",
            "AnalogBalance",
            "AsShotNeutral",
            "BaselineExposure",
        ]

        cfa_subifd_tags = [
            "PhotometricInterpretation",
            "Compression",
            "SamplesPerPixel",
            "PlanarConfiguration",
            "CFARepeatPatternDim",
            "CFAPattern",
            "CFAPlaneColor",
            "BlackLevel",
            "WhiteLevel",
            "ColumnInterleaveFactor",
            "RowInterleaveFactor",
            "JXLDistance",
            "JXLEffort",
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
                    global_tags_data[tag_name] = self._translate_dng_tag_value(
                        tag_name, page_tag.value)

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
                            current_page_tags[tag_name] = self._translate_dng_tag_value(
                                tag_name, page_tag.value
                            )

                info_list.append((current_page_id, page.shape, current_page_tags))

        return info_list

    def get_page_by_id(self, target_page_id: int) -> Optional[TiffPage]:
        """Retrieve a specific TiffPage by its flattened, 0-based ID."""
        for i, page in enumerate(self._iter_all_pages_recursive(self.pages)):
            if i == target_page_id:
                return page
        return None

    def get_raw_cfa_by_id(self, target_page_id: int) -> Optional[tuple[np.ndarray, str]]:
        """Retrieves the raw CFA array and CFAPattern string for a specific 'CFA' page.
        
        Returns a tuple (raw_cfa_array, cfa_pattern_value) or None if page not found/invalid.
        """
        p = self.get_page_by_id(target_page_id)
        return self.get_raw_cfa_from_page(p)

    def get_raw_cfa_from_page(self, page: Optional[TiffPage]) -> Optional[tuple[np.ndarray, str]]:
        """Retrieves the raw CFA array and CFAPattern string from a given TiffPage.
        
        Returns a tuple (raw_cfa_array, cfa_pattern_value) or None if page not valid/unsupported.
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

        # Fetch CFAPattern and map its 4-code tuple to a string using the inverse map.
        cfa_tag = p.tags.get(TIFF.TAGS.get("CFAPattern"))
        cfa_str = None
        if cfa_tag is not None:
            v = cfa_tag.value
            if isinstance(v, (bytes, bytearray)) and len(v) == 4:
                codes = tuple(int(b) for b in v)
                cfa_str = INVERSE_BAYER_PATTERN_MAP.get(codes)

        return raw_cfa, cfa_str

    def get_raw_linear_by_id(self, target_page_id: int) -> Optional[np.ndarray]:
        """Retrieves the raw data array for a specific 'LINEAR_RAW' page by its ID."""

        p = self._get_page_by_id(target_page_id)

        if p is None or p.photometric is None or p.photometric.name != "LINEAR_RAW":
            return None

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

        return None

    @property
    def xmp_metadata(self) -> Dict[str, str]:
        """Lazy-loaded dictionary of all XMP metadata attributes.
        
        Returns:
            Dictionary with XMP attribute names as keys (e.g., 'crs:Temperature', 'tiff:Orientation')
            and their string values. Empty dict if no XMP data found.
        """
        if self._xmp_metadata is None:
            self._xmp_metadata = self._parse_all_xmp_attributes()
        return self._xmp_metadata

    def _parse_all_xmp_attributes(self) -> Dict[str, str]:
        """Parse all XMP attributes and sequences from the XMP metadata into a dictionary.
        
        Returns:
            Dictionary mapping attribute names to values. Simple attributes map to strings
            (e.g., 'crs:Temperature': '3900'), while sequences map to comma-separated values
            (e.g., 'crs:ToneCurvePV2012': '0,0,56,30,124,125,188,212,255,255')
        """
        xmp_data = self.get_tag('XMP', return_type=str)
        if not xmp_data:
            return {}
        
        import re
        
        # Dictionary to store all XMP attributes
        attributes = {}
        
        # Pattern to match XML attributes in the format namespace:attribute="value"
        # This captures attributes like crs:Temperature="3900", tiff:Orientation="1", etc.
        attribute_pattern = r'([a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9]*)="([^"]*?)"'
        
        # Find all attribute matches
        matches = re.findall(attribute_pattern, xmp_data)
        
        for attr_name, attr_value in matches:
            attributes[attr_name] = attr_value
        

        # Pattern to match rdf:Seq structures like ToneCurvePV2012
        # Use flexible matching that handles any whitespace/newlines between tags
        # Matches: <crs:PropertyName>...<rdf:Seq>...<rdf:li>...</rdf:li>...</rdf:Seq>...</crs:PropertyName>
        seq_pattern = r'<([a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9]*?)>.*?<rdf:Seq>(.*?)</rdf:Seq>.*?</\1>'
        seq_matches = re.findall(seq_pattern, xmp_data, re.DOTALL)
        
        logger.debug(f"Found {len(seq_matches)} XMP sequences")
        for seq_name, seq_content in seq_matches:
            # Extract all rdf:li values from the sequence
            li_pattern = r'<rdf:li>([^<]*?)</rdf:li>'
            li_values = re.findall(li_pattern, seq_content)
            
            # Handle different sequence types based on content structure
            processed_values = []
            for li_value in li_values:
                # Split by comma and clean up values
                coords = [coord.strip() for coord in li_value.split(',')]
                
                if len(coords) == 1:
                    # Single value (e.g., ColorVariance: "-50.000000")
                    processed_values.append(coords[0])
                elif len(coords) == 2:
                    # Coordinate pair (e.g., ToneCurve: "0, 0")
                    processed_values.append(f"({coords[0]},{coords[1]})")
                else:
                    # Multiple values (e.g., PointColors with 19 values)
                    # Store as bracketed list for clarity
                    processed_values.append(f"[{','.join(coords)}]")
            
            attributes[seq_name] = ','.join(processed_values)
        
        logger.debug(f"Parsed {len(attributes)} XMP attributes from DNG file")
        return attributes

    def xmp_has(self, prop: str) -> bool:
        """Check if an XMP property exists.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
        
        Returns:
            True if the property exists in XMP metadata
        """
        # Auto-prepend 'crs:' if no namespace specified
        if ':' not in prop:
            prop = f'crs:{prop}'
        return prop in self.xmp_metadata

    def xmp(self, prop: str, return_type: Optional[Type] = None) -> Optional[Any]:
        """Get an XMP property value with optional type conversion.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
            return_type: Optional type to convert the value to (e.g., float, int)
        
        Returns:
            The property value, optionally converted to return_type. None if not found.
        """
        # Auto-prepend 'crs:' if no namespace specified
        if ':' not in prop:
            prop = f'crs:{prop}'
        
        value = self.xmp_metadata.get(prop)
        if value is None:
            return None
        
        if return_type is None:
            return value
        
        # Try to convert using the type's constructor
        try:
            return return_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert XMP property '{prop}' value '{value}' to type {return_type}: {e}")
            return None


def decode_raw(
    file: Union[str, Path, IO[bytes]],
    use_xmp: bool = True,
    output_dtype: type = np.uint16,
    **processing_params
) -> np.ndarray:
    """
    Decode a DNG file to a numpy array using Core Image processing.
    
    Args:
        file: Path to DNG file or file-like object containing DNG data
        use_xmp: Whether to read XMP metadata for default values
        output_dtype: Output numpy data type (np.uint8, np.uint16, np.float16, np.float32)
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
    
    Returns:
        RGB image array with shape (height, width, 3) and specified dtype
    """
    try:
        # Import here to avoid circular imports
        from . import color_mac
        import io
        
        # Read file once and use for both DngFile and process_raw_core_image
        if isinstance(file, io.IOBase):
            dng_input = file
        else:
            # Read file once into BytesIO
            with open(file, 'rb') as f:
                file_data = f.read()
            dng_input = io.BytesIO(file_data)

        dng_input.seek(0)
        dng_file = DngFile(dng_input)
        
        # Build processing options using configuration mapping
        options = {}
        
        # Only process parameters if we have XMP to read or explicit parameters to apply
        if use_xmp or processing_params:
            # Import ToneCurve here to avoid circular import issues
            from .color import ToneCurve as TC
            
            # Define mapping: param_name -> (option_name, xmp_name, value_type)
            # Use None for xmp_name to indicate CLI-only parameters (no XMP fallback)
            xmp_mappings = {
                "temperature": ("neutralTemperature", "Temperature", float),
                "tint": ("neutralTint", "Tint", float),
                "exposure": ("exposure", "Exposure2012", float),
                "tone_curve": ("toneCurve", "ToneCurvePV2012", TC),
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
        
        logger.debug(f"Processing {file} with options: {options}")
        
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

            tone_curve = options["toneCurve"]
            # Get normalized points and apply lin2srgb transformation
            normalized_points = tone_curve.to_normalized()
            transformed_points = [(lin2srgb(x), lin2srgb(y)) for x, y in normalized_points]
            # Convert back to 8-bit range and create new tone curve
            new_tone_curve = TC()
            new_tone_curve.points = [(int(x * 255), int(y * 255)) for x, y in transformed_points]
            options["toneCurve"] = new_tone_curve
        
        # Process with Core Image
        dng_input.seek(0)
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
    **processing_params
) -> bool:
    """
    Convert a DNG file to an image file.
    
    Args:
        file: Path to DNG file or file-like object containing DNG data
        output_path: Output file path (format determined by extension)
        use_xmp: Whether to read XMP metadata for default values
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import cv2
        
        # Decode DNG to 8-bit numpy array
        color_data = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=np.uint8,
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
    **processing_params
) -> bytes:
    """
    Convert a DNG file to encoded image bytes.
    
    Args:
        file: Path to DNG file or file-like object containing DNG data
        output_format: Output format ("jpg", "png", "tiff", etc.)
        use_xmp: Whether to read XMP metadata for default values
        **processing_params: Processing parameters (temperature, tint, exposure, 
                           tone_curve, noise_reduction, orientation, etc.)
        
    Returns:
        bytes: Encoded image data
    """
    try:
        import cv2
        
        # Decode DNG to 8-bit numpy array
        color_data = decode_raw(
            file=file,
            use_xmp=use_xmp,
            output_dtype=np.uint8,
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


