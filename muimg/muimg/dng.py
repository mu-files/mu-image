"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import cv2
import io
import logging
import numpy as np

from pathlib import Path
from tifffile import PHOTOMETRIC, TiffFile, TiffPage, TiffWriter, TIFF
from typing import Optional, Union, List, Dict, Tuple, Any, Type, Union

logger = logging.getLogger(__name__)

BAYER_PATTERN_MAP = {
    "RGGB": (0, 1, 1, 2),  # R G / G B
    "BGGR": (2, 1, 1, 0),  # B G / G R
    "GRBG": (1, 0, 2, 1),  # G R / B G
    "GBRG": (1, 2, 0, 1),  # G B / R G
}

# helper class to convert create a list of tags for tifffile.TiffWriter
class MetadataTags:

    @staticmethod
    def _matrix_to_rational_tuple(matrix: np.ndarray, denominator: int = 10000) -> tuple:
        """Converts a NumPy float matrix to a flat tuple of (numerator, denominator) pairs."""
        rational_pairs = []
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                numerator = int(round(val * denominator))
                rational_pairs.append((numerator, denominator))
        return tuple(item for pair in rational_pairs for item in pair)

    def __init__(self):
        self._tags = []

    def add_tag(self, tag):
        tag_code = -1

        if isinstance(tag[0], str):
            tag_code = TIFF.TAGS[tag[0]]
        elif isinstance(tag[0], int):
            tag_code = tag[0]

        tag_formatted_contents = (tag_code, TIFF.DATA_DTYPES[tag[1]], tag[2], tag[3], False)
        
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
        self._tags.extend(other._tags)

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

    def add_exif_dict(self, exif_dict: dict):
        # TODO: need to add more comprehensive list but just select those we need for now
        for ifd_name in ['0th', 'Exif']:
            if ifd_name in exif_dict and isinstance(exif_dict[ifd_name], dict):
                for tag_id, value in exif_dict[ifd_name].items():
                    if isinstance(value, str):
                        str_value = value
                        self.add_string_tag(tag_id, str_value)
                    elif tag_id == TIFF.TAGS["DateTimeOriginal"]:
                        # DateTimeOriginal is a string (date/time when original image was taken).
                        self.add_string_tag(tag_id, value)
                    elif tag_id == TIFF.TAGS["ExposureTime"]:
                        # ExposureTime is an unsigned rational (numerator, denominator).
                        self.add_tag((tag_id, "2I", 1, value))
                    elif tag_id == TIFF.TAGS["Temperature"]:
                        # Temperature is a signed rational (numerator, denominator).
                        self.add_tag((tag_id, "2i", 1, value))
                    elif tag_id == TIFF.TAGS["MakerNote"]:
                        # MakerNote is a byte array.
                        self.add_tag((tag_id, "B", len(value), value))
                    # TODO: Add other specific, non-string tag handlers here (e.g., shorts, longs).

    def get_tags(self):
        self._tags.sort(key=lambda x: x[0])
        return self._tags


# Default values for tags
ORIENTATION_HORIZONTAL = 1

# illuminants take values defined in the Exif standard by LightSource tag
CALIBRATIONILLUMINANT_UNKNOWN = 0
CALIBRATIONILLUMINANT_D55 = 20  # Standard D55 (warm daylight at sunrise or sunset)
CALIBRATIONILLUMINANT_D65 = 21  # Standard D65 (daylight)
CALIBRATIONILLUMINANT_D75 = 22  # Standard D75 (north sky daylight)
CALIBRATIONILLUMINANT_D50 = (
    23  # Standard D50 (daylight at the horizon during early morning or late afternoon)
)

PREVIEWCOLORSPACE_SRGB = 1


# Camera Color Profiles
class CameraProfiles:
    def __init__(self):
        # Initial definition with human-readable keys
        initial_profiles = {
            "ASI676MC": {
                "color_matrix1": np.array(
                    [
                        [1.402600, -0.642900, -0.063300],
                        [0.348200, 0.441700, 0.2185],
                        [0.327800, -0.009700, 0.840500],
                    ],
                    dtype=np.float64,
                ),
                "illuminant1": CALIBRATIONILLUMINANT_D55,
            },
            "ASI678MC": {
                "color_matrix1": np.array(
                    [
                        (1.0, -0.6518, 0.0555),
                        (0.4936, -0.2341, 0.0898),
                        (0.6951, -0.4777, 0.2660),
                    ],
                    dtype=np.float64,
                ),
                "illuminant1": CALIBRATIONILLUMINANT_D65,
            },
            # Add other camera models here, e.g.:
            # "AnotherCameraModel": {
            #     "color_matrix": np.array([...]),
            #     "illuminant": CALIBRATIONILLUMINANT_DAYLIGHT # Or another specific illuminant
            # },
            "DEFAULT": {
                "color_matrix1": np.identity(3, dtype=np.float64),
                "illuminant1": CALIBRATIONILLUMINANT_UNKNOWN,
            },
        }

        # Internal storage with uppercase keys for case-insensitive lookup
        self._normalized_profiles = {k.upper(): v for k, v in initial_profiles.items()}

    def get(self, camera_model_str: str) -> dict:
        """Retrieve a camera profile using case-insensitive key matching."""
        profile = self._normalized_profiles.get(camera_model_str.upper())
        if profile is None:
            logger.warning(f"Camera profile for '{camera_model_str}' not found. Falling back to DEFAULT.")
            return self._normalized_profiles.get("DEFAULT")
        return profile


_CAMERA_PROFILES_INSTANCE = CameraProfiles()


def _generate_dng_thumbnail(color_data: np.ndarray) -> Optional[np.ndarray]:
    """Generate an 8-bit RGB thumbnail from color data."""

    # Resize
    h_full, w_full = color_data.shape[:2]
    if h_full > w_full:
        new_h = int(max(h_full / 4, 256))
        new_w = int(w_full * (new_h / h_full))
    else:
        new_w = int(max(w_full / 4, 256))
        new_h = int(h_full * (new_w / w_full))

    thumbnail_resized = cv2.resize(color_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # No rotation, as main DNG data is not currently rotated in write_dng
    logger.info(f"Thumbnail generated successfully: {thumbnail_resized.shape[1]}x{thumbnail_resized.shape[0]}")
    return thumbnail_resized


def swizzle_cfa_data(raw_data: np.ndarray) -> np.ndarray:
    """Swizzle RGGB CFA data into a 2x2 grid of R, G1, G2, B sub-images."""

    # Extract the four channels assuming RGGB pattern
    # R pixels: top-left of each 2x2 block (e.g., raw_data[0,0], raw_data[0,2], ... raw_data[2,0], ...)
    r_channel = raw_data[0::2, 0::2]
    # G1 pixels: top-right of each 2x2 block (e.g., raw_data[0,1], raw_data[0,3], ...)
    g1_channel = raw_data[0::2, 1::2]
    # G2 pixels: bottom-left of each 2x2 block (e.g., raw_data[1,0], raw_data[1,2], ...)
    g2_channel = raw_data[1::2, 0::2]
    # B pixels: bottom-right of each 2x2 block (e.g., raw_data[1,1], raw_data[1,3], ...)
    b_channel = raw_data[1::2, 1::2]

    # Assemble the swizzled data using np.block
    swizzled_data = np.block([[r_channel, g1_channel], [g2_channel, b_channel]])

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


# TODO: pass in exiftags, list of preview/thumbs, calibration matrix and illuminant


def write_dng(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    camera_make: str = "Unknown",
    camera_model: str = "Unknown",
    cfa_pattern: str = "RGGB",
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    color_data: Optional[np.ndarray] = None,
    exif_dict: Optional[dict] = None,
    external_camera_profile: Optional["MetadataTags"] = None  
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        raw_data: Raw image data as numpy array (H, W)
        destination_file: Path or io.BytesIO object where to save the DNG file.
        bits_per_pixel: Number of bits per pixel (e.g. 12, 14, 16)
        camera_make: Make of the camera
        camera_model: Model of the camera
        cfa_pattern: CFA pattern string, e.g., 'RGGB'
        jxl_distance: JPEG XL Butteraugli distance. Lower is higher quality.
                     Default: None (no JXL compression).
        jxl_effort: JPEG XL compression effort (1-9). Higher is more compression/slower.
                    Only used if jxl_distance is also specified. Default: None (codec default).
        color_data: Optional color data for preview
        exif_dict: Optional EXIF data in dict format
        external_camera_profile: Optional MetadataTags instance to override default 
                                 color matrix and illuminant
    """

    if isinstance(destination_file, Path):
        logger.info(f"Writing DNG to {destination_file}")
    else:
        logger.info("Writing DNG to in-memory buffer")

    # TODO: implement param validation here - raw_data not none, W/H even, cfa pattern valid, bpp <= 16

    if raw_data.ndim != 2:
        raise ValueError(f"Expected 2D raw_data (height, width), got shape {raw_data.shape}")

    # Generate thumbnail if requested
    thumbnail_image = None
    if color_data is not None:
        try:
            thumbnail_image = _generate_dng_thumbnail(color_data)
            logger.info(f"Generated thumbnail: {thumbnail_image.shape[1]}x{thumbnail_image.shape[0]}")
        except Exception as e:
            logger.warning(f"Could not generate DNG thumbnail: {e}. Proceeding without thumbnail.")

    camera_model_utf8_bytes_null = (camera_model + "\x00").encode("utf-8")

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        bits_per_pixel = 16
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        bits_per_pixel = 8
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    '''
    DATA TYPES (2nd argument to add_tag) used in tifffile.py below have following mapping:
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

    # the tag names are from tifffile.py TiffTagRegistry
    # the tag types are from tifffile.py DATA_DTYPES
    dng_tags = MetadataTags()
    dng_tags.add_tag(("Orientation", "H", 1, ORIENTATION_HORIZONTAL))

    # Use external profile if provided
    if external_camera_profile is not None:
        dng_tags.extend(external_camera_profile)
    else:
        profile = _CAMERA_PROFILES_INSTANCE.get(camera_model)
        color_matrix_floats = profile["color_matrix1"]
        illuminant = profile["illuminant1"]

        dng_tags.add_matrix_as_rational_tag("ColorMatrix1", color_matrix_floats)
        dng_tags.add_tag(("CalibrationIlluminant1", "H", 1, illuminant))
    
    # Add AsShotNeutral only if not already provided in external_camera_profile
    has_as_shot_neutral = False
    if external_camera_profile is not None:
        # Check if AsShotNeutral is already in the external tags
        for tag in external_camera_profile.get_tags():
            if tag[0] == TIFF.TAGS["AsShotNeutral"]:
                has_as_shot_neutral = True
                break
    
    if not has_as_shot_neutral:
        # Use default AsShotNeutral values
        _as_shot_neutral_orig = ((72, 100), (100, 100), (100, 100))
        as_shot_neutral_values_flat = tuple(item for pair in _as_shot_neutral_orig for item in pair)
        dng_tags.add_tag(("AsShotNeutral", "2I", 3, as_shot_neutral_values_flat))
    
    dng_tags.add_string_tag("Make", camera_make)
    dng_tags.add_string_tag("Model", camera_model)
    dng_tags.add_string_tag("UniqueCameraModel", camera_model)
    dng_tags.add_tag(
        (
            "LocalizedCameraModel",
            "B",
            len(camera_model_utf8_bytes_null),
            camera_model_utf8_bytes_null,
        )
    )
    dng_tags.add_tag(("DNGVersion", "B", 4, (1, 7, 1, 0)))
    if jxl_distance is None:
        # need latest version for CFA compression but lots of old software can't handle it
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 4, 0, 0)))
    else:
        dng_tags.add_tag(("DNGBackwardVersion", "B", 4, (1, 7, 1, 0)))

    if( exif_dict is not None ):
        dng_tags.add_exif_dict(exif_dict)

    dng_cfa_tags = MetadataTags()
    dng_cfa_tags.add_cfa_pattern_tag(cfa_pattern)
    dng_cfa_tags.add_tag(("CFARepeatPatternDim", "H", 2, (2, 2)))
    dng_cfa_tags.add_tag(("CFAPlaneColor", "B", 3, (0, 1, 2)))

    try:
        with TiffWriter(destination_file, bigtiff=False) as tif:
            if thumbnail_image is not None:
                # Prepare thumbnail specific tags
                dng_tags.add_tag(("PreviewColorSpace", "I", 1, PREVIEWCOLORSPACE_SRGB))

                # Write Thumbnail to SubIFD 0
                # TODO: could overwrite software if that tag is present in exif_dict
                thumb_ifd_args = {
                    "photometric": "rgb",  # Interprets data as RGB
                    "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
                    "compression": "jpeg",  # JPEG compression for thumbnail
                    "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
                    "extratags": dng_tags.get_tags(),
                    "subfiletype": 1,  # Reduced resolution image (standard for DNG previews)
                    "subifds": 1,  # Has main image as subifd
                    "software": "muimg",
                }
                # set datasize to max uncompressed size to avoid writing strips
                datasize = thumbnail_image.shape[0] * thumbnail_image.shape[1] * 3
                tif.write(
                    thumbnail_image,  # Use the thumbnail_image directly
                    **thumb_ifd_args,
                    rowsperstrip=datasize,
                )

            # Prepare main image arguments
            main_image_ifd_args = {
                "subfiletype": 0,
                "photometric": "cfa",
                "subifds": 0,
                "software": "muimg",
            }

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
                compressionargs = {"distance": jxl_distance}
                if jxl_effort is None:
                    jxl_effort = 5
                compressionargs["effort"] = jxl_effort
                logger.info(
                    f"Attempting to write DNG with JXL compression, distance: {jxl_distance}, effort: {jxl_effort}"
                )

                main_image_ifd_args["compression"] = compression_type
                main_image_ifd_args["compressionargs"] = compressionargs

                # if compressing, need to swizzle the CFA data and indicate
                # this via tags
                processed_raw_data = swizzle_cfa_data(processed_raw_data)
                dng_cfa_tags.add_tag(("ColumnInterleaveFactor", "H", 1, 2))
                dng_cfa_tags.add_tag(("RowInterleaveFactor", "H", 1, 2))
                dng_cfa_tags.add_tag(("JXLDistance", "f", 1, jxl_distance))
                dng_cfa_tags.add_tag(("JXLEffort", "I", 1, jxl_effort))

            if thumbnail_image is None:
                dng_cfa_tags.extend(dng_tags)
            
            main_image_ifd_args["extratags"] = dng_cfa_tags.get_tags()

            # Write Main Raw Image to IFD
            datasize = (
                processed_raw_data.shape[0] * processed_raw_data.shape[1] * bits_per_pixel / 8
            )
            tif.write(processed_raw_data, **main_image_ifd_args, rowsperstrip=datasize)

        logger.info(f"Successfully wrote DNG file to {destination_file}")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
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

    def _get_page_by_id(self, target_page_id: int) -> Optional[TiffPage]:
        """Helper to retrieve a specific TiffPage by its flattened, 0-based ID."""
        for i, page in enumerate(self._iter_all_pages_recursive(self.pages)):
            if i == target_page_id:
                return page
        return None

    def get_raw_cfa_by_id(self, target_page_id: int) -> Optional[np.ndarray]:
        """Retrieves the raw data array for a specific 'CFA' page by its ID."""

        p = self._get_page_by_id(target_page_id)

        if p is None or p.photometric is None or p.photometric.name != "CFA":
            return None

        col_interleave_tag = p.tags.get(TIFF.TAGS["ColumnInterleaveFactor"])
        row_interleave_tag = p.tags.get(TIFF.TAGS["RowInterleaveFactor"])
        if (
            col_interleave_tag is not None
            and col_interleave_tag.value == 2
            and row_interleave_tag is not None
            and row_interleave_tag.value == 2
        ):
            return deswizzle_cfa_data(p.asarray())

        return p.asarray()

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

        pages_to_search = [self.pages[ifd]] if ifd is not None and ifd < len(self.pages) else self.pages

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
