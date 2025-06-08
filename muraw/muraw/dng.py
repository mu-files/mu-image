"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, Tuple, Union, List, Any

import tifffile
from tifffile import TIFF

# helper class to convert create a list of tags for tifffile.TiffWriter
class MetadataTags:
    BAYER_PATTERN_MAP = {
        'RGGB': (0, 1, 1, 2),  # R G / G B
        'BGGR': (2, 1, 1, 0),  # B G / G R
        'GRBG': (1, 0, 2, 1),  # G R / B G
        'GBRG': (1, 2, 0, 1)   # G B / R G
    }

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
        # Expects tag to be (tag_name_str, dtype_char_str, count, value)
        # writeonce is hardcoded to False for simplicity in this version of the class
        tag_formatted_contents = (TIFF.TAGS[tag[0]], TIFF.DATA_DTYPES[tag[1]], 
            tag[2], tag[3], False)
        self._tags.append(tag_formatted_contents)

    def add_string_tag(self, tag_name_str, string_value):
        """Helper to add a standard ASCII string tag with null termination."""
        string_value_with_null = string_value + '\x00'
        length = len(string_value_with_null)
        self.add_tag((tag_name_str, 's', length, string_value_with_null))

    def add_cfa_pattern_tag(self, cfa_pattern_key: str):
        """Helper to add the CFAPattern tag using the class's Bayer pattern map."""
        pattern_tuple = self.BAYER_PATTERN_MAP.get(cfa_pattern_key, self.BAYER_PATTERN_MAP['RGGB'])
        pattern_bytes = bytes(pattern_tuple)
        self.add_tag(('CFAPattern', 'B', 4, pattern_bytes))

    def add_matrix_as_rational_tag(self, tag_name_str: str, float_matrix_np: np.ndarray, 
                                 tiff_type_char: str, num_rational_pairs: int, 
                                 denominator: int = 10000):
        """Converts a float matrix to rationals and adds it as a tag."""
        flat_tuple_values = MetadataTags._matrix_to_rational_tuple(float_matrix_np, denominator)
        self.add_tag((tag_name_str, tiff_type_char, num_rational_pairs, flat_tuple_values))

    def get_tags(self):
        self._tags.sort(key=lambda x: x[0])
        return self._tags
        

# Default values for tags
ORIENTATION_HORIZONTAL = 1

# illuminants take values defined in the Exif standard by LightSource tag
CALIBRATIONILLUMINANT_D50 = 23  # Standard D50 (daylight at the horizon during early morning or late afternoon)
CALIBRATIONILLUMINANT_D55 = 20  # Standard D55 (warm daylight at sunrise or sunset)
CALIBRATIONILLUMINANT_D65 = 21  # Standard D65 (daylight)
CALIBRATIONILLUMINANT_D75 = 22  # Standard D75 (north sky daylight)
PREVIEWCOLORSPACE_SRGB = 1
DNG_VERSION_1_7_1_0 = (1, 7, 1, 0)
DNG_BCKWRD_VERSION_1_4_0_0 = (1, 4, 0, 0)

def _generate_dng_thumbnail(raw_cfa_data: np.ndarray, bayer_pattern_key: str) -> Optional[np.ndarray]:
    """Generate an 8-bit RGB thumbnail from raw CFA data."""
    print(f"Generating thumbnail for Bayer pattern: {bayer_pattern_key}")

    bayer_to_cvrgb_map = {
        'RGGB': cv2.COLOR_BAYER_BG2RGB,
        'BGGR': cv2.COLOR_BAYER_RG2RGB,
        'GRBG': cv2.COLOR_BAYER_GB2RGB,
        'GBRG': cv2.COLOR_BAYER_GR2RGB
    }
    
    cv_bayer_code = bayer_to_cvrgb_map.get(bayer_pattern_key)

    if not cv_bayer_code:
        msg = (f"Unknown or unsupported Bayer pattern key: "
               f"'{bayer_pattern_key}'")
        raise ValueError(msg)

    # Ensure data is suitable for cv2.cvtColor (typically uint8 or uint16 for Bayer)
    if raw_cfa_data.dtype != np.uint16 and raw_cfa_data.dtype != np.uint8:
        msg = (f"Unsupported raw data dtype for thumbnail generation: "
               f"{raw_cfa_data.dtype}. Expected uint8 or uint16.")
        raise ValueError(msg)
    
    thumbnail_rgb_full = cv2.cvtColor(raw_cfa_data, cv_bayer_code)

    # Scale to 8-bit
    if thumbnail_rgb_full.dtype == np.uint16:
        thumbnail_rgb_8bit = (thumbnail_rgb_full / 256).astype(np.uint8)  # Simple 16-bit to 8-bit scaling
    elif thumbnail_rgb_full.dtype == np.uint8:
        thumbnail_rgb_8bit = thumbnail_rgb_full

    # Resize
    max_thumb_dim = 256
    h_full, w_full = thumbnail_rgb_8bit.shape[:2]
    if h_full == 0 or w_full == 0:
        print("Warning: Thumbnail has zero dimension after scaling. Skipping resize.")
        return None
        
    if h_full > w_full:
        new_h = max_thumb_dim
        new_w = int(w_full * (max_thumb_dim / h_full))
    else:
        new_w = max_thumb_dim
        new_h = int(h_full * (max_thumb_dim / w_full))
    
    if new_w == 0 or new_h == 0:
        print(f"Warning: Calculated new thumbnail dimensions are zero "
              f"({new_w}x{new_h}). Skipping resize.")
        return None

    thumbnail_resized = cv2.resize(thumbnail_rgb_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # No rotation, as main DNG data is not currently rotated in write_dng
    print(f"Thumbnail generated successfully: {thumbnail_resized.shape[1]}x{thumbnail_resized.shape[0]}")
    return thumbnail_resized

# TODO: pass in data, destination, bpp, make, crop, model, exiftags, cfa, list of preview/thumbs,
# calibration matrix and illiuminant, jxl_dist and effort 
def write_dng(
    raw_data: np.ndarray,
    destination_file: Path,
    bits_per_pixel: int,
    camera_make: str = "Unknown",
    camera_model: str = "Unknown",
    cfa_pattern: str = 'RGGB',
    jxl_distance: Optional[float] = None,
    generate_thumbnail: bool = True
) -> None:
    """Write raw data to a DNG file using tifffile.
    
    Args:
        raw_data: Raw image data as numpy array (H, W)
        destination_file: Path where to save the DNG file
        bits_per_pixel: Number of bits per pixel (e.g. 12, 14, 16)
        camera_make: Make of the camera
        camera_model: Model of the camera
        cfa_pattern: CFA pattern string, e.g., 'RGGB'
        jxl_distance: JPEG XL Butteraugli distance. Lower is higher quality.
                     See comments in code for details. Default: None (no JXL compression).
        generate_thumbnail: Whether to generate a thumbnail image. Default: True
    """
    if raw_data.ndim != 2:
        raise ValueError(
            f"Expected 2D raw_data (height, width), got shape {raw_data.shape}"
        )

    # Generate thumbnail if requested
    thumbnail_image = None
    if generate_thumbnail and cfa_pattern is not None:
        try:
            thumbnail_image = _generate_dng_thumbnail(raw_data, cfa_pattern.upper())
            if thumbnail_image is not None:
                print(f"Generated thumbnail: {thumbnail_image.shape[1]}x{thumbnail_image.shape[0]}")
        except ValueError as e:
            print(f"Warning: Could not generate DNG thumbnail: {e}. Proceeding without thumbnail.")
        except Exception as e:
            print(f"Warning: Unexpected error generating thumbnail: {e}. Proceeding without thumbnail.")

    # format the DNG specific tags for CFAs
    color_matrix_floats = np.array([
        [1.402600, -0.642900, -0.063300],
        [0.348200, 0.441700, 0.2185],
        [0.327800, -0.009700, 0.840500]
    ], dtype=np.float64)
    illuminant = CALIBRATIONILLUMINANT_D55

    # TODO: come up with a way to calculate this
    _as_shot_neutral_orig = ((1,1), (1,1), (1,1))
    as_shot_neutral_values_flat = tuple(item for pair in _as_shot_neutral_orig for item in pair)

    camera_model_utf8_bytes_null = (camera_model + '\x00').encode('utf-8')

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    dng_tags = MetadataTags()
    dng_tags.add_tag(('Orientation', 'H', 1, ORIENTATION_HORIZONTAL))
    dng_tags.add_cfa_pattern_tag(cfa_pattern)
    dng_tags.add_matrix_as_rational_tag('ColorMatrix1', color_matrix_floats, '2i', 9)
    dng_tags.add_tag(('CalibrationIlluminant1', 'H', 1, illuminant))
    dng_tags.add_tag(('AsShotNeutral', '2I', 3, as_shot_neutral_values_flat))
    dng_tags.add_string_tag('Make', camera_make)
    dng_tags.add_string_tag('Model', camera_model)
    dng_tags.add_string_tag('UniqueCameraModel', camera_model)
    dng_tags.add_tag(('LocalizedCameraModel', 'B',
        len(camera_model_utf8_bytes_null), camera_model_utf8_bytes_null))
    dng_tags.add_tag(('CFARepeatPatternDim', 'H', 2, (2,2)))
    dng_tags.add_tag(('CFAPlaneColor', 'B', 3, (0,1,2)))
    dng_tags.add_tag(('DNGVersion', 'B', 4, DNG_VERSION_1_7_1_0))
    dng_tags.add_tag(('DNGBackwardVersion', 'B', 4, DNG_BCKWRD_VERSION_1_4_0_0))

    try:
        with tifffile.TiffWriter(destination_file, bigtiff=False) as tif:
            # Prepare main image arguments
            main_image_ifd0_args = {
                'subfiletype': 0,
                'photometric': 'cfa',
                'extratags': dng_tags.get_tags(),
                'software': "allsky/raw"
            }
            if thumbnail_image is not None:
                main_image_ifd0_args['subifds'] = 1      # Indicates it HAS a SubIFD (the thumbnail)

            if jxl_distance is not None:
                if not (0.0 <= jxl_distance <= 15.0):
                    print(
                        f"Warning: JXL distance {jxl_distance} is outside the "
                        f"typical range [0.0, 15.0]. "
                    )
                main_image_ifd0_args['compression'] = 'JPEGXL_DNG'
                main_image_ifd0_args['compressionargs'] = {'distance': jxl_distance}
                print(f"Attempting to write DNG with JXL compression, distance: {jxl_distance}")

            # Write Main Raw Image to IFD 0
            tif.write(
                processed_raw_data,
                **main_image_ifd0_args
            )
            print("Main DNG image data (IFD 0) written.")

            if thumbnail_image is not None:
                # Prepare thumbnail specific tags
                preview_extratags = MetadataTags() # Still using MetadataTags for thumbnail tags
                preview_extratags.add_tag(('PreviewColorSpace', 'I', 1, PREVIEWCOLORSPACE_SRGB))

                # Write Thumbnail to SubIFD 1
                thumb_subifd_args = {
                    'photometric': 'rgb',  # Interprets data as RGB
                    'planarconfig': 1,     # Standard for RGB: 1 = CONTIG
                    'compression': 'jpeg', # JPEG compression for thumbnail
                    'compressionargs': {'level': 90},  # JPEG quality (0-100, higher is better)
                    'extratags': preview_extratags.get_tags(),
                    'subfiletype': 1,  # Reduced resolution image (standard for DNG previews)
                    'subifds': 0       # No further SubIFDs
                }
                tif.write(
                    thumbnail_image, # Use the thumbnail_image directly
                    **thumb_subifd_args
                )
                print(f"Thumbnail (SubIFD 1) written to {destination_file}")
            
        print(f"Successfully wrote DNG file to {destination_file}")

    except Exception as e:
        print(f"Error saving DNG file with tifffile.TiffWriter: {e}")
        raise
