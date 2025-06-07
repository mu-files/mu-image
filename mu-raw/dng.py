"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, Tuple, Union, List, Any

import tifffile
from tifffile import TIFF

# Standard TIFF Tag Codes
TAG_MAKE = TIFF.TAGS['Make']
TAG_MODEL = TIFF.TAGS['Model']
TAG_ORIENTATION = TIFF.TAGS['Orientation']
TAG_NEWSUBFILETYPE = TIFF.TAGS['NewSubfileType']
TAG_IMAGE_WIDTH = TIFF.TAGS['ImageWidth']
TAG_IMAGE_LENGTH = TIFF.TAGS['ImageLength']
TAG_BITS_PER_SAMPLE = TIFF.TAGS['BitsPerSample']
TAG_SAMPLES_PER_PIXEL = TIFF.TAGS['SamplesPerPixel']

# DNG Specific or Critical TIFF Tag Codes
TAG_CFAREPEATPATTERNDIM = TIFF.TAGS['CFARepeatPatternDim']
TAG_CFAPATTERN = TIFF.TAGS['CFAPattern']
TAG_CFAPLANECOLOR = TIFF.TAGS['CFAPlaneColor']
TAG_DNGVERSION = TIFF.TAGS['DNGVersion']
TAG_DNGBACKWARDVERSION = TIFF.TAGS['DNGBackwardVersion']
TAG_UNIQUECAMERAMODEL = TIFF.TAGS['UniqueCameraModel']
TAG_LOCALIZEDCAMERAMODEL = TIFF.TAGS['LocalizedCameraModel']
TAG_BLACKLEVEL = TIFF.TAGS['BlackLevel']
TAG_WHITELEVEL = TIFF.TAGS['WhiteLevel']
TAG_COLORMATRIX1 = TIFF.TAGS['ColorMatrix1']
TAG_ANALOGBALANCE = TIFF.TAGS['AnalogBalance']
TAG_ASSHOTNEUTRAL = TIFF.TAGS['AsShotNeutral']
TAG_BASELINEEXPOSURE = TIFF.TAGS['BaselineExposure']
TAG_PREVIEWCOLORSPACE = TIFF.TAGS['PreviewColorSpace']
TAG_CALIBRATIONILLUMINANT1 = TIFF.TAGS['CalibrationIlluminant1']

# Default values for tags
ORIENTATION_HORIZONTAL = 1
# Common Bayer patterns: RGGB, BGGR, GRBG, GBRG
BAYER_PATTERN_MAP = {
    'RGGB': (0, 1, 1, 2),  # R G / G B
    'BGGR': (2, 1, 1, 0),  # B G / G R
    'GRBG': (1, 0, 2, 1),  # G R / B G
    'GBRG': (1, 2, 0, 1)   # G B / R G
}

# illuminants take values defined in the Exif standard by LightSource tag
CALIBRATIONILLUMINANT_D50 = 23  # Standard D50 (daylight at the horizon during early morning or late afternoon)
CALIBRATIONILLUMINANT_D55 = 20  # Standard D55 (warm daylight at sunrise or sunset)
CALIBRATIONILLUMINANT_D65 = 21  # Standard D65 (daylight)
CALIBRATIONILLUMINANT_D75 = 22  # Standard D75 (north sky daylight)
PREVIEWCOLORSPACE_SRGB = 1
DNG_VERSION_1_7_1_0 = (1, 7, 1, 0)
DNG_BACKWARD_VERSION_1_4_0_0 = (1, 4, 0, 0)

def _generate_dng_thumbnail(raw_cfa_data: np.ndarray, bayer_pattern_key: str) -> Optional[np.ndarray]:
    """Generate an 8-bit RGB thumbnail from raw CFA data."""
    print(f"Generating thumbnail for Bayer pattern: {bayer_pattern_key}")

    bayer_to_rgb_map = {
        'RGGB': cv2.COLOR_BAYER_BG2RGB,
        'BGGR': cv2.COLOR_BAYER_RG2RGB,
        'GRBG': cv2.COLOR_BAYER_GB2RGB,
        'GBRG': cv2.COLOR_BAYER_GR2RGB
    }
    
    cv_bayer_code = bayer_to_rgb_map.get(bayer_pattern_key)

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

    color_matrix_floats = np.array([
        [1.402600, -0.642900, -0.063300],
        [0.348200, 0.441700, 0.2185],
        [0.327800, -0.009700, 0.840500]
    ], dtype=np.float64)
    illuminant = CALIBRATIONILLUMINANT_D55

    # Convert color matrix to rational pairs
    denominator = 10000  
    color_matrix_rational_pairs = []
    for r in range(color_matrix_floats.shape[0]):
        for c in range(color_matrix_floats.shape[1]):
            val = color_matrix_floats[r, c]
            numerator = int(round(val * denominator))
            color_matrix_rational_pairs.append((numerator, denominator))
    color_matrix1_values_flat = tuple(item for pair in color_matrix_rational_pairs for item in pair)

    # TODO: come up with a way to calculate this
    _as_shot_neutral_orig = ((1,1), (1,1), (1,1))
    as_shot_neutral_values_flat = tuple(item for pair in _as_shot_neutral_orig for item in pair)

    camera_make_ascii_null = camera_make + '\x00'
    camera_model_ascii_null = camera_model + '\x00'
    camera_model_utf8_bytes_null = (camera_model + '\x00').encode('utf-8')

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    main_image_dng_metadata_tags = [
        (TAG_ORIENTATION, TIFF.DATA_DTYPES['H'], 1, ORIENTATION_HORIZONTAL, False),
        (TAG_CFAPATTERN, TIFF.DATA_DTYPES['B'], 4, 
        bytes(BAYER_PATTERN_MAP.get(cfa_pattern, BAYER_PATTERN_MAP['RGGB'])), False),
        (TAG_COLORMATRIX1, 10, 9, color_matrix1_values_flat, False),
        (TAG_CALIBRATIONILLUMINANT1, TIFF.DATA_DTYPES['H'], 1,
            illuminant, False),
        (TAG_ASSHOTNEUTRAL, 5, 3, as_shot_neutral_values_flat, False),
        (TAG_MAKE, 's', len(camera_make_ascii_null),
            camera_make_ascii_null, False),
        (TAG_MODEL, 's', len(camera_model_ascii_null),
            camera_model_ascii_null, False),
        (TAG_UNIQUECAMERAMODEL, 's', len(camera_model_ascii_null),
            camera_model_ascii_null, False),
        (TAG_LOCALIZEDCAMERAMODEL, TIFF.DATA_DTYPES['B'],
            len(camera_model_utf8_bytes_null), camera_model_utf8_bytes_null, False),
        (TAG_CFAREPEATPATTERNDIM, TIFF.DATA_DTYPES['H'], 2, (2,2), False),
        (TAG_CFAPLANECOLOR, TIFF.DATA_DTYPES['B'], 3, (0,1,2), True),
        (TAG_DNGVERSION, TIFF.DATA_DTYPES['B'], 4, DNG_VERSION_1_7_1_0, False),
        (TAG_DNGBACKWARDVERSION, TIFF.DATA_DTYPES['B'], 4,
            DNG_BACKWARD_VERSION_1_4_0_0, False)
    ]

    preview_extratags = []
    if thumbnail_image is not None and thumbnail_image.ndim == 3 and thumbnail_image.shape[2] == 3 and thumbnail_image.dtype == np.uint8:
        preview_extratags.extend([
            (TAG_PREVIEWCOLORSPACE, TIFF.DATA_DTYPES['I'], 1, PREVIEWCOLORSPACE_SRGB, False)
        ])
        preview_extratags.sort(key=lambda x: x[0])

    # Sort extratags by tag ID (the first element of each tuple)
    main_image_dng_metadata_tags.sort(key=lambda x: x[0])

    try:
        imwrite_kwargs = {
            'photometric': 'cfa',
            'extratags': main_image_dng_metadata_tags,
            'software': "allsky/raw"
        }

        if jxl_distance is not None:
            if not (0.0 <= jxl_distance <= 15.0):
                print(
                    f"Warning: JXL distance {jxl_distance} is outside the "
                    f"typical range [0.0, 15.0]. "
                )
            imwrite_kwargs['compression'] = 'JPEGXL_DNG'
            imwrite_kwargs['compressionargs'] = {'distance': jxl_distance}
            print(f"Attempting to write DNG with JXL compression, distance: {jxl_distance}")
        else:
            print("Writing DNG with no explicit JXL compression.")

        # Thumbnail / SubIFD handling
        thumbnail_ifd_dict = None
        if thumbnail_image is not None and thumbnail_image.ndim == 3 and thumbnail_image.shape[2] == 3 and thumbnail_image.dtype == np.uint8:
            thumb_h, thumb_w = thumbnail_image.shape[:2]
            print(f"Including thumbnail: {thumb_w}x{thumb_h}, RGB, JPEG compressed.")

            # Define tags for the thumbnail SubIFD
            thumbnail_tags = []
            # (TAG_ORIENTATION, TIFF.DATA_DTYPES['H'], 1, ORIENTATION_HORIZONTAL, False), # Example if needed

            thumbnail_ifd_dict = {
                'data': thumbnail_image,
                'tags': thumbnail_tags,
                'photometric': 'rgb',  # Interprets data as RGB
                'planarconfig': 1,     # Standard for RGB: 1 = CONTIG
                'compression': 'jpeg',  # JPEG compression for thumbnail
                'compressionargs': {'level': 90}  # JPEG quality (0-100, higher is better)
            }
        elif thumbnail_image is not None:
            print(f"Warning: Thumbnail image provided but not in expected uint8 HxWx3 RGB format. "
                  f"Shape: {thumbnail_image.shape}, Dtype: {thumbnail_image.dtype}. Skipping thumbnail.")

        # Use TiffWriter for more control, especially with SubIFDs
        with tifffile.TiffWriter(destination_file, bigtiff=False) as tif:
            if thumbnail_ifd_dict:
                # Write Main Raw Image to IFD 0 first
                print(f"Writing main DNG image data to IFD 0...")
                main_image_ifd0_args = imwrite_kwargs.copy()
                main_image_ifd0_args['subfiletype'] = 0  # Full-resolution image
                main_image_ifd0_args['subifds'] = 1      # Indicates it HAS a SubIFD (the thumbnail)
                main_image_ifd0_args['extratags'] = main_image_dng_metadata_tags  # Ensure correct tags

                tif.write(
                    processed_raw_data,
                    **main_image_ifd0_args
                )
                print("Main DNG image data (IFD 0) written.")

                # Write Thumbnail to SubIFD 1
                print(f"Writing thumbnail to SubIFD 1...")
                thumb_subifd_args = {
                    'photometric': thumbnail_ifd_dict['photometric'],
                    'planarconfig': thumbnail_ifd_dict['planarconfig'],
                    'compression': thumbnail_ifd_dict['compression'],
                    'compressionargs': thumbnail_ifd_dict['compressionargs'],
                    'subfiletype': 1,  # Reduced resolution image
                    'subifds': 0       # No further SubIFDs
                }
                tif.write(
                    thumbnail_ifd_dict['data'],
                    extratags=preview_extratags,
                    **thumb_subifd_args
                )
                print(f"Thumbnail (SubIFD 1) written to {destination_file}")
            else:
                # No thumbnail, write Main Raw Image to IFD 0
                print(f"Writing main DNG image data to IFD 0...")
                main_image_final_args = imwrite_kwargs.copy()
                main_image_final_args['subfiletype'] = 0  # Full-resolution image
                main_image_final_args['subifds'] = 0     # No SubIFDs
                main_image_final_args['extratags'] = main_image_dng_metadata_tags  # Ensure correct tags

                tif.write(
                    processed_raw_data,
                    **main_image_final_args
                )
                print(f"Main DNG image data (IFD 0) written to {destination_file}")
            
            print(f"Successfully wrote DNG file to {destination_file}")
    except Exception as e:
        print(f"Error saving DNG file with tifffile.TiffWriter: {e}")
        raise
