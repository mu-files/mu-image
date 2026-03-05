import io
import logging
import numpy as np
import os

from typing import IO, Optional, Union
from .color import SplineCurve

logger = logging.getLogger(__name__)

# --- Core Image (macOS specific) DNG Processing ---

# PyObjC imports - these will only work on macOS
try:
    import Quartz
    import Foundation
    from Foundation import NSURL, NSMutableData, NSNumber, NSValue, NSPoint, NSData 
    from Quartz import (
        CIFilter,
        CIContext,
        NSColorSpace,
        kCIFormatRGBA8,
        kCIFormatRGBA16,
        kCIFormatRGBAh,
        kCIFormatRGBAf,
        kCIContextWorkingColorSpace,
        CGColorSpaceCreateWithICCProfile,
        kCIContextUseSoftwareRenderer,
    )
    core_image_available = True
except ImportError:
    core_image_available = False

# Map user-friendly names to (QuartzConstantName, ValueTypeCategory)
# ValueTypeCategory helps in wrapping the Python value correctly.
RAW_FILTER_OPTION_MAP = {
    # Floats
    "baselineExposure": ("kCIInputBaselineExposureKey", "float"),
    "boostAmount": ("kCIInputBoostKey", "float"),
    "boostShadowAmount": ("kCIInputBoostShadowAmountKey", "float"),
    "colorNoiseReductionAmount": ("kCIInputColorNoiseReductionAmountKey", "float"),
    "contrastAmount": ("kCIInputContrastKey", "float"), # Standard CIFilter key
    "detailAmount": ("kCIInputNoiseReductionDetailAmountKey", "float"),
    "exposure": ("kCIInputEVKey", "float"),
    "extendedDynamicRangeAmount": ("kCIInputExtendedDynamicRangeAmountKey", "float"),
    "localToneMapAmount": ("kCIInputLocalToneMapAmountKey", "float"),
    "luminanceNoiseReductionAmount": ("kCIInputLuminanceNoiseReductionAmountKey", "float"),
    "moireReductionAmount": ("kCIInputMoireAmountKey", "float"),
    "neutralTemperature": ("kCIInputNeutralTemperatureKey", "float"),
    "neutralTint": ("kCIInputNeutralTintKey", "float"),
    "noiseReductionAmount": ("kCIInputNoiseReductionAmountKey", "float"),
    "noiseReductionContrastAmount": ("kCIInputNoiseReductionContrastAmountKey", "float"),
    "noiseReductionSharpnessAmount": ("kCIInputNoiseReductionSharpnessAmountKey", "float"),
    "scaleFactor": ("kCIInputScaleFactorKey", "float"),
    "shadowBias": ("kCIInputBiasKey", "float"),
    "sharpnessAmount": ("kCIInputSharpnessKey", "float"),

    # Bools
    "isChromaticNoiseTrackingEnabled": ("kCIInputEnableChromaticNoiseTrackingKey", "bool"),
    "isDraftModeEnabled": ("kCIInputAllowDraftModeKey", "bool"),
    "isEDRModeEnabled": ("kCIInputEnableEDRModeKey", "bool"),
    "isGamutMappingEnabled": ("kCIInputDisableGamutMapKey", "bool_inverted"), # True means set kCIInputDisableGamutMapKey to False
    "isLensCorrectionEnabled": ("kCIInputEnableVendorLensCorrectionKey", "bool"),
    "isSharpeningEnabled": ("kCIInputEnableSharpeningKey", "bool"),
    "ignoreImageOrientation": ("kCIInputIgnoreImageOrientationKey", "bool"),

    # Integers
    "imageOrientation": ("kCIInputImageOrientationKey", "int"),

    # CGPoints (expected as (x, y) tuples)
    "neutralChromaticity": (("kCIInputNeutralChromaticityXKey", "kCIInputNeutralChromaticityYKey"), "point_xy_floats"),
    "neutralLocation": ("kCIInputNeutralLocationKey", "point_nsvalue"),
}

def _create_tone_curve_filter(spline_curve):
    """
    Create a CIToneCurve filter from a SplineCurve object.
    
    Args:
        spline_curve: SplineCurve object with normalized 0-1 points

    Returns:
        CIFilter instance configured with tone curve points, or None if invalid
    """
    from Quartz import CIVector, CIFilter
    
    num_points = len(spline_curve.points)
    
    # Validate point count
    if num_points < 2:
        logger.warning(f"Invalid tone curve: {num_points} points provided. "
                      f"Minimum 2 points required. Skipping tone curve.")
        return None
    
    # CIToneCurve requires exactly 5 points - resample if needed
    if num_points == 5:
        final_points = spline_curve.points
    else:
        resampled = spline_curve.resample(5)
        final_points = resampled.points
    
    # Create tone curve filter
    tone_curve_filter = CIFilter.filterWithName_("CIToneCurve")
    
    # Set the 5 points
    for i in range(5):
        x, y = final_points[i]
        vector = CIVector.vectorWithX_Y_(float(x), float(y))
        tone_curve_filter.setValue_forKey_(vector, f"inputPoint{i}")
    
    return tone_curve_filter

def _strip_unique_camera_model(dng_data: bytes) -> bytes:
    """Strip UniqueCameraModel tag from DNG data to avoid Core Image camera-specific processing.
    
    When Core Image (CIRAWFilter) recognizes a known camera model via the UniqueCameraModel
    tag (e.g., "Sony ILCE-7C"), it applies camera-specific processing that can produce
    images significantly different to Adobe's dng_validate reference output.
    
    Investigation findings:
    - Original camera DNGs with UniqueCameraModel: Core Image renders correctly
    - DNGs rewritten by tifffile with UniqueCameraModel: Core Image renders ~7% differently
    - DNGs with UniqueCameraModel stripped or set to unknown model: renders correctly
    
    Uses tifffile to locate the tag, then patches the raw bytes to zero out the tag ID.
    This preserves the original compression and file structure.
    
    Args:
        dng_data: Raw DNG file bytes
        
    Returns:
        Modified DNG data with UniqueCameraModel tag zeroed out
    """
    from tifffile import TiffFile, TIFF
    
    UNIQUE_CAMERA_MODEL = TIFF.TAGS["UniqueCameraModel"]
    
    try:
        input_buffer = io.BytesIO(dng_data)
        
        with TiffFile(input_buffer) as tif:
            # Check IFD0 for UniqueCameraModel tag
            if tif.pages and UNIQUE_CAMERA_MODEL in tif.pages[0].tags:
                tag = tif.pages[0].tags[UNIQUE_CAMERA_MODEL]
                # tag.offset is the file offset of the tag entry (12 bytes: code, dtype, count, value/offset)
                tag_offset = tag.offset
                
                # Zero out the tag code (first 2 bytes of the 12-byte entry)
                data = bytearray(dng_data)
                data[tag_offset:tag_offset+2] = b'\x00\x00'
                logger.debug("Stripped UniqueCameraModel tag from DNG data for Core Image processing")
                return bytes(data)
        
        return dng_data
        
    except Exception as e:
        logger.warning(f"Failed to strip UniqueCameraModel: {e}. Using original data.")
        return dng_data


class CoreImageContext:
    """
    Manages the lifecycle of a Core Image context for rendering DNG files.
    This class is a context manager, ensuring resources are properly released for a single operation.
    """

    def __init__(self, use_gpu: bool = False, icc_profile_path: Optional[str] = None):
        if not core_image_available:
            raise RuntimeError(
                "Core Image is not available on this system. "
                "This class requires macOS with PyObjC installed."
            )

        working_space_ns = None
        output_space_cg = None

        if icc_profile_path:
            custom_cg_space = None
            try:
                with open(icc_profile_path, "rb") as f:
                    profile_data = f.read()
                custom_cg_space = CGColorSpaceCreateWithICCProfile(profile_data)
            except IOError as e:
                logger.warning(f"Could not read profile '{icc_profile_path}': {e}")

            if custom_cg_space:
                working_space_ns = NSColorSpace.alloc().initWithCGColorSpace_(
                    custom_cg_space
                )
                output_space_cg = custom_cg_space
            else:
                logger.warning(
                    f"Failed to create color space from '{icc_profile_path}'. "
                    f"Falling back to sRGB."
                )

        if not working_space_ns:
            srgb_ns_space = NSColorSpace.sRGBColorSpace()
            working_space_ns = srgb_ns_space
            output_space_cg = srgb_ns_space.CGColorSpace()

        context_options = {kCIContextWorkingColorSpace: working_space_ns}
        if not use_gpu:
            context_options[kCIContextUseSoftwareRenderer] = True

        self.context = CIContext.contextWithOptions_(context_options)
        self.output_space_cg = output_space_cg
        self._closed = False

        if not self.context:
            raise RuntimeError("Failed to create Core Image context.")

    def close(self):
        """Explicitly release the Core Image resources."""
        if getattr(self, "_closed", True):
            return

        if hasattr(self, "context"):
            self.context = None

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def process_raw_core_image(
    dng_input: Union[str, os.PathLike, IO[bytes]],
    raw_filter_options: Optional[dict] = None,
    use_gpu: bool = False,
    icc_profile_path: Optional[str] = None,
    output_dtype: type = np.uint16,
    use_system_camera_profiles: bool = True,
) -> Optional[np.ndarray]:
    """
    Processes a DNG file by creating a temporary Core Image context for the operation.
    This ensures no state is carried over between calls.
    
    Args:
        dng_input: Path to DNG file or file-like object containing DNG data
        raw_filter_options: Dictionary of Core Image raw filter options
        use_gpu: Whether to use GPU acceleration for processing
        icc_profile_path: Optional path to ICC profile for color space conversion
        output_dtype: Output numpy data type. Supported: np.uint8, np.uint16, np.float16, np.float32
        use_system_camera_profiles: If True, use macOS built-in camera profiles. If False,
            strip UniqueCameraModel tag to force generic DNG processing.
    
    Returns:
        RGB image array with shape (height, width, 3) and specified dtype, or None on failure
    """
    if not core_image_available:
        raise RuntimeError("Core Image is not available on this system.")

    from objc import autorelease_pool

    with autorelease_pool():
        with CoreImageContext(use_gpu=use_gpu, icc_profile_path=icc_profile_path) as context:
            try:
                # --- Prepare Filter Options ---
                options_copy = dict(raw_filter_options) if raw_filter_options else {}
                tone_curve = options_copy.pop('toneCurve', None)
                tone_curve_linear = options_copy.pop('toneCurveLinear', None)

                raw_options = {}
                for key, value in options_copy.items():
                    if key not in RAW_FILTER_OPTION_MAP:
                        logger.warning(f"Unknown RAW filter option key: {key}. Skipping.")
                        continue

                    quartz_key_name_or_tuple, value_type = RAW_FILTER_OPTION_MAP[key]

                    try:
                        objc_value = None
                        
                        if value_type == "float":
                            objc_value = NSNumber.numberWithFloat_(float(value))
                        elif value_type == "bool":
                            objc_value = NSNumber.numberWithBool_(bool(value))
                        elif value_type == "bool_inverted":
                            objc_value = NSNumber.numberWithBool_(not bool(value))
                        elif value_type == "int":
                            objc_value = NSNumber.numberWithInt_(int(value))
                        elif value_type == "point_nsvalue":
                            ns_point = NSPoint(x=float(value[0]), y=float(value[1]))
                            objc_value = NSValue.valueWithPoint_(ns_point)
                        elif value_type == "point_xy_floats":
                            # Special case: two separate keys
                            key_x_name, key_y_name = quartz_key_name_or_tuple
                            quartz_key_x = getattr(Quartz, key_x_name)
                            quartz_key_y = getattr(Quartz, key_y_name)
                            raw_options[quartz_key_x] = NSNumber.numberWithFloat_(float(value[0]))
                            raw_options[quartz_key_y] = NSNumber.numberWithFloat_(float(value[1]))
                        
                        # Common case: single key-value pair
                        if objc_value is not None:
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                    except AttributeError:
                        logger.warning(f"Quartz constant for '{quartz_key_name_or_tuple}' not found. Skipping.")
                    except Exception as e:
                        logger.warning(f"Error processing option {key} with value {value}: {e}. Skipping.")

                # Add linear space filter to raw_options if provided
                if tone_curve_linear is not None:
                    from Quartz import kCIInputLinearSpaceFilter
                    raw_options[kCIInputLinearSpaceFilter] = _create_tone_curve_filter(tone_curve_linear)

                # --- Create CIRAWFilter ---
                # Always read DNG data and strip UniqueCameraModel to avoid CI camera-specific processing
                if isinstance(dng_input, (str, os.PathLike)):
                    with open(dng_input, 'rb') as f:
                        dng_data = f.read()
                elif hasattr(dng_input, "read"):
                    dng_data = dng_input.read()
                else:
                    raise TypeError("dng_input must be a file path or a file-like object.")
                
                if not dng_data:
                    raise ValueError("DNG data is empty.")
                
                # Optionally strip UniqueCameraModel to force generic processing
                if not use_system_camera_profiles:
                    dng_data = _strip_unique_camera_model(dng_data)
                
                from Quartz import kCGImageSourceTypeIdentifierHint
                raw_options[kCGImageSourceTypeIdentifierHint] = "com.adobe.raw-image"
                ns_data = NSData.dataWithBytes_length_(dng_data, len(dng_data))
                raw_filter = CIFilter.filterWithImageData_options_(ns_data, raw_options or None)

                if not raw_filter:
                    raise RuntimeError("Failed to create CIRAWFilter.")

                output_ci_image = raw_filter.outputImage()

                # --- Apply Tone Curve for Contrast ---
                if tone_curve is not None:
                    post_tone_filter = _create_tone_curve_filter(tone_curve)
                    if post_tone_filter is not None:
                        post_tone_filter.setValue_forKey_(output_ci_image, "inputImage")
                        output_ci_image = post_tone_filter.outputImage()

                extent = output_ci_image.extent()
                width = int(extent.size.width)
                height = int(extent.size.height)

                # --- Render to Bitmap ---
                # Map numpy dtype to Core Image format
                dtype_to_format = {
                    np.float16: (kCIFormatRGBAh, 2),  # Half-float, 2 bytes per channel
                    np.float32: (kCIFormatRGBAf, 4),  # Float, 4 bytes per channel
                    np.uint8: (kCIFormatRGBA8, 1),    # 8-bit, 1 byte per channel
                    np.uint16: (kCIFormatRGBA16, 2),  # 16-bit, 2 bytes per channel
                }
                
                if output_dtype not in dtype_to_format:
                    supported_types = list(dtype_to_format.keys())
                    raise ValueError(
                        f"Unsupported output dtype: {output_dtype}. "
                        f"Supported types: {supported_types}"
                    )
                
                pixel_format, bytes_per_channel = dtype_to_format[output_dtype]
                row_bytes = width * 4 * bytes_per_channel  # 4 channels (RGBA)
                bitmap_buffer = NSMutableData.dataWithLength_(height * row_bytes)

                context.context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                    output_ci_image,
                    bitmap_buffer,
                    row_bytes,
                    extent,
                    pixel_format,
                    context.output_space_cg,
                )

                # Convert final buffer to NumPy array once
                rgba_image = np.frombuffer(
                    bitmap_buffer, dtype=output_dtype).reshape((height, width, 4)).copy()

                return rgba_image[:, :, :3]

            except Exception as e:
                raise RuntimeError(f"An error occurred during Core Image processing: {e}") from e


