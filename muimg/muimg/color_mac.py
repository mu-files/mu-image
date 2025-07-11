import logging
import numpy as np
import os

from typing import IO, Optional, Union

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
        kCIFormatRGBA16,
        kCIContextWorkingColorSpace,
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

        from Quartz import (
            CGColorSpaceCreateWithICCProfile,
            kCIContextUseSoftwareRenderer,
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


def process_dng(
    dng_input: Union[str, os.PathLike, IO[bytes]],
    raw_filter_options: Optional[dict] = None,
    use_gpu: bool = False,
    icc_profile_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Processes a DNG file by creating a temporary Core Image context for the operation.
    This ensures no state is carried over between calls.
    """
    if not core_image_available:
        raise RuntimeError("Core Image is not available on this system.")

    from objc import autorelease_pool

    with autorelease_pool():
        with CoreImageContext(use_gpu=use_gpu, icc_profile_path=icc_profile_path) as context:
            try:
                # --- Prepare Filter Options ---
                options_copy = dict(raw_filter_options) if raw_filter_options else {}
                contrast_strength = options_copy.pop('contrastStrength', None)

                raw_options = {}
                for key, value in options_copy.items():
                    if key not in RAW_FILTER_OPTION_MAP:
                        logger.warning(f"Unknown RAW filter option key: {key}. Skipping.")
                        continue

                    map_entry = RAW_FILTER_OPTION_MAP[key]
                    quartz_key_name_or_tuple, value_type = map_entry

                    # --- Validate Input Types ---
                    if value_type == "float" and not isinstance(value, (float, int)):
                        raise TypeError(f"Invalid type for '{key}'. Expected float, got {type(value).__name__}.")
                    if value_type == "bool" and not isinstance(value, bool):
                        raise TypeError(f"Invalid type for '{key}'. Expected bool, got {type(value).__name__}.")
                    if value_type == "bool_inverted" and not isinstance(value, bool):
                        raise TypeError(f"Invalid type for '{key}'. Expected bool, got {type(value).__name__}.")
                    if value_type == "int" and not isinstance(value, int):
                        raise TypeError(f"Invalid type for '{key}'. Expected int, got {type(value).__name__}.")
                    if value_type in ("point_xy_floats", "point_nsvalue"):
                        if not isinstance(value, (list, tuple)) or len(value) != 2:
                            raise TypeError(f"Invalid type or format for '{key}'. Expected a list/tuple of 2 numbers, got {value}.")
                        if not all(isinstance(v, (float, int)) for v in value):
                            raise TypeError(f"Invalid item types for '{key}'. Both items must be numbers, got {[type(v).__name__ for v in value]}.")

                    try:
                        if value_type == "float":
                            objc_value = NSNumber.numberWithFloat_(float(value))
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                        elif value_type == "bool":
                            objc_value = NSNumber.numberWithBool_(bool(value))
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                        elif value_type == "bool_inverted":
                            objc_value = NSNumber.numberWithBool_(not bool(value))
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                        elif value_type == "int":
                            objc_value = NSNumber.numberWithInt_(int(value))
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                        elif value_type == "point_xy_floats":
                            key_x_name, key_y_name = quartz_key_name_or_tuple
                            quartz_key_x = getattr(Quartz, key_x_name)
                            quartz_key_y = getattr(Quartz, key_y_name)
                            raw_options[quartz_key_x] = NSNumber.numberWithFloat_(float(value[0]))
                            raw_options[quartz_key_y] = NSNumber.numberWithFloat_(float(value[1]))
                        elif value_type == "point_nsvalue":
                            ns_point = NSPoint(x=float(value[0]), y=float(value[1]))
                            objc_value = NSValue.valueWithPoint_(ns_point)
                            quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                            raw_options[quartz_key] = objc_value
                    except AttributeError:
                        logger.warning(f"Quartz constant for '{quartz_key_name_or_tuple}' not found. Skipping.")
                    except Exception as e:
                        logger.warning(f"Error processing option {key} with value {value}: {e}. Skipping.")

                # --- Create CIRAWFilter ---
                if isinstance(dng_input, (str, os.PathLike)):
                    image_url = NSURL.fileURLWithPath_(str(dng_input))
                    raw_filter = CIFilter.filterWithImageURL_options_(image_url, raw_options or None)
                elif hasattr(dng_input, "read"):
                    dng_data = dng_input.read()
                    if not dng_data:
                        raise ValueError("DNG data from file-like object is empty.")
                    from Quartz import kCGImageSourceTypeIdentifierHint
                    raw_options[kCGImageSourceTypeIdentifierHint] = "com.adobe.raw-image"
                    ns_data = NSData.dataWithBytes_length_(dng_data, len(dng_data))
                    raw_filter = CIFilter.filterWithImageData_options_(ns_data, raw_options or None)
                else:
                    raise TypeError("dng_input must be a file path or a file-like object.")

                if not raw_filter:
                    raise RuntimeError("Failed to create CIRAWFilter.")

                output_ci_image = raw_filter.outputImage()

                # --- Apply Tone Curve for Contrast ---
                if contrast_strength is not None:
                    from Quartz import CIVector
                    s = min(max(float(contrast_strength), 0.0), 1.0)
                    SHADOW_PULL_FACTOR = 0.3
                    HIGHLIGHT_PUSH_FACTOR = 0.015
                    p0 = CIVector.vectorWithX_Y_(0.0, 0.0)
                    p1 = CIVector.vectorWithX_Y_(0.53, 0.53 - s * SHADOW_PULL_FACTOR)
                    p2 = CIVector.vectorWithX_Y_(0.73, 0.73)
                    p3 = CIVector.vectorWithX_Y_(0.90, 0.90 + s * HIGHLIGHT_PUSH_FACTOR)
                    p4 = CIVector.vectorWithX_Y_(1.0, 1.0)
                    tone_curve_filter = CIFilter.filterWithName_("CIToneCurve")
                    tone_curve_filter.setValue_forKey_(output_ci_image, "inputImage")
                    tone_curve_filter.setValue_forKey_(p0, "inputPoint0")
                    tone_curve_filter.setValue_forKey_(p1, "inputPoint1")
                    tone_curve_filter.setValue_forKey_(p2, "inputPoint2")
                    tone_curve_filter.setValue_forKey_(p3, "inputPoint3")
                    tone_curve_filter.setValue_forKey_(p4, "inputPoint4")
                    output_ci_image = tone_curve_filter.outputImage()

                extent = output_ci_image.extent()
                width = int(extent.size.width)
                height = int(extent.size.height)

                # --- Render to Bitmap ---
                pixel_format = kCIFormatRGBA16
                pixel_dtype = np.uint16
                row_bytes = width * 8
                bitmap_buffer = NSMutableData.dataWithLength_(height * row_bytes)

                context.context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                    output_ci_image,
                    bitmap_buffer,
                    row_bytes,
                    extent,
                    pixel_format,
                    context.output_space_cg,
                )

                rgba_image = np.frombuffer(bitmap_buffer, dtype=pixel_dtype).reshape((height, width, 4)).copy()

                return rgba_image[:, :, :3]

            except Exception as e:
                raise RuntimeError(f"An error occurred during Core Image processing: {e}") from e


