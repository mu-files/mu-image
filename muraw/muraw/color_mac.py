import numpy as np
from typing import Optional
import time

# --- Core Image (macOS specific) DNG Processing ---

# PyObjC imports - these will only work on macOS
try:
    import Quartz
    from Foundation import NSURL, NSMutableData, NSNumber, NSValue, NSPoint 
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

def process_dng_with_core_image(
    dng_file_path: str,
    icc_profile_path: Optional[str] = None,
    raw_filter_options: Optional[dict] = None,
    use_gpu: bool = False,
) -> Optional[np.ndarray]:
    """
    Processes a DNG file using Core Image on macOS.

    Args:
        dng_file_path: Path to the DNG file.
        icc_profile_path: Optional path to an ICC profile for color management.
        raw_filter_options: Dictionary of options for CIRAWFilter.
        use_gpu: If False (default), forces software (CPU) rendering.
                 If True, uses GPU-accelerated rendering if available.

    Returns:
        A NumPy array (RGB, uint8) if successful, otherwise None. on error.
    """
    if not core_image_available:
        print(
            "Core Image / PyObjC is not available. "
            "This function only works on macOS with PyObjC installed."
        )
        return None

    try:
        start_time = time.perf_counter()
        image_url = NSURL.fileURLWithPath_(dng_file_path)
        if not image_url:
            print(f"Error: Could not create NSURL for path: {dng_file_path}")
            return None

        # --- Prepare CIRAWFilter Options ---
        options = {}
        if raw_filter_options:
            for key, value in raw_filter_options.items():
                if key not in RAW_FILTER_OPTION_MAP:
                    print(f"Warning: Unknown RAW filter option key: {key}. Skipping.")
                    continue

                map_entry = RAW_FILTER_OPTION_MAP[key]
                quartz_key_name_or_tuple, value_type = map_entry
                objc_value = None

                try:
                    if value_type == "float":
                        objc_value = NSNumber.numberWithFloat_(float(value))
                        quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                        options[quartz_key] = objc_value
                        print(f"Setting RAW option {key} ({quartz_key_name_or_tuple}) to {value}")
                    elif value_type == "bool":
                        objc_value = NSNumber.numberWithBool_(bool(value))
                        quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                        options[quartz_key] = objc_value
                        print(f"Setting RAW option {key} ({quartz_key_name_or_tuple}) to {value}")
                    elif value_type == "bool_inverted":
                        objc_value = NSNumber.numberWithBool_(not bool(value))
                        quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                        options[quartz_key] = objc_value
                        print(f"Setting RAW option {key} ({quartz_key_name_or_tuple}) to {bool(value)} (inverted to {not bool(value)} for key)")
                    elif value_type == "int":
                        objc_value = NSNumber.numberWithInt_(int(value))
                        quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                        options[quartz_key] = objc_value
                        print(f"Setting RAW option {key} ({quartz_key_name_or_tuple}) to {value}")
                    elif value_type == "point_xy_floats":
                        # Expects value to be a tuple (x, y)
                        key_x_name, key_y_name = quartz_key_name_or_tuple
                        quartz_key_x = getattr(Quartz, key_x_name)
                        quartz_key_y = getattr(Quartz, key_y_name)
                        options[quartz_key_x] = NSNumber.numberWithFloat_(float(value[0]))
                        options[quartz_key_y] = NSNumber.numberWithFloat_(float(value[1]))
                        print(f"Setting RAW option {key} (X: {key_x_name}, Y: {key_y_name}) to {value}")
                    elif value_type == "point_nsvalue":
                        # Expects value to be a tuple (x, y)
                        ns_point = NSPoint(x=float(value[0]), y=float(value[1]))
                        objc_value = NSValue.valueWithPoint_(ns_point)
                        quartz_key = getattr(Quartz, quartz_key_name_or_tuple)
                        options[quartz_key] = objc_value
                        print(f"Setting RAW option {key} ({quartz_key_name_or_tuple}) to {value}")
                    else:
                        print(f"Warning: Unsupported value type '{value_type}' for key {key}. Skipping.")
                except AttributeError:
                    print(f"Warning: Quartz constant for '{quartz_key_name_or_tuple}' not found. Option {key} cannot be set. Skipping.")
                except Exception as e:
                    print(f"Warning: Error processing option {key} with value {value}: {e}. Skipping.")

        raw_filter = CIFilter.filterWithImageURL_options_(image_url, options or None)
        if not raw_filter:
            print("Error: Failed to create CIRAWFilter. Check file path and permissions.")
            return None

        # Validate that we actually got a CIRAWFilter instance
        filter_class_name = raw_filter.className()
        if filter_class_name != "CIRAWFilterImpl":
            print(
                f"Warning: Expected a CIRAWFilter but got {filter_class_name}. "
                f"The provided file may not be a supported raw format. "
                f"Raw-specific options will likely be ignored."
            )

        # Request the output image
        output_ci_image = raw_filter.outputImage()
        extent = output_ci_image.extent()
        width = int(extent.size.width)
        height = int(extent.size.height)

        # Define the rendering format
        pixel_format = kCIFormatRGBA16
        pixel_dtype = np.uint16
        row_bytes = width * 8
        
        # Create a mutable buffer for the bitmap data
        bitmap_buffer = NSMutableData.dataWithLength_(height * row_bytes)

        # --- Create the Working and Output Color Spaces ---
        from Quartz import CGColorSpaceCreateWithICCProfile

        working_space_ns = None
        output_space_cg = None

        if icc_profile_path:
            custom_cg_space = None
            try:
                with open(icc_profile_path, "rb") as f:
                    profile_data = f.read()
                custom_cg_space = CGColorSpaceCreateWithICCProfile(profile_data)
            except IOError as e:
                print(f"Warning: Could not read profile '{icc_profile_path}': {e}")
            
            # TODO: throw error instead of fallback
            if custom_cg_space:
                working_space_ns = NSColorSpace.alloc().initWithCGColorSpace_(custom_cg_space)
                output_space_cg = custom_cg_space # Use the same CG space for output
                print(f"Successfully created color space from profile: {icc_profile_path}")
            else:
                print(f"Warning: Failed to create color space from '{icc_profile_path}'. Falling back to sRGB.")

        if not working_space_ns: # Fallback if profile path not given or loading failed
            srgb_ns_space = NSColorSpace.sRGBColorSpace()
            working_space_ns = srgb_ns_space
            output_space_cg = srgb_ns_space.CGColorSpace()
            if icc_profile_path is None:
                 print("No profile path provided. Using sRGB for working and output spaces.")

        # Create the CIContext with the chosen working space and rendering options.
        context_options = {kCIContextWorkingColorSpace: working_space_ns}
        if not use_gpu:
            from Quartz import kCIContextUseSoftwareRenderer
            context_options[kCIContextUseSoftwareRenderer] = True
            print("Forcing software rendering (CPU).")
        else:
            print("Using hardware-accelerated rendering (GPU) where available.")

        context = CIContext.contextWithOptions_(context_options)

        # Render the CIImage to the bitmap buffer.
        context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
            output_ci_image,
            bitmap_buffer,
            row_bytes,
            extent,
            pixel_format,
            output_space_cg # Use the chosen output CGColorSpace
        )

        # Convert to NumPy array and reshape. Then slice off the alpha channel to return RGB.
        rgba_image = np.frombuffer(bitmap_buffer, dtype=pixel_dtype).reshape((height, width, 4))

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Core Image processing finished in {duration:.4f} seconds.")

        return rgba_image[:, :, :3]

    except Exception as e:
        import traceback
        print(f"An error occurred during Core Image processing: {e}")
        traceback.print_exc()
        return None


def list_available_rgb_color_spaces():
    """
    Queries and prints all available RGB color spaces registered on the system.
    """
    if not core_image_available:
        print("Cannot list methods, PyObjC/Quartz not available.")
        return

    print("\n--- Available Registered RGB NSColorSpaces ---")
    # The constant NSColorModelRGB is not reliably exposed by PyObjC in this environment.
    # We will use its underlying integer value, which is 1.
    NSColorModelRGB = 1
    
    # This is the correct, modern API to get a list of all spaces for a model.
    available_spaces = NSColorSpace.availableColorSpacesWithModel_(NSColorModelRGB)
    
    if not available_spaces:
        print("No registered RGB color spaces found.")
    else:
        for space in available_spaces:
            # localizedName provides the user-friendly name.
            print(f"- {space.localizedName()}")
            
    print("--- End of List ---\n")


if __name__ == '__main__':
    # When this script is run directly, list the available color spaces.
    list_available_rgb_color_spaces()

