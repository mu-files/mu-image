import numpy as np
from typing import Optional

# --- Core Image (macOS specific) DNG Processing ---

# PyObjC imports - these will only work on macOS
try:
    from Foundation import NSURL, NSData, NSMutableData
    from Quartz import (
        CIFilter,
        CIContext,
        NSColorSpace,
        kCIFormatRGBA8, # 8-bit integer
        kCIFormatRGBA16, # 16-bit unsigned integer RGBA
        kCIContextOutputColorSpace,
        kCIContextWorkingColorSpace,
    )
    core_image_available = True
except ImportError:
    core_image_available = False


def process_dng_with_core_image(
    dng_file_path: str,
    icc_profile_path: str = None,
    temperature: Optional[float] = None, # e.g., 6500 for D65
    tint: Optional[float] = None,       # e.g., -150 to +150
) -> Optional[np.ndarray]:
    """
    Processes a DNG file using macOS Core Image CIRAWFilter.

    Args:
        dng_file_path: Absolute path to the DNG file.
        temperature: Optional color temperature (e.g., 2000-50000).
        tint: Optional tint adjustment (e.g., -150 to +150).

    Returns:
        A NumPy array (height, width, 3) with RGB data, or None on error.
    """
    if not core_image_available:
        print(
            "Core Image / PyObjC is not available. "
            "This function only works on macOS with PyObjC installed."
        )
        return None

    try:
        image_url = NSURL.fileURLWithPath_(dng_file_path)
        if not image_url:
            print(f"Error: Could not create NSURL for path: {dng_file_path}")
            return None

        raw_filter = CIFilter.filterWithImageURL_options_(image_url, None)

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

        # Create the CIContext with the chosen working space.
        context_options = {kCIContextWorkingColorSpace: working_space_ns}
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

