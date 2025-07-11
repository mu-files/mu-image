import sys
import os
import pytest
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict
from muimg.dng import DngFile
from muimg.color import process_cfa_raw, process_linear_raw
from muimg.color_mac import core_image_available, CoreImageContext
from tifffile import TIFF

# Corrected path to test files, relative to this test script
TEST_FILES_DIR = Path(__file__).parent / "rawfiles"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"


def generate_dng_structure_report(
    dng_file: DngFile, dng_file_path_for_naming: Path, output_dir: Path
) -> None:
    """Reads DNG structure and writes detailed info to a text file."""
    output_txt_path = output_dir / f"{dng_file_path_for_naming.stem}_structure_report.txt"

    dng_subfiletype_map = {0: "main", 1: "preview"}  # NewSubFileType

    with open(output_txt_path, "w") as f_out:  # dng_file is now passed as an argument
        f_out.write(f"--- Structure Report for: {dng_file_path_for_naming.name} ---\n\n")
        try:
            # The DngFile object (tif) is now passed as 'dng_file'
            tif = dng_file
            f_out.write(f"Total top-level pages found: {len(tif.pages)}\n\n")

            raw_pages_info = tif.get_raw_pages_info()
            f_out.write(
                f"Found {len(raw_pages_info)} raw page(s) with CFA or LinearRaw interpretation:\n"
            )
            for idx, (page_id_val, photometric_name, shape_tuple) in enumerate(raw_pages_info):
                f_out.write(
                    f"  - Raw Page (list index {idx}, page_id {page_id_val}): photometric='{photometric_name}', shape={shape_tuple}\n"
                )
            f_out.write("\n--- Detailed Page Inspection ---\n")

            for i, page in enumerate(tif.pages):
                page_array_str = "N/A"
                try:
                    page_array = page.asarray()
                    if page_array is not None:
                        page_array_str = str(page_array.shape)
                except Exception as e_array:
                    page_array_str = f"Error getting array: {e_array}"

                subfiletype_str = (
                    page.subfiletype.name if page.subfiletype is not None else "unknown"
                )
                photometric_str = (
                    page.photometric.name if page.photometric is not None else "unknown"
                )
                newsubfiletype_tag = page.tags.get(254)  # NewSubFileType ID is 254
                dng_type_val = newsubfiletype_tag.value if newsubfiletype_tag is not None else -1
                dng_type_str = dng_subfiletype_map.get(dng_type_val, "unknown DNG type")

                f_out.write(
                    f"\nTop-Level Page {i}: Shape={page_array_str}, TIFF_type='{subfiletype_str}', DNG_type='{dng_type_str}', Photometric='{photometric_str}'\n"
                )
                f_out.write(f"  Tags for Top-Level Page {i}:\n")
                for tag_id, tag_obj in page.tags.items():
                    tag_name_str = TIFF.TAGS.get(tag_id, str(tag_id))
                    value_type_str = type(tag_obj.value).__name__
                    value_str = str(tag_obj.value)
                    if len(value_str) > 128:
                        value_str = f"<value is {len(value_str)} chars, truncated>"
                    f_out.write(
                        f"    {tag_name_str} ({tag_id}) [type: {value_type_str}]: {value_str}\n"
                    )

                if page.pages:
                    f_out.write(
                        f"  -> Found nested 'pages' attribute with {len(page.pages)} sub-pages.\n"
                    )
                    for j, nested_page in enumerate(page.pages):
                        nested_array_str = "N/A"
                        try:
                            nested_array = nested_page.asarray()
                            if nested_array is not None:
                                nested_array_str = str(nested_array.shape)
                        except Exception as e_nested_array:
                            nested_array_str = f"Error getting array: {e_nested_array}"

                        nested_subfiletype_str = (
                            nested_page.subfiletype.name
                            if nested_page.subfiletype is not None
                            else "unknown"
                        )
                        nested_photometric_str = (
                            nested_page.photometric.name
                            if nested_page.photometric is not None
                            else "unknown"
                        )
                        nested_newsubfiletype_tag = nested_page.tags.get(254)
                        nested_dng_type_val = (
                            nested_newsubfiletype_tag.value
                            if nested_newsubfiletype_tag is not None
                            else -1
                        )
                        nested_dng_type_str = dng_subfiletype_map.get(
                            nested_dng_type_val, "unknown DNG type"
                        )
                        f_out.write(
                            f"    -> Nested Page {j}: Shape={nested_array_str}, TIFF_type='{nested_subfiletype_str}', DNG_type='{nested_dng_type_str}', Photometric='{nested_photometric_str}'\n"
                        )
                        f_out.write(f"      Tags for Nested Page {j}:\n")
                        for tag_id, tag_obj in nested_page.tags.items():
                            tag_name_str = TIFF.TAGS.get(tag_id, str(tag_id))
                            value_type_str = type(tag_obj.value).__name__
                            value_str = str(tag_obj.value)
                            if len(value_str) > 128:
                                value_str = f"<value is {len(value_str)} chars, truncated>"
                            f_out.write(
                                f"        {tag_name_str} ({tag_id}) [type: {value_type_str}]: {value_str}\n"
                            )
                else:
                    f_out.write(f"  -> No nested 'pages' attribute found on Page {i}.\n")
        except Exception as e_main:
            f_out.write(
                f"\n!!! Error processing DNG file {dng_file_path_for_naming.name} for report: {e_main} !!!\n"
            )
            import traceback

            traceback.print_exc(file=f_out)

        f_out.write(f"\n--- End of Report for: {dng_file_path_for_naming.name} ---\n")
    # This print goes to console, not the file, to indicate completion
    print(f"    Structure report generated: {output_txt_path.name}")


def _decode_and_save_cfa_page(
    raw_cfa_data: np.ndarray, tags: Dict, output_jpeg_path: Path
) -> None:
    """Decodes a CFA data array using process_cfa_raw, scales, and saves as JPEG."""

    # Extract CFA pattern from tags for logging, process_cfa_raw will do its own extraction and validation
    cfa_pattern_item = tags.get('CFAPattern') # Item might be a string or a TiffTag-like object
    if cfa_pattern_item is None:
        cfa_pattern_str = "Not Found in tags"
    elif isinstance(cfa_pattern_item, str):
        cfa_pattern_str = cfa_pattern_item  # Directly use the string if that's what we got
    else:
        cfa_pattern_str = f"Unexpected format ({type(cfa_pattern_item).__name__})"
    
    print(
        f"        -> CFA Data: shape={raw_cfa_data.shape}, dtype={raw_cfa_data.dtype}, Pattern='{cfa_pattern_str}' (from tags for logging)"
    )

    try:
        # Use process_cfa_raw from muimg.color module
        color_image_full_depth = process_cfa_raw(raw_cfa_data, tags)
    except Exception as e:
        print(f"          ERROR: Failed to process CFA data using process_cfa_raw: {e}")
        # Print error and return to prevent crash during test report generation.
        return

    # Scale to 8-bit based on original data type
    if color_image_full_depth.dtype == np.uint16:
        color_image_8bit = (color_image_full_depth // 256).astype(np.uint8)
    elif color_image_full_depth.dtype == np.uint8:
        color_image_8bit = color_image_full_depth

    cv2.imwrite(str(output_jpeg_path), color_image_8bit)
    print(f"          -> Saved debayered image to {output_jpeg_path.name}")


def _decode_and_save_linear_page(raw_linear_data: np.ndarray, tags: Dict, output_jpeg_path: Path) -> None:
    """Decodes a LinearRaw data array, scales based on data type, and saves it as a JPEG image."""
    print(
        f"        -> LinearRaw Data: shape={raw_linear_data.shape}, dtype={raw_linear_data.dtype}"
    )

    color_image_full_depth = process_linear_raw(raw_linear_data, tags)

    # Scale to 8-bit based on original data type
    if color_image_full_depth.dtype == np.uint16:
        color_image_8bit = (color_image_full_depth // 256).astype(np.uint8)
    elif color_image_full_depth.dtype == np.uint8:
        color_image_8bit = color_image_full_depth

    cv2.imwrite(str(output_jpeg_path), color_image_8bit)

    print(f"          -> Saved LinearRaw image to {output_jpeg_path.name}")


def decode_and_save_dng_images(
    dng_file: DngFile, dng_file_path_for_naming: Path, output_dir: Path
) -> None:
    """Decodes DNG raw pages and saves them as JPEG images by calling helper functions."""
    try:
        raw_pages_info = dng_file.get_raw_pages_info()
        print(
            f"  Image Generation: Found {len(raw_pages_info)} raw page(s) with CFA or LinearRaw interpretation."
        )

        if not raw_pages_info:
            print("    Image Generation: No CFA or LinearRaw pages found for processing.")
            return

        for page_id, shape, tags in raw_pages_info:
            photo_interp = tags.get("PhotometricInterpretation")

            # Construct a unique name for each page if there are multiple raw pages
            base_name = dng_file_path_for_naming.stem
            if len(raw_pages_info) > 1:
                output_jpeg_name = f"{base_name}_page{page_id}.jpg"
            else:
                output_jpeg_name = f"{base_name}.jpg"
            output_jpeg_path = output_dir / output_jpeg_name

            print(
                f"    Image Generation: Processing Raw Page ID {page_id} - photometric='{photo_interp}', shape={shape}, output='{output_jpeg_path.name}'"
            )

            if photo_interp == "CFA":
                cfa_pattern_str = tags.get("CFAPattern")

                print(
                    f"      Attempting to get CFA data for page ID {page_id}... (Pattern from tags: {cfa_pattern_str})"
                )
                # get_raw_cfa_by_id now primarily returns the data, pattern is from tags
                raw_cfa_data = dng_file.get_raw_cfa_by_id(page_id)
                if raw_cfa_data is not None:

                    _decode_and_save_cfa_page(raw_cfa_data, tags, output_jpeg_path)
                else:
                    error_suffix = (
                        f": {error_msg_cfa}"
                        if isinstance(error_msg_cfa, str) and error_msg_cfa
                        else ""
                    )
                    print(f"        -> Failed to get CFA data for page ID {page_id}{error_suffix}")
            elif photo_interp == "LINEAR_RAW":
                print(f"      Attempting to get LinearRaw data for page ID {page_id}...")
                raw_linear_data = dng_file.get_raw_linear_by_id(page_id)
                if raw_linear_data is not None:

                    _decode_and_save_linear_page(raw_linear_data, tags, output_jpeg_path)
                else:
                    print(
                        f"        -> Failed to get LinearRaw data for page ID {page_id} (or page is not LINEAR_RAW)."
                    )
            else:
                print(
                    f"      Image Generation: Skipping page with unhandled photometric interpretation: {photo_interp}"
                )
    except Exception as e:
        print(
            f"  Error during image decoding/saving for {dng_file_path_for_naming.name} (within decode_and_save_dng_images): {e}"
        )
        import traceback

        traceback.print_exc()  # Add more detailed traceback


@pytest.mark.skipif(not core_image_available, reason="Core Image not available on this system.")
def test_core_image_processing():
    """Tests Core Image DNG processing, including white balance."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    dng_files_to_test = [f for f in TEST_FILES_DIR.glob("*.dng")]

    if not dng_files_to_test:
        pytest.skip("No DNG files found in the test directory.")

    print(f"\n--- Testing Core Image DNG Processing ---")
    print(f"Found {len(dng_files_to_test)} DNG file(s) to process.")

    with CoreImageContext() as context:
        for dng_file_path in dng_files_to_test:
            print(f"\nProcessing file: {dng_file_path.name}")
            output_jpg_path_default = (
                TEST_OUTPUT_DIR / f"{dng_file_path.stem}_core_image_default.jpg"
            )

            # Test 1: Process with default settings (as-shot white balance)
            try:
                output_image = context.process_dng(str(dng_file_path))
                if output_image is not None:
                    # Convert to 8-bit if it's 16-bit
                    if output_image.dtype == np.uint16:
                        output_image = (output_image / 256).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV
                    bgr_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_jpg_path_default), bgr_image)
                    print(f"  -> Saved default processed image to {output_jpg_path_default.name}")
                else:
                    print("  -> Core Image processing with default settings returned None.")
            except Exception as e:
                print(f"  -> Error during default processing: {e}")

            # Test 2: Process with custom neutral white balance (temp/tint)
            # This is a common operation, so it's a good test case.
            output_jpg_path_wb = (
                TEST_OUTPUT_DIR / f"{dng_file_path.stem}_core_image_wb_5500k.jpg"
            )
            wb_options = {"neutralTemperature": 5500, "neutralTint": 0}
            try:
                output_image_wb = context.process_dng(
                    str(dng_file_path), raw_filter_options=wb_options
                )
                if output_image_wb is not None:
                    if output_image_wb.dtype == np.uint16:
                        output_image_wb = (output_image_wb / 256).astype(np.uint8)
                    bgr_image_wb = cv2.cvtColor(output_image_wb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_jpg_path_wb), bgr_image_wb)
                    print(f"  -> Saved custom WB processed image to {output_jpg_path_wb.name}")
                else:
                    print("  -> Core Image processing with custom WB returned None.")
            except Exception as e:
                print(f"  -> Error during custom WB processing: {e}")

        # Test 3: Process a file that doesn't exist to check for graceful handling
        print("\n--- Testing non-existent file ---")
        non_existent_file = TEST_FILES_DIR / "non_existent_file.dng"
        try:
            output_image = context.process_dng(str(non_existent_file))
            assert output_image is None, "Processing a non-existent file should return None."
            print("  -> Correctly handled non-existent file (returned None).")
        except Exception as e:
            pytest.fail(f"Processing a non-existent file raised an unexpected exception: {e}")

        # Test 4: Process with no options specified (should work like default)
        print("\n--- Testing with raw_filter_options=None ---")
        dng_file_path = dng_files_to_test[0] # Use the first file for this quick test
        try:
            output_image = context.process_dng(str(dng_file_path), raw_filter_options=None)
            assert output_image is not None, "Processing with options=None should succeed."
            print("  -> Correctly handled raw_filter_options=None.")
        except Exception as e:
            pytest.fail(f"Processing with raw_filter_options=None raised an exception: {e}")


def test_cpu_rendering_flag(capsys):
    """Verify that the use_gpu flag correctly toggles CPU/GPU rendering mode for different DNGs."""
    if not core_image_available:
        pytest.skip("Core Image not available on this system.")

    target_dng_path = TEST_FILES_DIR / "iphone.linearRGB.lossy.dng"
    if not target_dng_path.exists():
        pytest.skip(f"Test DNG file not found: {target_dng_path}")

    # Test 1: Forcing CPU rendering
    print(f"\n--- Testing CPU Rendering for {target_dng_path.name} ---")
    with CoreImageContext(use_gpu=False) as context:
        context.process_dng(str(target_dng_path))
    captured_cpu = capsys.readouterr()
    assert "Forcing software rendering (CPU)." in captured_cpu.out
    assert "Core Image processing finished in" in captured_cpu.out
    print(f"Successfully verified CPU rendering message for {target_dng_path.name}.")

    # Test 2: Allowing GPU rendering (default)
    print(f"\n--- Testing GPU Rendering for {target_dng_path.name} ---")
    with CoreImageContext(use_gpu=True) as context:
        context.process_dng(str(target_dng_path))
    captured_gpu = capsys.readouterr()
    assert "Using hardware-accelerated rendering (GPU) where available." in captured_gpu.out
    assert "Core Image processing finished in" in captured_gpu.out
    print(f"Successfully verified GPU rendering message for {target_dng_path.name}.")


def test_process_non_raw_file_warns_user(capsys):
    """Verify that processing a non-raw file (JPEG) triggers the expected warning."""
    if not core_image_available:
        pytest.skip("Core Image not available on this system.")

    # 1. Create a dummy JPEG file
    dummy_jpeg_path = TEST_OUTPUT_DIR / "dummy_test_image.jpg"
    dummy_image_data = np.zeros((10, 10, 3), dtype=np.uint8)  # Small black image
    cv2.imwrite(str(dummy_jpeg_path), dummy_image_data)

    # 2. Process the dummy JPEG with the Core Image function
    print(f"\n--- Testing Non-Raw File Warning --- ")
    print(f"Processing {dummy_jpeg_path} to check for filter type warning...")
    with CoreImageContext() as context:
        context.process_dng(str(dummy_jpeg_path))

    # 3. Capture stdout and assert the warning is present
    captured = capsys.readouterr()
    assert "Warning: Expected a CIRAWFilter but got CIPhotoFilter" in captured.out
    print("Successfully verified warning for non-raw file type.")

    # 4. Clean up the dummy file
    dummy_jpeg_path.unlink()


if __name__ == "__main__":
    # First, run the original comprehensive test function
    print("\n--- Running Core Image Processing Tests ---")
    test_core_image_processing()
    # test_process_dng() # This was the other commented out original call

    # Then, add the specific iPhone DNG GPU/CPU test
    if core_image_available:
        print("\n\n--- Specific Test: iPhone DNG GPU vs CPU ---")
        iphone_dng_path = TEST_FILES_DIR / "iphone.linearRGB.lossy.dng"
        if iphone_dng_path.exists():
            print("\n--- Processing iPhone DNG with GPU ---")
            with CoreImageContext(use_gpu=True) as context:
                context.process_dng(str(iphone_dng_path))

            print("\n--- Processing iPhone DNG with CPU ---")
            with CoreImageContext(use_gpu=False) as context:
                context.process_dng(str(iphone_dng_path))
        else:
            print(f"Error: iPhone DNG file not found at {iphone_dng_path} for specific GPU/CPU test.")
    else:
        print("Core Image / PyObjC is not available. Cannot run specific iPhone GPU/CPU test.")

    print("\n\n--- All Direct Script Tests Finished ---")
