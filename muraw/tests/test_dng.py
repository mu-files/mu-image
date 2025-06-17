import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict
from muraw.dng import DngFile
from muraw.color import process_cfa_raw, process_linear_raw
from muraw.color_mac import core_image_available, process_dng_with_core_image
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
        # Use process_cfa_raw from muraw.color module
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


def test_process_dng_with_core_image():
    """Processes all DNG files in TEST_FILES_DIR using Core Image and saves them.
    This test directly calls muraw.color_mac.process_dng_with_core_image.
    """
    if not TEST_FILES_DIR.is_dir():
        print(f"Error: Test files directory not found: {TEST_FILES_DIR}")
        return

    dng_files = sorted(list(TEST_FILES_DIR.glob("*.dng")))
    if not dng_files:
        print(f"No DNG files found in {TEST_FILES_DIR} for Core Image test.")
        return

    print(f"\n--- Running Core Image Processing Test for all DNGs ---")
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dng_file_path in dng_files:
        print(f"\n  Processing with Core Image: {dng_file_path.name}")
        if not dng_file_path.exists():
            print(f"    Error: DNG file not found at {dng_file_path}, skipping.")
            continue

        if core_image_available:
            # Directly call the function from color_mac, providing a standard daylight
            # temperature (6500K) and a neutral tint to test white balance control.
            core_image_result = process_dng_with_core_image(
                str(dng_file_path), temperature=6500.0, tint=0.0
            )
            if core_image_result is not None:
                output_filename = TEST_OUTPUT_DIR / f"{dng_file_path.stem}_core_image.tif"  # Changed to .tif
                try:
                    # Convert RGB to BGR for OpenCV's imwrite function
                    bgr_image = cv2.cvtColor(core_image_result, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_filename), bgr_image)
                    print(f"    Successfully saved Core Image output to: {output_filename}")
                except Exception as e_save:
                    print(f"    Error saving Core Image output for {dng_file_path.name}: {e_save}")
            else:
                print(f"    Core Image processing (from color_mac) failed or returned None for {dng_file_path.name}.")
        else:
            print(f"    Core Image not available on this system. Skipping {dng_file_path.name}.")
    
    print("\n--- Core Image Processing Test Finished ---")

def test_process_dng():
    if not TEST_FILES_DIR.is_dir():
        print(f"Error: Test files directory not found: {TEST_FILES_DIR}")
        sys.exit(1)

    dng_files = sorted(list(TEST_FILES_DIR.glob("*.dng")))

    if not dng_files:
        print(f"No DNG files found in {TEST_FILES_DIR}")
        return

    # --- Call the simple test for the first DNG file ---


    print("\n--- Starting Full DNG Processing Loop (existing tests) ---")

    for dng_file_path in dng_files:
        print(f"\n--- Processing file: {dng_file_path.name} ---")
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        dng_obj: Optional[DngFile] = None
        try:
            dng_obj = DngFile(dng_file_path)
        except Exception as e:
            print(f"  CRITICAL: Could not open or parse DNG file {dng_file_path.name}: {e}")
            print(f"  Skipping all processing for this file.")
            continue  # Move to the next DNG file

        # If we reach here, dng_obj was successfully created
        try:
            generate_dng_structure_report(dng_obj, dng_file_path, TEST_OUTPUT_DIR)
        except Exception as e:
            print(f"  Error generating structure report for {dng_file_path.name}: {e}")
            # We can still attempt to decode images even if the report fails

        try:
            decode_and_save_dng_images(dng_obj, dng_file_path, TEST_OUTPUT_DIR)
        except Exception as e:
            print(f"  Error calling decode_and_save_dng_images for {dng_file_path.name}: {e}")


if __name__ == "__main__":
    test_process_dng_with_core_image()
    #test_process_dng()
