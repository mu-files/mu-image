import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional
from muraw.dng import DngFile
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
    raw_cfa_data: np.ndarray, cfa_pattern: str, output_jpeg_path: Path
) -> None:
    """Decodes a CFA data array, scales based on data type, and saves it as a JPEG image."""
    BAYER_PATTERNS_TO_CV2 = {
        "RGGB": cv2.COLOR_BAYER_RG2RGB,
        "BGGR": cv2.COLOR_BAYER_BG2RGB,
        "GRBG": cv2.COLOR_BAYER_GR2RGB,
        "GBRG": cv2.COLOR_BAYER_GB2RGB,
    }
    print(
        f"        -> CFA Data: shape={raw_cfa_data.shape}, dtype={raw_cfa_data.dtype}, Pattern='{cfa_pattern}'"
    )

    debayer_code = BAYER_PATTERNS_TO_CV2.get(cfa_pattern)
    if debayer_code is None:
        print(f"          -> Could not debayer: Unknown CFA pattern '{cfa_pattern}'.")
        # Try to save raw data scaled if it's 2D (monochrome)
        if len(raw_cfa_data.shape) == 2:
            scaled_gray_data: Optional[np.ndarray] = None  # Ensure it's defined
            if raw_cfa_data.dtype == np.uint16:
                scaled_gray_data = (raw_cfa_data // 256).astype(np.uint8)
            elif raw_cfa_data.dtype == np.uint8:
                scaled_gray_data = raw_cfa_data
            # For other dtypes, scaled_gray_data remains None in this specific scenario

            if scaled_gray_data is not None:
                cv2.imwrite(str(output_jpeg_path), scaled_gray_data)
                print(f"          -> Saved scaled grayscale raw data to {output_jpeg_path.name}")
            else:
                # This will be hit if dtype was not uint16 or uint8
                print(
                    f"          WARNING: Unsupported dtype {raw_cfa_data.dtype} for direct grayscale saving of unknown CFA pattern. Image not saved."
                )
        return  # This return is for the 'if debayer_code is None:' block

    """
    TODO: handle this code later 
    # Handle White Level
    wl = 65535.0 # Default for 16-bit data if not specified or invalid
    if white_level is not None:
        if isinstance(white_level, (list, tuple, np.ndarray)):
            # If white level is an array (e.g. per channel for some formats), use the mean for simple scaling.
            # A more sophisticated approach might scale each channel independently if needed before debayering.
            current_wl = np.mean(white_level)
            if current_wl > 0:
                wl = float(current_wl)
            print(f"          DEBUG: CFA White level is an array, using mean: {wl}")
        elif float(white_level) > 0:
            wl = float(white_level)
        else:
            print(f"          WARNING: CFA White level is not positive ({white_level}). Using default {wl}.")
    else:
        print(f"          WARNING: CFA White level is None. Using default {wl}.")

    # Scaling: (data - 0) / white_level
    scaled_data = np.clip(float_data / wl, 0.0, 1.0)
    """

    # Debayer the image (e.g., uint16 CFA -> uint16 RGB)
    color_image_full_depth = cv2.cvtColor(raw_cfa_data, debayer_code)

    # Scale to 8-bit based on original data type
    if color_image_full_depth.dtype == np.uint16:
        color_image_8bit = (color_image_full_depth // 256).astype(np.uint8)
    elif color_image_full_depth.dtype == np.uint8:
        color_image_8bit = color_image_full_depth
    else:
        # This case should ideally not happen if input raw_cfa_data is uint8 or uint16
        # and cvtColor preserves depth for these types with corresponding debayer codes.
        print(
            f"          WARNING: Debayered image has unexpected dtype {color_image_full_depth.dtype}. Attempting to normalize and scale."
        )
        try:
            min_val, max_val = np.min(color_image_full_depth), np.max(color_image_full_depth)
            if max_val > min_val:
                norm_data = (color_image_full_depth - min_val) / (max_val - min_val)
                color_image_8bit = (norm_data * 255).astype(np.uint8)
            else:
                color_image_8bit = np.zeros_like(color_image_full_depth, dtype=np.uint8)
        except Exception:
            print(
                f"          Could not convert debayered image of dtype {color_image_full_depth.dtype} to 8-bit."
            )
            return

    cv2.imwrite(str(output_jpeg_path), color_image_8bit)
    print(f"          -> Saved debayered image to {output_jpeg_path.name}")


def _decode_and_save_linear_page(raw_linear_data: np.ndarray, output_jpeg_path: Path) -> None:
    """Decodes a LinearRaw data array, scales based on data type, and saves it as a JPEG image."""
    print(
        f"        -> LinearRaw Data: shape={raw_linear_data.shape}, dtype={raw_linear_data.dtype}"
    )

    processed_image_8bit: Optional[np.ndarray] = None

    if raw_linear_data.dtype == np.uint16:
        processed_image_8bit = (raw_linear_data // 256).astype(np.uint8)
    elif raw_linear_data.dtype == np.uint8:
        processed_image_8bit = raw_linear_data

    if processed_image_8bit is None:
        print(
            f"          Failed to process LinearRaw data for {output_jpeg_path.name}. Skipping save."
        )
        return

    final_image_to_save: np.ndarray = processed_image_8bit
    # For color LinearRaw (3 channels), DNG usually stores as RGB.
    # OpenCV imwrite expects BGR for color images.
    if len(processed_image_8bit.shape) == 3 and processed_image_8bit.shape[2] == 3:
        final_image_to_save = cv2.cvtColor(processed_image_8bit, cv2.COLOR_RGB2BGR)
    elif not (
        len(processed_image_8bit.shape) == 2
        or (len(processed_image_8bit.shape) == 3 and processed_image_8bit.shape[2] == 1)
    ):
        print(
            f"        -> LinearRaw data after scaling has unexpected shape {processed_image_8bit.shape}. Attempting to save as is."
        )
        # It might be an issue, but try to save anyway if cv2 can handle it.

    cv2.imwrite(str(output_jpeg_path), final_image_to_save)
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

                    _decode_and_save_cfa_page(raw_cfa_data, cfa_pattern_str, output_jpeg_path)
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
                    _decode_and_save_linear_page(raw_linear_data, output_jpeg_path)
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


def main():
    if not TEST_FILES_DIR.is_dir():
        print(f"Error: Test files directory not found: {TEST_FILES_DIR}")
        sys.exit(1)

    dng_files = sorted(list(TEST_FILES_DIR.glob("*.dng")))

    if not dng_files:
        print(f"No DNG files found in {TEST_FILES_DIR}")
        return

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
    main()
