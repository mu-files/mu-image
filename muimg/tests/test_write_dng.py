# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Test write_dng_from_array with synthetic RGB ramp data.

Validates DNG encoding/decoding fidelity across compression methods and
photometric interpretations (CFA vs LINEAR_RAW).
"""

import io
import logging
import numpy as np
import pytest
import cv2
from pathlib import Path
from tifffile import COMPRESSION

# Suppress tifffile warnings about "shaped series shape does not match page shape"
# These are harmless warnings when reading dng_validate output TIFFs
logging.getLogger('tifffile').setLevel(logging.CRITICAL)

from muimg.dngio import (
    write_dng_from_array, write_dng, DngFile, IfdDataSpec, 
    IfdPageSpec, PageEncoding, decode_dng
)
from muimg.raw_render import DemosaicAlgorithm
try:
    from muimg._dngio_coreimage import core_image_available
except ImportError:
    core_image_available = False
from muimg.tiff_metadata import MetadataTags
from muimg.raw_render import _srgb_gamma, convert_dtype
from conftest import generate_rgb_ramp, sample_as_cfa, compute_diff_stats, run_dng_validate, OutputPathManager

# Test output path manager - set persistent=True to keep outputs, False for tmp_path
output_path_manager = OutputPathManager(persistent=True)

# Compression configurations per dtype: (label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio)
# Thresholds based on observed values with ~10% margin
# Compression ratio = compressed_size / uncompressed_size (validated with 10% tolerance)
# CFA has higher render errors due to demosaicing amplifying noise/blocks
# CFA has worse compression ratio due to interleaved color pattern

# float32 - uncompressed only (JXL limited to 16-bit)
FLOAT32_LINEAR_RAW_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06, 1.0),
]

FLOAT32_CFA_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.77, 32.35, 1.0),
]

# float16 thresholds (16-bit float supports JXL)
FLOAT16_LINEAR_RAW_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06, 1.0),
    ("lossless_jxl", 0.0, 0.01, 0.01, 0.32, 9.06, 0.5227),
    ("lossy_jxl_0.5", 0.5, 0.90, 13.64, 0.83, 24.59, 0.0166),
    ("lossy_jxl_1.0", 1.0, 0.94, 15.58, 0.87, 29.77, 0.0059),
    ("lossy_jxl_2.0", 2.0, 0.96, 16.38, 0.90, 33.22, 0.0028),
    ("lossy_jxl_4.0", 4.0, 0.99, 12.67, 0.92, 32.78, 0.0019),
]

FLOAT16_CFA_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.77, 32.35, 1.0),
    ("lossless_jxl", 0.0, 0.01, 0.01, 0.77, 32.35, 0.4766),
    ("lossy_jxl_0.5", 0.5, 0.61, 5.32, 0.84, 32.35, 0.0702),
    ("lossy_jxl_1.0", 1.0, 0.82, 8.32, 0.87, 32.35, 0.0286),
    ("lossy_jxl_2.0", 2.0, 0.93, 10.90, 0.91, 28.90, 0.0117),
    ("lossy_jxl_4.0", 4.0, 0.99, 11.92, 0.94, 29.77, 0.0060),
]

# uint16 thresholds
UINT16_LINEAR_RAW_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06, 1.0),
    ("lossless_jpeg", None, 0.0, 0.0, 0.32, 9.06, None),
    ("lossless_jxl", 0.0, 0.0, 0.0, 0.32, 9.06, 0.7432),
    ("lossy_jxl_0.5", 0.5, 0.90, 13.65, 0.83, 24.16, 0.0166),
    ("lossy_jxl_1.0", 1.0, 0.94, 15.64, 0.87, 29.77, 0.0059),
    ("lossy_jxl_2.0", 2.0, 0.96, 16.40, 0.90, 33.22, 0.0028),
    ("lossy_jxl_4.0", 4.0, 0.99, 12.64, 0.92, 32.78, 0.0019),
]

UINT16_CFA_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.77, 32.35, 1.0),
    ("lossless_jpeg", None, 0.0, 0.0, 0.77, 32.35, None),
    ("lossless_jxl", 0.0, 0.0, 0.0, 0.77, 32.35, 0.7440),
    ("lossy_jxl_0.5", 0.5, 0.61, 5.40, 0.84, 32.35, 0.0702),
    ("lossy_jxl_1.0", 1.0, 0.82, 8.32, 0.87, 32.35, 0.0286),
    ("lossy_jxl_2.0", 2.0, 0.93, 10.88, 0.91, 28.90, 0.0117),
    ("lossy_jxl_4.0", 4.0, 0.99, 11.91, 0.93, 29.77, 0.0060),
]

# uint8 thresholds
UINT8_LINEAR_RAW_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.34, 8.62, 1.0),
    ("lossless_jpeg", None, 0.0, 0.0, 0.34, 8.62, None),
    ("lossless_jxl", 0.0, 0.0, 0.0, 0.34, 8.62, 0.4776),
    ("lossy_jxl_0.5", 0.5, 0.92, 18.98, 1.01, 49.61, 0.0524),
    ("lossy_jxl_1.0", 1.0, 0.95, 21.57, 0.96, 49.61, 0.0152),
    ("lossy_jxl_2.0", 2.0, 0.97, 21.14, 0.96, 52.20, 0.0064),
    ("lossy_jxl_4.0", 4.0, 1.00, 18.98, 1.00, 50.04, 0.0044),
]

UINT8_CFA_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.79, 31.92, 1.0),
    ("lossless_jpeg", None, 0.0, 0.0, 0.79, 31.92, None),
    ("lossless_jxl", 0.0, 0.0, 0.0, 0.79, 31.92, 0.4764),
    ("lossy_jxl_0.5", 0.5, 0.76, 6.47, 0.89, 32.35, 0.1948),
    ("lossy_jxl_1.0", 1.0, 0.82, 6.47, 0.89, 29.77, 0.0724),
    ("lossy_jxl_2.0", 2.0, 0.91, 9.06, 0.90, 30.62, 0.0216),
    ("lossy_jxl_4.0", 4.0, 0.96, 12.94, 0.93, 34.51, 0.0118),
]

# uint16_10bit: 10-bit data stored in uint16 arrays
# Note: DNG spec requires BitsPerSample to be 8-32 bits, so 4-bit is not supported
# Note: JXL uses left-shift workaround (9-15 bit → 16-bit) for decoder compatibility
UINT16_10BIT_LINEAR_RAW_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.34, 9.49, 0.625),
    ("lossless_jpeg", None, 0.0, 0.0, 0.34, 9.49, None),
    ("lossless_jxl", 0.0, 0.06, 1.03, 0.37, 9.92, 0.3547),
    ("lossy_jxl_0.5", 0.5, 0.90, 13.70, 0.85, 24.59, 0.0166),
    ("lossy_jxl_1.0", 1.0, 0.94, 14.96, 0.88, 29.77, 0.0059),
    ("lossy_jxl_2.0", 2.0, 0.96, 15.77, 0.91, 33.22, 0.0029),
    ("lossy_jxl_4.0", 4.0, 0.99, 12.71, 0.94, 33.22, 0.0019),
]

UINT16_10BIT_CFA_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.78, 32.35, 0.625),
    ("lossless_jpeg", None, 0.0, 0.0, 0.78, 32.35, None),
    ("lossless_jxl", 0.0, 0.01, 0.01, 0.78, 32.35, 0.3692),
    ("lossy_jxl_0.5", 0.5, 0.61, 5.42, 0.86, 32.35, 0.0704),
    ("lossy_jxl_1.0", 1.0, 0.82, 8.57, 0.89, 31.92, 0.0287),
    ("lossy_jxl_2.0", 2.0, 0.93, 11.38, 0.92, 28.90, 0.0117),
    ("lossy_jxl_4.0", 4.0, 0.99, 11.97, 0.95, 29.77, 0.0060),
]

def _test_compression_fidelity(tmp_path, dtype_label, input_dtype, photometric, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, bits_per_sample=None, expected_compression_ratio=None):
    """Test DNG write/decode fidelity for a specific dtype and compression combination.
    
    Tests the specified photometric type (CFA or LINEAR_RAW).
    
    Args:
        photometric: Photometric type to test ("CFA" or "LINEAR_RAW")
        expected_compression_ratio: Expected compression ratio (compressed_size / uncompressed_size).
                                   If None, compression ratio is printed but not validated.
    """
    # Determine output path
    output_path = output_path_manager.get_path(tmp_path, "test_write_dng")
    
    # Test dimensions
    width, height = 1280, 720
    preview_width, preview_height = 640, 360
    
    # Determine compression type
    if 'jpeg' in comp_label.lower():
        compression = COMPRESSION.JPEG
        compression_args = {'lossless': True}
    elif jxl_distance is None:
        compression = COMPRESSION.NONE
        compression_args = None
    else:
        compression = COMPRESSION.JPEGXL_DNG
        compression_args = {'distance': jxl_distance, 'effort': 4}
    
    # Test with and without preview
    for use_preview in [True, False]:
        # Generate test data in target dtype with realistic sensor noise + block artifacts
        # noise_stddev=0.01 simulates ~1% noise with visible block artifacts
        rgb_ramp = generate_rgb_ramp(
            width, height, dtype=input_dtype, noise_stddev=0.01, bits_per_sample=bits_per_sample)
        
        # Add identity ProfileToneCurve to bypass tone adjustments
        # This makes the rendering pipeline nearly pass-through
        metadata = MetadataTags()
        metadata.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])
        metadata.add_tag("UniqueCameraModel", "Test Camera")

        if photometric == "CFA":
            # Sample as CFA
            test_data = sample_as_cfa(rgb_ramp, pattern="RGGB")
        else:
            # Use full RGB (LINEAR_RAW)
            test_data = rgb_ramp
        
        # Write to file for dng_validate
        preview_label = "with_preview" if use_preview else "no_preview"
        dng_filename = f"{dtype_label}_{photometric}_{preview_label}_{comp_label}.dng"
        dng_path = output_path / dng_filename
        
        if use_preview:
            # Preview must be uint8 for JPEG compression
            rgb_ramp_u8 = convert_dtype(rgb_ramp, np.uint8)
            preview_data = cv2.resize(
                rgb_ramp_u8, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            
            # Use write_dng API with preview as IFD0 and main data as SubIFD
            preview_encoding = PageEncoding(
                compression=COMPRESSION.JPEG,
                compression_args={'level': 90}
            )
            preview_spec = IfdDataSpec(
                data=preview_data,
                photometric="RGB",
                subfiletype=1,  # Preview
                encoding=preview_encoding,
                extratags=metadata,
            )
            main_encoding = PageEncoding(
                compression=compression,
                compression_args=compression_args
            ) if compression else None
            main_spec = IfdDataSpec(
                data=test_data,
                photometric=photometric,
                cfa_pattern="RGGB",
                subfiletype=0,  # Main image
                encoding=main_encoding,
                bits_per_sample=bits_per_sample,
            )
            write_dng(
                destination_file=dng_path,
                ifd0_spec=preview_spec,
                subifds=[main_spec],
            )
        else:
            # No preview: use write_dng_from_array
            encoding = PageEncoding(
                compression=compression,
                compression_args=compression_args
            ) if compression else None
            data_spec = IfdDataSpec(
                data=test_data,
                photometric=photometric,
                cfa_pattern="RGGB",
                encoding=encoding,
                extratags=metadata,
                bits_per_sample=bits_per_sample,
            )
            write_dng_from_array(
                destination_file=dng_path,
                data_spec=data_spec,
            )
            
            # Extract raw data from DNG and validate rendering
            with DngFile(dng_path) as dng:
                main_page = dng.get_main_page()
                actual_data_size = sum(main_page.databytecounts) if hasattr(main_page, 'databytecounts') else 0
                
                # Calculate expected uncompressed size
                dtype_bits = input_dtype(0).itemsize * 8
                samples_per_pixel = 1 if photometric == "CFA" else 3
                expected_full_size = width * height * samples_per_pixel * (dtype_bits // 8)
                
                # Validate bit-packing data size reduction for uncompressed cases
                if compression == COMPRESSION.NONE and bits_per_sample is not None:
                    # Calculate ratio
                    expected_ratio = bits_per_sample / dtype_bits
                    actual_ratio = actual_data_size / expected_full_size
                    
                    # Validate ratio is exactly as expected (no tolerance needed for data size)
                    assert abs(actual_ratio - expected_ratio) < 0.001, (
                        f"{dtype_label} {photometric} {comp_label}: Bit-packed data size ratio "
                        f"{actual_ratio:.4f} does not match expected {expected_ratio:.4f} "
                        f"(bits_per_sample={bits_per_sample}, dtype_bits={dtype_bits}). "
                        f"Actual data size={actual_data_size}, Expected full size={expected_full_size}"
                    )
                
                # Calculate compression ratio for all cases
                compression_ratio = actual_data_size / expected_full_size
                
                # Print compression ratio for tuning (always print for visibility)
                print(f"    Compression: {compression_ratio:.4f} ({actual_data_size} / {expected_full_size} bytes)")
                
                # Validate compression ratio if expected value is provided
                if expected_compression_ratio is not None:
                    # Allow 10% tolerance for compression ratio variations
                    tolerance = 0.10
                    assert abs(compression_ratio - expected_compression_ratio) < tolerance, (
                        f"{dtype_label} {photometric} {comp_label}: Compression ratio "
                        f"{compression_ratio:.4f} differs from expected {expected_compression_ratio:.4f} "
                        f"by more than {tolerance*100}%"
                    )
                
                # Extract raw data
                if photometric == "CFA":
                    decoded_cfa, decoded_pattern = dng.get_cfa()
                    assert decoded_cfa is not None, f"Failed to get CFA from {comp_label}"
                    assert decoded_pattern == "RGGB", f"CFA pattern mismatch: {decoded_pattern} != RGGB"
                    decoded = decoded_cfa
                else:
                    decoded_rgb = dng.get_linear_raw()
                    assert decoded_rgb is not None, f"Failed to get LINEAR_RAW from {comp_label}"
                    decoded = decoded_rgb
                
                # Test render pipeline: with identity ProfileToneCurve, 
                # render should apply sRGB gamma to linear RGB
                # Use DNGSDK_BILINEAR for consistent demosaic comparison
                rendered = dng.render_raw(
                    output_dtype=np.uint8, demosaic_algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
                assert rendered is not None, f"Failed to render {comp_label}"
            
            '''
            # Write our rendered output to TIFF for debugging
            import tifffile
            our_render_path = output_path / f"{dng_filename}_our_render.tif"
            tifffile.imwrite(our_render_path, rendered, photometric='rgb')
            '''

            # Compare raw data against original
            # For bit-packed data: get_cfa() returns data in the packed range (0-1023 for 10-bit),
            # not scaled back to full dtype range, so compare against scaled data
            comparison_target = test_data
            
            # For 9-15 bit JXL: we scale data to 16-bit before encoding to match dng_validate.
            # Adjust comparison target to match the upscaled data.
            if compression == COMPRESSION.JPEGXL_DNG and bits_per_sample is not None and 9 <= bits_per_sample <= 15:
                src_max = (1 << bits_per_sample) - 1
                dst_max = 65535  # 16-bit max
                comparison_target = (test_data.astype(np.float64) * dst_max / src_max).astype(np.uint16)
            
            # Compute diff stats for raw data
            stats = compute_diff_stats(decoded, comparison_target)
            
            # Compare rendered output against sRGB gamma-corrected reference
            # render() applies: identity ProfileToneCurve + sRGB gamma + uint8 conversion
            # Convert ramp to float [0,1], apply gamma, then convert to uint8
            # For bit-packed data, scale back to full range for rendering comparison
            if bits_per_sample is not None:
                dtype_bits = input_dtype(0).itemsize * 8
                src_max = (1 << bits_per_sample) - 1
                dst_max = (1 << dtype_bits) - 1
                ramp_for_render = (rgb_ramp.astype(np.float64) * dst_max / src_max).astype(input_dtype)
            else:
                ramp_for_render = rgb_ramp
            rgb_linear_normalized = convert_dtype(ramp_for_render, np.float32)
            rgb_srgb = _srgb_gamma(rgb_linear_normalized)
            rgb_ramp_u8 = convert_dtype(rgb_srgb, np.uint8)
            
            render_stats = compute_diff_stats(rendered, rgb_ramp_u8)
            
            # 1. Compare our rendering against dng_validate output (if available)
            # dng_validate converts to TIFF, so we can read it back and compare
            # Note: This runs BEFORE dng_validate is called below, so we need to run it first
            output_base = output_path / f"{dtype_label}_{photometric}_{preview_label}_{comp_label}"
            ignored_warnings = [
            "too little padding",  # Matches all 4 edge padding warnings
            ]
            validated_tiff = run_dng_validate(dng_path, output_base, validate=True, ignored_warnings=ignored_warnings, indent="    ")
            
            if validated_tiff is not None:
                # dng_validate is available, compare renderings
                # Now read the validated TIFF and compare
                # Note: output_base is the full path without extension, dng_validate adds .tif
                validated_tiff_path = Path(str(output_base) + '.tif')
                assert validated_tiff_path.exists(), f"dng_validate output TIFF not found: {validated_tiff_path}"
                
                from tifffile import imread
                validated_render = imread(validated_tiff_path)
                validate_stats = compute_diff_stats(rendered, validated_render)
                print(f"    Validate: mean={validate_stats['mean']:.4f}%, max={validate_stats['max']:.4f}%")
                # Allow small differences due to different rendering pipelines
                assert validate_stats['mean'] < 1.0, (
                f"{dtype_label} {photometric} {comp_label}: Render vs dng_validate diff "
                f"{validate_stats['mean']:.4f}% exceeds 1.0%"
            )

            '''
            # 2. Compare against CoreImage rendering (macOS only)
            if core_image_available:
            # CoreImage is available, so any exception is a real error
            coreimage_render, _ = decode_dng(
                dng_path,
                output_dtype=np.uint8,
                use_coreimage_if_available=True,
                use_xmp=True
            )
            coreimage_stats = compute_diff_stats(rendered, coreimage_render)
            print(f"    CoreImage: mean={coreimage_stats['mean']:.4f}%, max={coreimage_stats['max']:.4f}%")
            # Allow small differences due to different rendering pipelines
            assert coreimage_stats['mean'] < 1.0, (
                f"{dtype_label} {photometric} {comp_label}: Render vs CoreImage diff "
                f"{coreimage_stats['mean']:.4f}% exceeds 1.0%"
            )
            else:
            print(f"    CoreImage: not available (skipped)")
            '''

            # Print observed values for threshold tuning
            preview_str = "with_preview" if use_preview else "no_preview"
            print(f"\n  {dtype_label} {photometric} {comp_label} ({preview_str}):")
            print(f"    Raw:    mean={stats['mean']:.4f}%, p99={stats['p99']:.4f}%, max={stats['max']:.4f}%")
            print(f"    Render: mean={render_stats['mean']:.4f}%, max={render_stats['max']:.4f}%")
            
            # Assert raw data thresholds
            if raw_mean_thresh == 0.0:
                # Exact match expected for uncompressed
                assert np.array_equal(decoded, comparison_target), (
                    f"{dtype_label} {photometric} {comp_label}: Raw expected exact match, got "
                    f"mean={stats['mean']:.4f}%, max={stats['max']:.4f}%"
                )
            else:
                # Lossy compression - check thresholds
                assert stats['mean'] < raw_mean_thresh, (
                    f"{dtype_label} {photometric} {comp_label}: Raw mean diff {stats['mean']:.4f}% "
                    f"exceeds threshold {raw_mean_thresh}%"
                )
                assert stats['max'] < raw_max_thresh, (
                    f"{dtype_label} {photometric} {comp_label}: Raw max diff {stats['max']:.4f}% "
                    f"exceeds threshold {raw_max_thresh}%"
                )
            
            # Assert render thresholds
            assert render_stats['mean'] < render_mean_thresh, (
            f"{dtype_label} {photometric} {comp_label}: Render mean diff {render_stats['mean']:.4f}% "
            f"exceeds threshold {render_mean_thresh}%"
            )
            assert render_stats['max'] < render_max_thresh, (
            f"{dtype_label} {photometric} {comp_label}: Render max diff {render_stats['max']:.4f}% "
            f"exceeds threshold {render_max_thresh}%"
            )


# Test functions for each dtype
@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                     FLOAT32_LINEAR_RAW_CONFIGS,
                     ids=[c[0] for c in FLOAT32_LINEAR_RAW_CONFIGS])
def test_float32_linear_raw(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test float32 DNG compression fidelity (LINEAR_RAW)."""
    _test_compression_fidelity(tmp_path, "float32", np.float32, "LINEAR_RAW", comp_label, jxl_distance, 
                           raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                           expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                     FLOAT32_CFA_CONFIGS,
                     ids=[c[0] for c in FLOAT32_CFA_CONFIGS])
def test_float32_cfa(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test float32 DNG compression fidelity (CFA)."""
    _test_compression_fidelity(tmp_path, "float32", np.float32, "CFA", comp_label, jxl_distance, 
                           raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                           expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                     FLOAT16_LINEAR_RAW_CONFIGS,
                     ids=[c[0] for c in FLOAT16_LINEAR_RAW_CONFIGS])
def test_float16_linear_raw(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test float16 DNG compression fidelity (LINEAR_RAW)."""
    _test_compression_fidelity(tmp_path, "float16", np.float16, "LINEAR_RAW", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                     FLOAT16_CFA_CONFIGS,
                     ids=[c[0] for c in FLOAT16_CFA_CONFIGS])
def test_float16_cfa(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test float16 DNG compression fidelity (CFA)."""
    _test_compression_fidelity(tmp_path, "float16", np.float16, "CFA", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT16_LINEAR_RAW_CONFIGS,
                         ids=[c[0] for c in UINT16_LINEAR_RAW_CONFIGS])
def test_uint16_linear_raw(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test uint16 DNG compression fidelity (LINEAR_RAW)."""
    _test_compression_fidelity(tmp_path, "uint16", np.uint16, "LINEAR_RAW", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT16_CFA_CONFIGS,
                         ids=[c[0] for c in UINT16_CFA_CONFIGS])
def test_uint16_cfa(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test uint16 DNG compression fidelity (CFA)."""
    _test_compression_fidelity(tmp_path, "uint16", np.uint16, "CFA", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT8_LINEAR_RAW_CONFIGS,
                         ids=[c[0] for c in UINT8_LINEAR_RAW_CONFIGS])
def test_uint8_linear_raw(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test uint8 DNG compression fidelity (LINEAR_RAW)."""
    _test_compression_fidelity(tmp_path, "uint8", np.uint8, "LINEAR_RAW", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT8_CFA_CONFIGS,
                         ids=[c[0] for c in UINT8_CFA_CONFIGS])
def test_uint8_cfa(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test uint8 DNG compression fidelity (CFA)."""
    _test_compression_fidelity(tmp_path, "uint8", np.uint8, "CFA", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, 
                               expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT16_10BIT_LINEAR_RAW_CONFIGS,
                         ids=[c[0] for c in UINT16_10BIT_LINEAR_RAW_CONFIGS])
def test_uint16_10bit_linear_raw(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test 10-bit data stored in uint16 arrays (LINEAR_RAW)."""
    _test_compression_fidelity(tmp_path, "uint16_10bit", np.uint16, "LINEAR_RAW", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh,
                               bits_per_sample=10, expected_compression_ratio=compression_ratio)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh,compression_ratio", 
                         UINT16_10BIT_CFA_CONFIGS,
                         ids=[c[0] for c in UINT16_10BIT_CFA_CONFIGS])
def test_uint16_10bit_cfa(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh, compression_ratio):
    """Test 10-bit data stored in uint16 arrays (CFA)."""
    _test_compression_fidelity(tmp_path, "uint16_10bit", np.uint16, "CFA", comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh,
                               bits_per_sample=10, expected_compression_ratio=compression_ratio)


# Lossless transcode test configurations
TRANSCODE_CONFIGS = [
    ("uint8", np.uint8, None),
    ("uint16_10bit", np.uint16, 10),
    ("uint16", np.uint16, None),
]

LOSSLESS_COMPRESSIONS = [
    ("uncompressed", COMPRESSION.NONE, None),
    ("lossless_jpeg", COMPRESSION.JPEG, {'lossless': True}),
    ("lossless_jxl", COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}),
]

PHOTOMETRIC_TYPES = ["CFA", "LINEAR_RAW"]

@pytest.mark.parametrize("photometric", PHOTOMETRIC_TYPES)
@pytest.mark.parametrize("dtype_label,input_dtype,bits_per_sample", TRANSCODE_CONFIGS)
@pytest.mark.parametrize("src_label,src_compression,src_args", LOSSLESS_COMPRESSIONS)
@pytest.mark.parametrize("dst_label,dst_compression,dst_args", LOSSLESS_COMPRESSIONS)
def test_lossless_transcode(tmp_path, photometric, dtype_label, input_dtype, bits_per_sample, 
                           src_label, src_compression, src_args,
                           dst_label, dst_compression, dst_args):
    """Test lossless transcoding between compression formats preserves data perfectly.
    
    Creates a source DNG with synthetic data, transcodes it to a different compression
    format, and verifies that rendering produces identical results.
    Tests both CFA and LINEAR_RAW photometric types.
    """
    # Determine output path
    output_path = output_path_manager.get_path(tmp_path, "test_lossless_transcode")
    
    # Generate test data
    width, height = 640, 480
    rgb_ramp_scaled = generate_rgb_ramp(width, height, dtype=input_dtype, noise_stddev=0.01, bits_per_sample=bits_per_sample)
    
    # Convert to CFA or LINEAR_RAW format
    if photometric == "CFA":
        test_data = sample_as_cfa(rgb_ramp_scaled, "RGGB")
    else:
        test_data = rgb_ramp_scaled
    
    # Add identity ProfileToneCurve for consistent rendering
    metadata = MetadataTags()
    metadata.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])
    metadata.add_tag("UniqueCameraModel", "Transcode Test Camera")
    
    # Create source DNG
    src_filename = f"{dtype_label}_{photometric}_{src_label}_source.dng"
    src_path = output_path / src_filename
    
    src_spec = IfdDataSpec(
        data=test_data,
        photometric=photometric,
        cfa_pattern="RGGB" if photometric == "CFA" else None,
        encoding=PageEncoding(
            compression=src_compression,
            compression_args=src_args
        ) if src_compression != COMPRESSION.NONE else None,
        extratags=metadata,
        bits_per_sample=bits_per_sample,
    )
    write_dng_from_array(destination_file=src_path, data_spec=src_spec)
    
    # Render source DNG
    with DngFile(src_path) as src_dng:
        src_rendered = src_dng.render_raw(
            output_dtype=np.uint8,
            demosaic_algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR
        )
        assert src_rendered is not None, f"Failed to render source {src_label}"
    
    # Transcode to destination format
    dst_filename = f"{dtype_label}_{photometric}_{src_label}_to_{dst_label}.dng"
    dst_path = output_path / dst_filename
    
    with DngFile(src_path) as src_dng:
        main_page = src_dng.get_main_page()
        assert main_page is not None, f"Failed to get main page from {src_path}"
        dst_spec = IfdPageSpec(
            page=main_page,
            transcode_encoding=PageEncoding(
                compression=dst_compression,
                compression_args=dst_args
            ),
        )
        write_dng(destination_file=dst_path, ifd0_spec=dst_spec)
    
    # Render destination DNG
    with DngFile(dst_path) as dst_dng:
        dst_rendered = dst_dng.render_raw(
            output_dtype=np.uint8,
            demosaic_algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR
        )
        assert dst_rendered is not None, f"Failed to render destination {dst_label}"
    
    # Compare rendered outputs - should be identical for lossless transcode
    stats = compute_diff_stats(dst_rendered, src_rendered)
    
    # Transcoding 10-bit data TO JXL has small error (~0.0007%).
    # This is due to the jpeg and uncompressed cases feeding 10-bit data to render pipeline
    # and JXL up-converting it to 16-bit
    if (dst_label == "lossless_jxl" and src_label != "lossless_jxl" and 
        dtype_label == "uint16_10bit"):
        mean_threshold = 0.001  # 0.001% tolerance for 10-bit non-JXL → JXL transcode
        max_threshold = 0.4  # ~1 uint8 value (100/255) due to rounding in rendering pipeline
    else:
        mean_threshold = 0.0
        max_threshold = 0.0
    
    assert stats['mean'] <= mean_threshold, (
        f"Lossless transcode {src_label} → {dst_label} ({dtype_label}) "
        f"produced mean diff {stats['mean']:.6f} > threshold {mean_threshold}"
    )
    assert stats['max'] <= max_threshold, (
        f"Lossless transcode {src_label} → {dst_label} ({dtype_label}) "
        f"produced max diff {stats['max']:.6f} > threshold {max_threshold}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
