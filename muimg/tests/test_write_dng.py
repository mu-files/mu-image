"""Test write_dng_from_array with synthetic RGB ramp data.

Validates DNG encoding/decoding fidelity across compression methods and
photometric interpretations (CFA vs LINEAR_RAW).
"""

import io
import numpy as np
import pytest
import cv2
from pathlib import Path
from tifffile import COMPRESSION

from muimg.dngio import write_dng_from_array, DngFile
from muimg.tiff_metadata import MetadataTags
from muimg.raw_render import _srgb_gamma, convert_dtype
from conftest import generate_rgb_ramp, sample_as_cfa, compute_diff_stats, run_dng_validate


# Compression configurations per dtype: (label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)
# Thresholds based on observed values with ~10% margin
# Note: RGB (LINEAR_RAW) shows higher loss than CFA at same distance

# float32 - uncompressed only (JXL limited to 16-bit)
# Observed: Raw mean=0.0%, max=0.0%, Render mean=0.30%, max=7.06%
FLOAT32_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06),
]

# float16 thresholds (16-bit float supports JXL)
# Observed: uncompressed Raw max=0.0%, Render mean=0.30%, max=7.06%
#           lossless_jxl Raw mean=0.0062%, max=0.0977%, Render mean=0.30%, max=7.06%
#           lossy_jxl_0.5 Raw mean=0.1542%, max=0.9277%, Render mean=0.33%, max=9.80%
#           lossy_jxl_1.0 Raw mean=0.1645%, max=1.0742%, Render mean=0.33%, max=9.80%
#           lossy_jxl_2.0 Raw mean=0.1601%, max=1.3672%, Render mean=0.33%, max=9.80%
#           lossy_jxl_4.0 Raw mean=0.1509%, max=2.0020%, Render mean=0.32%, max=10.20%
FLOAT16_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06),
    ("lossless_jxl", 0.0, 0.02, 2.10, 0.32, 9.06),
    ("lossy_jxl_0.5", 0.5, 0.18, 2.44, 0.35, 11.80),
    ("lossy_jxl_1.0", 1.0, 0.19, 3.07, 0.35, 11.80),
    ("lossy_jxl_2.0", 2.0, 0.24, 3.37, 0.37, 11.80),
    ("lossy_jxl_4.0", 4.0, 0.21, 4.25, 0.37, 12.20),
]

# uint16 thresholds
# Observed: uncompressed Raw max=0.0%, Render mean=0.30%, max=7.06%
#           lossless_jxl Raw mean=0.0074%, max=0.0717%, Render mean=0.30%, max=7.06%
#           lossy_jxl_0.5 Raw mean=0.1543%, max=0.8911%, Render mean=0.33%, max=9.80%
#           lossy_jxl_1.0 Raw mean=0.1645%, max=1.0681%, Render mean=0.33%, max=9.80%
#           lossy_jxl_2.0 Raw mean=0.2122%, max=1.4878%, Render mean=0.35%, max=9.80%
#           lossy_jxl_4.0 Raw mean=0.1918%, max=2.5757%, Render mean=0.35%, max=10.20%
UINT16_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.32, 9.06),
    ("lossless_jxl", 0.0, 0.03, 2.07, 0.32, 9.06),
    ("lossy_jxl_0.5", 0.5, 0.18, 2.89, 0.35, 11.80),
    ("lossy_jxl_1.0", 1.0, 0.19, 3.07, 0.35, 11.80),
    ("lossy_jxl_2.0", 2.0, 0.24, 3.49, 0.37, 11.80),
    ("lossy_jxl_4.0", 4.0, 0.21, 4.58, 0.37, 12.20),
]

# uint8 thresholds
# Observed: uncompressed Raw max=0.0%, Render mean=0.32%, max=7.84%
#           lossless_jxl Raw mean=0.0321%, max=1.9608%, Render mean=0.32%, max=14.90%
#           lossy_jxl_0.5 Raw mean=0.1580%, max=3.5294%, Render mean=0.34%, max=16.08%
#           lossy_jxl_1.0 Raw mean=0.1885%, max=3.5294%, Render mean=0.35%, max=18.82%
#           lossy_jxl_2.0 Raw mean=0.1841%, max=5.0980%, Render mean=0.35%, max=23.53%
#           lossy_jxl_4.0 Raw mean=0.2013%, max=8.6275%, Render mean=0.41%, max=31.37%
UINT8_CONFIGS = [
    ("uncompressed", None, 0.0, 0.0, 0.34, 9.84),
    ("lossless_jxl", 0.0, 0.05, 3.96, 0.34, 16.90),
    ("lossy_jxl_0.5", 0.5, 0.18, 5.53, 0.37, 18.08),
    ("lossy_jxl_1.0", 1.0, 0.21, 5.53, 0.37, 20.82),
    ("lossy_jxl_2.0", 2.0, 0.21, 7.10, 0.37, 25.53),
    ("lossy_jxl_4.0", 4.0, 0.22, 10.63, 0.43, 33.37),
]

def _test_compression_fidelity(tmp_path, dtype_label, input_dtype, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh):
    """Test DNG write/decode fidelity for a specific dtype and compression combination.
    
    Tests both CFA (Bayer) and LINEAR_RAW (demosaiced RGB) photometric types.
    """
    
    # Test dimensions
    width, height = 1280, 720
    preview_width, preview_height = 640, 360
    
    # Determine compression settings from jxl_distance
    if jxl_distance is None:
        compression = None
        compression_args = None
    else:
        compression = COMPRESSION.JPEGXL_DNG
        compression_args = {'distance': jxl_distance, 'effort': 4}
    
    # Test both photometric types and preview configurations
    for photometric in ["CFA", "LINEAR_RAW"]:
        for use_preview in [True, False]:
            # Generate test data in target dtype
            rgb_ramp = generate_rgb_ramp(width, height, dtype=input_dtype)
            
            # Add identity ProfileToneCurve to bypass tone adjustments
            # This makes the rendering pipeline nearly pass-through
            metadata = MetadataTags()
            metadata.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])
            metadata.add_tag("UniqueCameraModel", "Test Camera")
            
            if use_preview:
                # Preview must be uint8 for JPEG compression
                rgb_ramp_u8 = convert_dtype(rgb_ramp, np.uint8)
                preview_data = cv2.resize(rgb_ramp_u8, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            else:
                preview_data = None
            
            if photometric == "CFA":
                # Sample as CFA
                test_data = sample_as_cfa(rgb_ramp, pattern="RGGB")
            else:
                # Use full RGB (LINEAR_RAW)
                test_data = rgb_ramp
            
            # Write to file for dng_validate
            preview_label = "with_preview" if use_preview else "no_preview"
            dng_filename = f"{dtype_label}_{photometric}_{preview_label}_{comp_label}.dng"
            dng_path = tmp_path / dng_filename
            
            # Build write_dng_from_array kwargs
            write_kwargs = {
                'destination_file': dng_path,
                'data': test_data,
                'ifd0_tags': metadata,
                'photometric': photometric,
                'compression': compression,
                'compression_args': compression_args,
                'preview_image': preview_data,
            }
            if photometric == "CFA":
                write_kwargs['cfa_pattern'] = "RGGB"
            
            write_dng_from_array(**write_kwargs)
            
            # Extract raw data from DNG and validate rendering
            with DngFile(dng_path) as dng:
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
                rendered = dng.render_raw(output_dtype=np.uint8, demosaic_algorithm="DNGSDK_BILINEAR")
                assert rendered is not None, f"Failed to render {comp_label}"
            
            # Compare raw data against original
            comparison_target = test_data
            
            # Compute diff stats for raw data
            stats = compute_diff_stats(decoded, comparison_target)
            
            # Compare rendered output against sRGB gamma-corrected reference
            # render() applies: identity ProfileToneCurve + sRGB gamma + uint8 conversion
            # Convert ramp to float [0,1], apply gamma, then convert to uint8
            rgb_linear_normalized = convert_dtype(rgb_ramp, np.float32)
            rgb_srgb = _srgb_gamma(rgb_linear_normalized)
            rgb_ramp_u8 = convert_dtype(rgb_srgb, np.uint8)
            
            render_stats = compute_diff_stats(rendered, rgb_ramp_u8)
            
            # Print observed values for threshold tuning
            preview_str = "with_preview" if use_preview else "no_preview"
            print(f"\n  {dtype_label} {photometric} {comp_label} ({preview_str}):")
            print(f"    Raw:    mean={stats['mean']:.4f}%, p99={stats['p99']:.4f}%, max={stats['max']:.4f}%")
            print(f"    Render: mean={render_stats['mean']:.4f}%, max={render_stats['max']:.4f}%")
            
            # Run dng_validate if available (inline with test results)
            output_base = tmp_path / f"{dtype_label}_{photometric}_{preview_label}_{comp_label}"
            ignored_warnings = [
                "too little padding",  # Matches all 4 edge padding warnings
            ]
            validated_tiff = run_dng_validate(dng_path, output_base, validate=True, ignored_warnings=ignored_warnings, indent="    ")
            assert validated_tiff is not None, f"dng_validate failed for {dng_filename}"
            
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
@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh", 
                         FLOAT32_CONFIGS,
                         ids=[c[0] for c in FLOAT32_CONFIGS])
def test_float32(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh):
    """Test float32 DNG compression fidelity."""
    _test_compression_fidelity(tmp_path, "float32", np.float32, comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh", 
                         FLOAT16_CONFIGS,
                         ids=[c[0] for c in FLOAT16_CONFIGS])
def test_float16(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh):
    """Test float16 DNG compression fidelity."""
    _test_compression_fidelity(tmp_path, "float16", np.float16, comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh", 
                         UINT16_CONFIGS,
                         ids=[c[0] for c in UINT16_CONFIGS])
def test_uint16(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh):
    """Test uint16 DNG compression fidelity."""
    _test_compression_fidelity(tmp_path, "uint16", np.uint16, comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)

@pytest.mark.parametrize("comp_label,jxl_distance,raw_mean_thresh,raw_max_thresh,render_mean_thresh,render_max_thresh", 
                         UINT8_CONFIGS,
                         ids=[c[0] for c in UINT8_CONFIGS])
def test_uint8(tmp_path, comp_label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh):
    """Test uint8 DNG compression fidelity."""
    _test_compression_fidelity(tmp_path, "uint8", np.uint8, comp_label, jxl_distance, 
                               raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
