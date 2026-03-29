"""Test write_dng_from_array with synthetic RGB ramp data.

Validates DNG encoding/decoding fidelity across compression methods and
photometric interpretations (CFA vs LINEAR_RAW).
"""

import io
import numpy as np
import pytest
import cv2

from muimg.dngio import write_dng_from_array, decode_dng, DngFile
from muimg.tiff_metadata import MetadataTags
from muimg.raw_render import _srgb_gamma
from conftest import generate_rgb_ramp, sample_as_cfa, compute_diff_stats


def test_write_dng_compression_fidelity():
    """Test DNG write/decode fidelity across compression methods and photometric types.
    
    Tests both CFA (Bayer) and LINEAR_RAW (demosaiced RGB) with:
    - Uncompressed
    - Lossless JXL
    - Lossy JXL at various quality levels
    """
    # Test dimensions
    width, height = 1280, 720
    preview_width, preview_height = 640, 360
    
    # Compression configurations: (label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh)
    # Thresholds based on observed values with ~10% margin
    # Note: RGB (LINEAR_RAW) shows higher loss than CFA at same distance
    compression_configs = [
        ("uncompressed", None, 0.0, 0.0, 0.34, 7.8),      # Raw: 0%, Render: mean=0.30%, max=7.06%
        ("lossless_jxl", 0.0, 0.009, 0.08, 0.34, 7.8),    # Raw: mean=0.0074%, max=0.0717%, Render: mean=0.30%, max=7.06%
        ("lossy_jxl_0.5", 0.5, 0.17, 0.98, 0.37, 10.8),   # Raw: mean=0.1543%, max=0.8911%, Render: mean=0.33%, max=9.80%
        ("lossy_jxl_1.0", 1.0, 0.18, 1.18, 0.37, 10.8),   # Raw: mean=0.1645%, max=1.0681%, Render: mean=0.33%, max=9.80%
        ("lossy_jxl_1.5", 1.5, 0.20, 1.35, 0.38, 10.8),   # Raw: mean=0.1788%, max=1.2207%, Render: mean=0.34%, max=9.80%
        ("lossy_jxl_2.0", 2.0, 0.24, 1.64, 0.39, 10.8),   # Raw: mean=0.2122%, max=1.4878%, Render: mean=0.35%, max=9.80%
        ("lossy_jxl_2.5", 2.5, 0.26, 1.73, 0.40, 10.8),   # Raw: mean=0.2297%, max=1.5656%, Render: mean=0.36%, max=9.80%
        ("lossy_jxl_3.0", 3.0, 0.27, 2.02, 0.41, 10.8),   # Raw: mean=0.2408%, max=1.8341%, Render: mean=0.37%, max=9.80%
    ]
    
    # Photometric types
    photometric_types = ["cfa", "linear_raw"]
    
    # Preview configurations
    preview_configs = [
        ("with_preview", True),
        ("no_preview", False),
    ]
    
    for preview_label, use_preview in preview_configs:
        for photometric in photometric_types:
            print(f"\n{'='*60}")
            print(f"Testing {photometric} ({preview_label})")
            print(f"{'='*60}")
            
            # Generate test data
            rgb_ramp = generate_rgb_ramp(width, height)
            
            # Add identity ProfileToneCurve to bypass tone adjustments
            # This makes the rendering pipeline nearly pass-through
            metadata = MetadataTags()
            metadata.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])
            
            if use_preview:
                # Preview must be uint8 for JPEG compression
                preview_rgb = cv2.resize(rgb_ramp, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
                preview_data = (preview_rgb / 256).astype(np.uint8)
            else:
                preview_data = None
            
            if photometric == "cfa":
                # Sample as CFA
                test_data = sample_as_cfa(rgb_ramp, pattern="RGGB")
                cfa_pattern = "RGGB"
            else:
                # Use full RGB
                test_data = rgb_ramp
                cfa_pattern = None
            
            for label, jxl_distance, raw_mean_thresh, raw_max_thresh, render_mean_thresh, render_max_thresh in compression_configs:
                print(f"\n  {label}:")
                print(f"    test_data shape: {test_data.shape}, dtype: {test_data.dtype}")
                print(f"    preview_data shape: {preview_data.shape if preview_data is not None else 'None'}, dtype: {preview_data.dtype if preview_data is not None else 'N/A'}")
                
                # Write to in-memory stream
                stream = io.BytesIO()
                write_dng_from_array(
                    destination_file=stream,
                    data=test_data,
                    ifd0_tags=metadata,
                    photometric=photometric,
                    cfa_pattern=cfa_pattern,
                    jxl_distance=jxl_distance,
                    jxl_effort=4,
                    preview_image=preview_data,
                )
                
                # Get file size
                file_size = stream.tell()
                size_kb = file_size / 1024
                print(f"    Size: {size_kb:.1f} KB")
                
                # Extract raw data from DNG and validate rendering
                stream.seek(0)
                with DngFile(stream) as dng:
                    if photometric == "cfa":
                        decoded_cfa, decoded_pattern = dng.get_cfa()
                        assert decoded_cfa is not None, f"Failed to get CFA from {label}"
                        assert decoded_pattern == cfa_pattern, f"CFA pattern mismatch: {decoded_pattern} != {cfa_pattern}"
                        decoded = decoded_cfa
                    else:
                        decoded_rgb = dng.get_linear_raw()
                        assert decoded_rgb is not None, f"Failed to get LINEAR_RAW from {label}"
                        decoded = decoded_rgb
                    
                    # Test render pipeline: with identity ProfileToneCurve, 
                    # render should apply sRGB gamma to linear RGB
                    # Use DNGSDK_BILINEAR for consistent demosaic comparison
                    rendered = dng.render(output_dtype=np.uint8, demosaic_algorithm="DNGSDK_BILINEAR")
                    assert rendered is not None, f"Failed to render {label}"
                
                # Compare raw data against original
                comparison_target = test_data
                
                # Compute diff stats for raw data
                stats = compute_diff_stats(decoded, comparison_target)
                print(f"    Raw mean diff: {stats['mean']:.4f}%")
                print(f"    Raw p99 diff: {stats['p99']:.4f}%")
                print(f"    Raw max diff: {stats['max']:.4f}%")
                
                # Compare rendered output against sRGB gamma-corrected RGB ramp
                # render() applies: identity ProfileToneCurve + sRGB gamma + uint8 conversion
                rgb_linear_normalized = rgb_ramp.astype(np.float32) / 65535.0
                rgb_srgb = _srgb_gamma(rgb_linear_normalized)
                rgb_u8 = np.clip(rgb_srgb * 255.0, 0, 255).astype(np.uint8)
                
                render_stats = compute_diff_stats(rendered, rgb_u8)
                print(f"    Render mean diff: {render_stats['mean']:.4f}%")
                print(f"    Render max diff: {render_stats['max']:.4f}%")
                
                # Assert raw data thresholds
                if raw_mean_thresh == 0.0:
                    # Exact match expected for uncompressed
                    assert np.array_equal(decoded, comparison_target), (
                        f"{photometric} {label}: Raw expected exact match, got "
                        f"mean={stats['mean']:.4f}%, max={stats['max']:.4f}%"
                    )
                else:
                    # Lossy compression - check thresholds
                    assert stats['mean'] < raw_mean_thresh, (
                        f"{photometric} {label}: Raw mean diff {stats['mean']:.4f}% "
                        f"exceeds threshold {raw_mean_thresh}%"
                    )
                    assert stats['max'] < raw_max_thresh, (
                        f"{photometric} {label}: Raw max diff {stats['max']:.4f}% "
                        f"exceeds threshold {raw_max_thresh}%"
                    )
                
                # Assert render thresholds
                assert render_stats['mean'] < render_mean_thresh, (
                    f"{photometric} {label}: Render mean diff {render_stats['mean']:.4f}% "
                    f"exceeds threshold {render_mean_thresh}%"
                )
                assert render_stats['max'] < render_max_thresh, (
                    f"{photometric} {label}: Render max diff {render_stats['max']:.4f}% "
                    f"exceeds threshold {render_max_thresh}%"
                )
    
    print(f"\n{'='*60}")
    print("✓ All compression tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
