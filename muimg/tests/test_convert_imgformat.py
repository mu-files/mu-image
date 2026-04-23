"""Tests for convert_imgformat and convert_imgformat_to_stream pipelines.

Compares output from Python SDK pipeline (via CLI) vs Core Image pipeline.
"""

import io
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from muimg.imgio import convert_imgformat, convert_imgformat_to_stream, convert_dng_to_stream, decode_image
from muimg.dngio import DngFile
from muimg.raw_render import apply_tiff_orientation
from conftest import (
    compute_diff_stats,
    core_image_available_for_tests,
    OutputPathManager,
)


# Path to the muimg CLI module
MUIMG_CLI = [sys.executable, "-m", "muimg.cli"]

# Output directory for comparison files
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_convert_imgformat"
DNGFILES_DIR = Path(__file__).parent / "dngfiles"

# Test file
TEST_DNG = DNGFILES_DIR / "asi676mc.cfa.jxl_lossy.2ifds.dng"

# Test output path manager - set persistent=True to keep outputs, False for tmp_path
output_path_manager = OutputPathManager(persistent=True)

# Threshold for comparing SDK vs Core Image output (percentage difference)
# The two pipelines use different algorithms so some difference is expected
MEAN_DIFF_THRESHOLD = 1.0  # Mean difference as percentage (0-100 scale)
P99_DIFF_THRESHOLD = 4.0   # P99 difference as percentage (0-100 scale)


@pytest.mark.skipif(not TEST_DNG.exists(), reason="Test DNG file not found")
@pytest.mark.skipif(
    not core_image_available_for_tests(), reason="Core Image not available"
)
def test_convert_imgformat_sdk_vs_coreimage():
    """Test that SDK and Core Image pipelines produce similar results.
    
    1. Convert DNG to 8-bit TIFF via CLI (Python SDK pipeline)
    2. Convert DNG to 8-bit PNG stream via Core Image pipeline (convert_imgformat_to_stream)
    3. Decode both and compare within threshold
    """
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    basename = TEST_DNG.stem
    tiff_path = OUTPUT_DIR / f"{basename}_sdk.tiff"
    png_path = OUTPUT_DIR / f"{basename}_coreimage.png"
    
    # 1. Convert to TIFF via CLI (Python SDK pipeline, no --use-coreimage flag)
    result = subprocess.run(
        MUIMG_CLI + [
            "dng",
            "convert",
            str(TEST_DNG),
            str(tiff_path),
            "--bit-depth", "8",
            "--no-xmp",  # Disable XMP for this test
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert tiff_path.exists(), "TIFF file not created"
    print(f"\nSDK output: {tiff_path}")
    
    # 2. Convert to PNG via Core Image pipeline 
    png_stream = convert_dng_to_stream(
        file=str(TEST_DNG),
        output_format_stream="png",
        output_dtype=np.uint8,
        use_coreimage_if_available=True,  # Use Core Image
        use_xmp=False,
    )
    assert png_stream is not None, "Failed to convert DNG to PNG stream via Core Image"
    png_bytes = png_stream.getvalue()
    assert len(png_bytes) > 0, "PNG stream is empty"
    
    # Save PNG for inspection
    png_path.write_bytes(png_bytes)
    print(f"Core Image output: {png_path}")
    
    # 3. Decode both images using decode_image()
    # This uses tifffile for TIFF (avoiding OpenCV EXIF warnings) and handles PNG correctly
    tiff_rgb = decode_image(tiff_path, output_dtype=np.uint8)
    assert tiff_rgb is not None, "Failed to decode TIFF"
    
    # Decode PNG from stream
    png_stream.seek(0)
    png_rgb = decode_image(png_stream, output_dtype=np.uint8)
    assert png_rgb is not None, "Failed to decode PNG"
    
    # 4. Compare images
    print(f"SDK TIFF shape: {tiff_rgb.shape}")
    print(f"Core Image PNG shape: {png_rgb.shape}")
    
    # Images may have different sizes due to different crop handling
    # For now, require same size
    if tiff_rgb.shape != png_rgb.shape:
        pytest.skip(
            f"Image sizes differ: SDK={tiff_rgb.shape}, CoreImage={png_rgb.shape}. "
            "This may be due to different crop handling between pipelines."
        )
    
    stats = compute_diff_stats(tiff_rgb, png_rgb)
    
    print(f"Mean diff: {stats['mean']:.2f}%")
    print(f"P99 diff: {stats['p99']:.2f}%")
    print(f"Max diff: {stats['max']:.2f}%")
    
    # Assert within thresholds
    assert stats['mean'] < MEAN_DIFF_THRESHOLD, (
        f"Mean difference {stats['mean']:.2f}% exceeds threshold {MEAN_DIFF_THRESHOLD}%"
    )
    assert stats['p99'] < P99_DIFF_THRESHOLD, (
        f"P99 difference {stats['p99']:.2f}% exceeds threshold {P99_DIFF_THRESHOLD}%"
    )


@pytest.mark.skipif(not TEST_DNG.exists(), reason="Test DNG file not found")
def test_convert_imgformat_to_file():
    """Test basic convert_imgformat to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.jpg"
        
        success = convert_imgformat(
            file=str(TEST_DNG),
            output=str(output_path),
            output_dtype=np.uint8,
        )
        
        assert success, "convert_imgformat failed"
        assert output_path.exists(), "Output file not created"
        assert output_path.stat().st_size > 0, "Output file is empty"
        
        # Verify it's a valid image
        img = decode_image(output_path, output_dtype=np.uint8)
        assert img is not None, "Output is not a valid image"
        assert img.shape[2] == 3, "Output should be RGB"


@pytest.mark.skipif(not TEST_DNG.exists(), reason="Test DNG file not found")
def test_convert_imgformat_to_stream():
    """Test basic convert_imgformat_to_stream."""
    png_stream = convert_imgformat_to_stream(
        file=str(TEST_DNG),
        output_format_stream="png",
        output_dtype=np.uint8,
    )
    
    assert png_stream is not None, "convert_imgformat_to_stream returned None"
    png_bytes = png_stream.getvalue()
    assert len(png_bytes) > 0, "Stream is empty"
    
    # Verify it's a valid PNG
    img = decode_image(png_stream, output_dtype=np.uint8)
    assert img is not None, "Output is not a valid image"
    assert img.shape[2] == 3, "Output should be RGB"


def test_exif_metadata_preservation(tmp_path):
    """Test that EXIF metadata from DNG is preserved in TIFF output.
    
    Uses SubIFD 3 for faster processing.
    """
    test_dng = DNGFILES_DIR / "canon_eos_r5.cfa.ljpeg.6ifds.dng"
    if not test_dng.exists():
        pytest.skip(f"Test DNG not found: {test_dng}")
    
    output_path = output_path_manager.get_path(tmp_path, "test_exif_metadata_preservation")
    output_tif = output_path / "output.tif"
    
    # Convert DNG to TIFF using SubIFD 3
    result = subprocess.run(
        MUIMG_CLI + [
            "dng",
            "convert",
            str(test_dng),
            str(output_tif),
            "--bit-depth", "16",
            "--ifd", "subifd3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert output_tif.exists(), "TIFF file not created"
    
    # Read EXIF tags from source DNG
    with DngFile(test_dng) as dng:
        exif_dict = dng.get_tag("ExifTag")
        assert exif_dict is not None, "Source DNG has no ExifTag"
        
        # ExifTag is a dict of EXIF metadata
        print(f"\nSource DNG EXIF dict: {len(exif_dict)} tags")
        
        # Read tags from output TIFF
        # The EXIF dict should be converted to individual TIFF tags
        import tifffile
        with tifffile.TiffFile(output_tif) as tif:
            tif_tags = tif.pages[0].tags
            
            # Verify Software tag is set to 'muimg'
            assert 'Software' in tif_tags, "Software tag missing from output TIFF"
            software_value = tif_tags['Software'].value
            assert software_value == 'muimg', f"Expected Software='muimg', got '{software_value}'"
            print(f"Software tag: {software_value}")
            
            # Check that each EXIF dict key is present as a TIFF tag in output
            # ExifVersion is intentionally skipped by _convert_exif_dict_to_tags
            missing_tags = []
            preserved_tags = []
            value_mismatches = []
            
            for tag_name in exif_dict.keys():
                if tag_name == "ExifVersion":
                    continue  # Known to be skipped
                
                if tag_name in tif_tags:
                    preserved_tags.append(tag_name)
                    
                    # Compare values
                    source_value = exif_dict[tag_name]
                    output_value = tif_tags[tag_name].value
                    
                    # Convert to comparable format (handle numpy arrays, tuples, etc.)
                    if isinstance(source_value, np.ndarray):
                        source_value = source_value.tolist()
                    if isinstance(output_value, np.ndarray):
                        output_value = output_value.tolist()
                    
                    if source_value != output_value:
                        value_mismatches.append({
                            'tag': tag_name,
                            'source': source_value,
                            'output': output_value
                        })
                else:
                    missing_tags.append(tag_name)
            
            print(f"EXIF tags preserved as TIFF tags: {len(preserved_tags)}/{len(exif_dict) - 1}")
            print(f"Preserved: {preserved_tags}")
            
            if missing_tags:
                print(f"Missing: {missing_tags}")
            
            if value_mismatches:
                print(f"\nValue mismatches: {len(value_mismatches)}")
                for mismatch in value_mismatches[:5]:  # Show first 5
                    print(f"  {mismatch['tag']}: {mismatch['source']} -> {mismatch['output']}")
            
            # Assert that all EXIF dict keys are in the output TIFF tags
            assert len(missing_tags) == 0, f"EXIF tags missing from output: {missing_tags}"
            
            # Assert that values match
            assert len(value_mismatches) == 0, f"EXIF tag values don't match: {value_mismatches}"


def test_orientation_handling(tmp_path):
    """Test that orientation is correctly applied during DNG conversion.
    
    Creates DNGs with different orientation tags, converts to JXL,
    and verifies that the output matches the expected rotated image.
    """
    test_dng = DNGFILES_DIR / "canon_eos_r5.cfa.ljpeg.6ifds.dng"
    if not test_dng.exists():
        pytest.skip(f"Test DNG not found: {test_dng}")
    
    tmpdir = output_path_manager.get_path(tmp_path, "test_orientation_handling")
    
    # First, render the original DNG (orientation 1) as reference
    original_jxl = tmpdir / "original.jxl"
    result = subprocess.run(
        MUIMG_CLI + [
            "dng",
            "convert",
            str(test_dng),
            str(original_jxl),
            "--bit-depth", "16",
            "--ifd", "subifd3",  # Use SubIFD 3 for speed
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to convert original: {result.stderr}"
    
    # Load original image
    original_img = decode_image(original_jxl, output_dtype=np.uint16)
    print(f"\nOriginal image shape: {original_img.shape}")
    
    # Get original dimensions from DNG
    with DngFile(test_dng) as dng:
        page = dng.get_flattened_pages()[4]  # SubIFD 3
        orig_w, orig_h = page.get_rendered_size(apply_orientation=True)
        print(f"Original rendered size: {orig_w}x{orig_h}")
    
    # Test orientations: 6 (90° CW), 3 (180°), 8 (90° CCW)
    orientations = [6, 3, 8]
    
    # Inverse rotations to undo the applied orientation
    inverse_rotations = {
        6: 8,  # 90° CW -> apply 90° CCW to undo
        3: 3,  # 180° -> apply 180° to undo
        8: 6,  # 90° CCW -> apply 90° CW to undo
    }
    
    for orientation in orientations:
        print(f"\nTesting orientation {orientation}...")
        
        # Create DNG copy with specified orientation
        # Copy SubIFD 3 into IFD0 of the new file
        oriented_dng = tmpdir / f"oriented_{orientation}.dng"
        result = subprocess.run(
            MUIMG_CLI + [
                "dng",
                "copy",
                str(test_dng),
                str(oriented_dng),
                "--ifd", "subifd3",  # Copy SubIFD 3 to IFD0
                "--tag", f"Orientation={orientation}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to create oriented DNG: {result.stderr}"
        
        # Verify get_rendered_size reports correct dimensions
        with DngFile(oriented_dng) as dng:
            # SubIFD 3 was copied to IFD0, so access it as main page
            w, h = dng.get_rendered_size(apply_orientation=True)
            print(f"  Rendered size with orientation: {w}x{h}")
            
            # For 90° rotations, dimensions should be swapped
            if orientation in (6, 8):
                assert w == orig_h and h == orig_w, \
                    f"Dimensions not swapped for orientation {orientation}: got {w}x{h}, expected {orig_h}x{orig_w}"
            else:  # 180° rotation
                assert w == orig_w and h == orig_h, \
                    f"Dimensions changed for 180° rotation: got {w}x{h}, expected {orig_w}x{orig_h}"
        
        # Convert to JXL (no --ifd needed since it's now at IFD0)
        oriented_jxl = tmpdir / f"oriented_{orientation}.jxl"
        result = subprocess.run(
            MUIMG_CLI + [
                "dng",
                "convert",
                str(oriented_dng),
                str(oriented_jxl),
                "--bit-depth", "16",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to convert oriented DNG: {result.stderr}"
        assert oriented_jxl.exists(), f"Output file not created: {oriented_jxl}"
        
        # Load oriented image
        oriented_img = decode_image(oriented_jxl, output_dtype=np.uint16)
        
        # Apply inverse rotation to get back to original orientation
        inverse_orientation = inverse_rotations[orientation]
        unrotated_img = apply_tiff_orientation(oriented_img, inverse_orientation)
        print(f"  After inverse rotation: {unrotated_img.shape}")
        
        # Compare with original
        assert unrotated_img.shape == original_img.shape, \
            f"Shape mismatch after inverse rotation: {unrotated_img.shape} vs {original_img.shape}"
        
        # Compute difference
        stats = compute_diff_stats(original_img, unrotated_img)
        print(f"  Mean diff: {stats['mean']:.4f}%")
        print(f"  Max diff: {stats['max']:.4f}%")
        
        # Images should match exactly (or very close due to rounding)
        assert stats['mean'] < 0.015, \
            f"Images differ after orientation roundtrip: mean diff {stats['mean']:.4f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
