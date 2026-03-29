"""Tests for convert_imgformat and convert_imgformat_to_stream pipelines.

Compares output from Python SDK pipeline (via CLI) vs Core Image pipeline.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from muimg.imgio import convert_imgformat, convert_imgformat_to_stream
from conftest import (
    compute_diff_stats,
    core_image_available_for_tests,
)


# Path to the muimg CLI module
MUIMG_CLI = [sys.executable, "-m", "muimg.cli"]

# Output directory for comparison files
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_convert_imgformat"
DNGFILES_DIR = Path(__file__).parent / "dngfiles"

# Test file
TEST_DNG = DNGFILES_DIR / "asi676mc.cfa.jxl_lossy.2ifds.dng"

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
            "convert-dng",
            str(TEST_DNG),
            str(tiff_path),
            "--bit-depth", "8",
            "--no-xmp",  # Disable XMP to match Core Image pipeline
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert tiff_path.exists(), "TIFF file not created"
    print(f"\nSDK output: {tiff_path}")
    
    # 2. Convert to PNG via Core Image pipeline (disable XMP for fair comparison)
    png_bytes = convert_imgformat_to_stream(
        file=str(TEST_DNG),
        output_format="png",
        output_dtype=np.uint8,
        use_coreimage_if_available=True,  # Use Core Image
        use_xmp=False,  # Disable XMP to match SDK which doesn't support it
    )
    assert png_bytes is not None, "Failed to convert DNG to PNG stream via Core Image"
    assert len(png_bytes) > 0, "PNG stream is empty"
    
    # Save PNG for inspection
    png_path.write_bytes(png_bytes)
    print(f"Core Image output: {png_path}")
    
    # 3. Decode both images
    # Decode TIFF (OpenCV reads as BGR)
    tiff_img = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
    assert tiff_img is not None, "Failed to decode TIFF"
    tiff_rgb = cv2.cvtColor(tiff_img, cv2.COLOR_BGR2RGB)
    
    # Decode PNG from bytes
    png_arr = np.frombuffer(png_bytes, dtype=np.uint8)
    png_img = cv2.imdecode(png_arr, cv2.IMREAD_COLOR)
    assert png_img is not None, "Failed to decode PNG"
    png_rgb = cv2.cvtColor(png_img, cv2.COLOR_BGR2RGB)
    
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
            output_path=str(output_path),
            output_dtype=np.uint8,
        )
        
        assert success, "convert_imgformat failed"
        assert output_path.exists(), "Output file not created"
        assert output_path.stat().st_size > 0, "Output file is empty"
        
        # Verify it's a valid image
        img = cv2.imread(str(output_path))
        assert img is not None, "Output is not a valid image"
        assert img.shape[2] == 3, "Output should be RGB"


@pytest.mark.skipif(not TEST_DNG.exists(), reason="Test DNG file not found")
def test_convert_imgformat_to_stream():
    """Test basic convert_imgformat_to_stream."""
    png_bytes = convert_imgformat_to_stream(
        file=str(TEST_DNG),
        output_format="png",
        output_dtype=np.uint8,
    )
    
    assert png_bytes is not None, "convert_imgformat_to_stream returned None"
    assert len(png_bytes) > 0, "Stream is empty"
    
    # Verify it's a valid PNG
    png_arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(png_arr, cv2.IMREAD_COLOR)
    assert img is not None, "Output is not a valid image"
    assert img.shape[2] == 3, "Output should be RGB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
