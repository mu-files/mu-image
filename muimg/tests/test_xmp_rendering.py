"""Tests for XMP-based rendering against Photoshop reference images.

Validates that XMP metadata (Temperature, Tint, Exposure2012, ToneCurvePV2012)
is correctly parsed and applied during rendering, comparing output against
Photoshop-rendered TIFF files.
"""
import pytest
import numpy as np
from pathlib import Path
import tifffile
import logging

import muimg
from conftest import compute_diff_stats

# Suppress tifffile logging about Photoshop TIFF metadata inconsistencies
logging.getLogger('tifffile').setLevel(logging.CRITICAL)


# Test data directory
XMP_TEST_DIR = Path(__file__).parent.parent.parent.parent / "mu-image-testdata" / "xmptestfiles"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_xmp_rendering"


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

# Test files with Photoshop reference images
# Format: (dng_filename, tiff_filename, threshold)
# Thresholds are 1.1x above measured values
# Using LINEAR_RAW DNGs with DefaultBlackRender=1 to match Photoshop baseline
TEST_CASES = [
    # Color pattern tests (300x200, 6x4 patches)
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.none.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.none.tif", 1.07),  # measured 0.97%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.tif", 0.93),  # measured 0.84%
    # Full-size tests (4144x2822)
    ("asi676mc.linearraw.uncomp.1ifds.none.dng", "asi676mc.linearraw.uncomp.1ifds.none.tif", 1.19),  # measured 1.08%
    ("asi676mc.linearraw.uncomp.1ifds.exposure.dng", "asi676mc.linearraw.uncomp.1ifds.exposure.tif", 0.73),  # measured 0.66%
]


@pytest.mark.parametrize("dng_name,tif_name,threshold", TEST_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_xmp_rendering(dng_name, tif_name, threshold, output_dir):
    """Test XMP-based rendering against Photoshop reference.
    
    Validates that rendered output matches Photoshop reference within threshold.
    XMP NOOP filtering (Exposure2012=0, WhiteBalance="As Shot", linear tone curves)
    is handled automatically by the rendering pipeline.
    """
    dng_path = XMP_TEST_DIR / dng_name
    tif_path = XMP_TEST_DIR / tif_name
    
    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")
    if not tif_path.exists():
        pytest.skip(f"Reference file not found: {tif_path}")
    
    # Render with automatic XMP extraction and NOOP filtering
    with muimg.DngFile(dng_path) as dng:
        result = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            strict=False,
            use_xmp=True,
        )
        
        assert result is not None, f"Rendering failed for {dng_name}"
    
    # Save rendered output for manual inspection
    output_name = dng_path.stem + "_muimg.tif"
    tifffile.imwrite(str(output_dir / output_name), result)
    
    # Load Photoshop reference
    ref = tifffile.imread(str(tif_path))
    
    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(
            f"Shape mismatch for {dng_name}:\n"
            f"  MUIMG:     {result.shape}\n"
            f"  Photoshop: {ref.shape}"
        )
    
    # Compare against reference
    stats = compute_diff_stats(result, ref)
    
    print(f"\n  [XMP] {dng_name}: diff={stats['mean']:.2f}% (threshold={threshold}%)")
    
    assert stats["mean"] < threshold, (
        f"XMP rendering diff {stats['mean']:.2f}% > {threshold}% for {dng_name}"
    )

