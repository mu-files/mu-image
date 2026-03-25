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
# Format: (dng_filename, tiff_filename, threshold, highlight_preserving_exposure)
# Thresholds are 1.1x above measured values
# Using LINEAR_RAW DNGs with DefaultBlackRender=1 to match Photoshop baseline
# highlight_preserving_exposure: True (default) = highlight compression, False = DNG SDK behavior
TEST_CASES = [
    # Color pattern tests (300x200, 6x4 patches)
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.none.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.none.tif", 1.07, True),  # measured 0.97%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.tif", 0.93, True),  # measured 0.84%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.temp-tint.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.temp-tint.tif", 1.17, True),  # measured 1.06%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.rcurve.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.rcurve.tif", 1.03, True),  # measured 0.94%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.gcurve.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.gcurve.tif", 1.05, True),  # measured 0.95%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.bcurve.dng", "asi676mc.linearraw.uncomp.1ifds.colorpattern.bcurve.tif", 0.89, True),  # measured 0.81%
    ("asi676mc.linearraw.uncomp.1ifds.rgbcurve.dng", "asi676mc.linearraw.uncomp.1ifds.rgbcurve.tif", 1.25, True),  # measured 1.14%

    # synthetic color patterns
    ("linear_gradient_test.exp-curve.dng", "linear_gradient_test.exp-curve.tif", 3.21, True),  # measured 2.92% - TODO: revisit large difference
    ("rgb_ramp_test.rcurve.dng", "rgb_ramp_test.rcurve.tif", 0.69, True),  # measured 0.63%
    ("rgb_ramp_test.gcurve.dng", "rgb_ramp_test.gcurve.tif", 0.74, True),  # measured 0.67%
    ("rgb_ramp_test.bcurve.dng", "rgb_ramp_test.bcurve.tif", 0.77, True),  # measured 0.70%
    ("rgb_ramp_test.bcurve2.dng", "rgb_ramp_test.bcurve2.tif", 1.05, True),  # measured 0.95%
    ("rgb_ramp_test.multi-curve.dng", "rgb_ramp_test.multi-curve.tif", 1.54, True),  # measured 1.40% - TODO: revisit large difference
    ("rgb_ramp_test.maincurve.dng", "rgb_ramp_test.maincurve.tif", 1.61, True),  # measured 1.46% - TODO: revisit large difference
    ("rgb_ramp_test.mainrgbcurve.dng", "rgb_ramp_test.mainrgbcurve.tif", 1.29, True),  # measured 1.17%

    # Full-size ASI676MC tests (4144x2822)
    ("asi676mc.linearraw.uncomp.1ifds.none.dng", "asi676mc.linearraw.uncomp.1ifds.none.tif", 1.19, True),  # measured 1.08%
    ("asi676mc.linearraw.uncomp.1ifds.exposure.dng", "asi676mc.linearraw.uncomp.1ifds.exposure.tif", 0.73, True),  # measured 0.66%
    ("asi676mc.linearraw.uncomp.1ifds.curve.dng", "asi676mc.linearraw.uncomp.1ifds.curve.tif", 1.32, True), 

    # Canon EOS R5 tests (2056x1366)
    ("canon_eos_r5.none.dng", "canon_eos_r5.none.tif", 1.25, True),  # measured 1.14%
    ("canon_eos_r5.exposure.dng", "canon_eos_r5.exposure.tif", 2.28, True),  # measured 2.07% - TODO: revisit large difference
    ("canon_eos_r5.baselineexposure-bl1.dng", "canon_eos_r5.baselineexposure-bl1.tif", 2.43, True),  # measured 2.21% - TODO: revisit large difference
    ("canon_eos_r5.temp-tint.dng", "canon_eos_r5.temp-tint.tif", 1.25, True),  # measured 1.14%

    # Lens distrotion tests
    ("lumixs9.dng", "lumixs9.tif", None, True),
    ("nikon_z_9.cfa.ljpeg.2ifds.dng", "nikon_z_9.cfa.ljpeg.2ifds.tif", None, True)

]


@pytest.mark.parametrize("dng_name,tif_name,threshold,highlight_preserving_exposure", TEST_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_xmp_rendering(dng_name, tif_name, threshold, highlight_preserving_exposure, output_dir):
    """Test XMP-based rendering against Photoshop reference.
    
    Validates that rendered output matches Photoshop reference within threshold.
    XMP NOOP filtering (Exposure2012=0, WhiteBalance="As Shot", linear tone curves)
    is handled automatically by the rendering pipeline.
    
    Args:
        highlight_preserving_exposure: If True, use highlight compression (cubic spline for negative exposure).
                         If False, use DNG SDK exposure behavior (quadratic approximation).
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
            rendering_params={'highlight_preserving_exposure': highlight_preserving_exposure},
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
    
    # Use default threshold of 1.5% if not specified
    effective_threshold = threshold if threshold is not None else 1.5
    threshold_label = f"{effective_threshold}%" if threshold is not None else f"{effective_threshold}% (default)"
    
    # Print overall and per-channel diffs
    print(f"\n  [XMP] {dng_name}: diff={stats['mean']:.2f}% (threshold={threshold_label})")
    if 'mean_R' in stats:
        print(f"        Per-channel: R={stats['mean_R']:.2f}% G={stats['mean_G']:.2f}% B={stats['mean_B']:.2f}%")
    
    assert stats["mean"] < effective_threshold, (
        f"XMP rendering diff {stats['mean']:.2f}% > {effective_threshold}% for {dng_name}"
    )

