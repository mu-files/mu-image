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
from conftest import compute_diff_stats, core_image_available_for_tests

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
# Format: (file_stem, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail)
# Thresholds are 1.1x above measured values
# file_stem: base filename without .dng/.tif extension
# muimg_threshold: threshold for MUIMG/SDK pipeline (None = use default 1.5%)
# muimg_xfail: True to mark as expected failure for MUIMG pipeline
# ci_threshold: threshold for Core Image pipeline (None = use default 2.0%)
# ci_xfail: True to mark as expected failure for Core Image pipeline
TEST_CASES = [
    # Color pattern tests (300x200, 6x4 patches)
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.none", 1.07, False, 2.25, False),  # muimg: 0.97%, CI: 2.16%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure", 0.93, False, 1.38, False),  # muimg: 0.84%, CI: 1.25%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.temp-tint", 1.17, False, 2.25, False),  # muimg: 1.06%, CI: 2.14%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.rcurve", 1.03, False, 2.00, False),  # muimg: 0.94%, CI: 1.82%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.gcurve", 1.05, False, 2.25, False),  # muimg: 0.95%, CI: 2.04%
    ("asi676mc.linearraw.uncomp.1ifds.colorpattern.bcurve", 0.89, False, 2.25, False),  # muimg: 0.81%, CI: 2.10%
    ("asi676mc.linearraw.uncomp.1ifds.rgbcurve", 1.25, False, 1.30, False),  # muimg: 1.14%, CI: 1.18%

    # synthetic color patterns
    ("linear_gradient_test.exp-curve", 3.21, False, None, True),  # muimg: 2.92%, CI: 13.72% - TODO: revisit large difference
    ("rgb_ramp_test.rcurve", 0.69, False, 1.94, False),  # muimg: 0.63%, CI: 1.76%
    ("rgb_ramp_test.gcurve", 0.74, False, 2.25, False),  # muimg: 0.67%, CI: 2.13%
    ("rgb_ramp_test.bcurve", 0.77, False, 2.16, False),  # muimg: 0.70%, CI: 1.96%
    ("rgb_ramp_test.bcurve2", 1.05, False, None, True),  # muimg: 0.95%, CI: 3.57%
    ("rgb_ramp_test.multi-curve", 1.54, False, None, True),  # muimg: 1.40%, CI: 2.42% - TODO: revisit large difference
    ("rgb_ramp_test.maincurve", 1.61, False, None, True),  # muimg: 1.46%, CI: 2.40% - TODO: revisit large difference
    ("rgb_ramp_test.mainrgbcurve", 1.29, False, None, True),  # muimg: 1.17%, CI: 2.47%

    # Full-size ASI676MC tests (4144x2822)
    ("asi676mc.linearraw.uncomp.1ifds.none", 1.19, False, 1.25, False),  # muimg: 1.08%, CI: 1.14%
    ("asi676mc.linearraw.uncomp.1ifds.exposure", 0.73, False, 0.81, False),  # muimg: 0.66%, CI: 0.73%
    ("asi676mc.linearraw.uncomp.1ifds.curve", 1.32, False, 1.33, False),  # muimg: 1.17%, CI: 1.21%

    # Canon EOS R5 tests (2056x1366)
    ("canon_eos_r5.none", 1.25, False, 1.85, False),  # muimg: 1.14%, CI: 1.68%
    ("canon_eos_r5.exposure", 2.28, False, None, True),  # muimg: 2.07%, CI: 4.07% - TODO: revisit large difference
    ("canon_eos_r5.baselineexposure-bl1", 2.43, False, 2.0, False),  # muimg: 2.21%, CI: 1.88% - TODO: revisit large difference
    ("canon_eos_r5.temp-tint", 1.25, False, 1.86, False),  # muimg: 1.14%, CI: 1.69%

    # other
    ("R0000762", 1.01, False, 0.53, False),  # RICOH GR IIIx, muimg: 0.92%, CI: 0.48%
    ("P4260210", 1.98, False, 1.19, False),  # Olympus E-M10MarkIV, muimg: 1.80%, CI: 1.08%
    ("DSC_1437", None, True, None, True),  # NIKON Z 8, muimg: 2.00% (FixVignetteRadial opcode unsupported), CI: 6.06% (looks like CI does not support WarpRectilinear opcode)

    # Lens distortion tests
    ("lumixs9", 1.47, False, None, True),  # muimg: 1.34%, CI: 2.83%
    ("nikon_z_9.cfa.ljpeg.2ifds", 2.38, False, None, True),  # muimg: 2.16%, CI: 3.61%
    ("DSCF0033", None, True, None, True),  # FUJIFILM:X-T30 III, muimg: X-Trans CFA not supported, CI: 10.16%
]


@pytest.mark.parametrize("file_stem,muimg_threshold,muimg_xfail,ci_threshold,ci_xfail", TEST_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_muimg_xmp_rendering(file_stem, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail, output_dir):
    """Test MUIMG XMP-based rendering against Photoshop reference.
    
    Validates that rendered output matches Photoshop reference within threshold.
    XMP NOOP filtering (Exposure2012=0, WhiteBalance="As Shot", linear tone curves)
    is handled automatically by the rendering pipeline.
    
    Args:
        file_stem: Base filename without extension
        muimg_threshold: Threshold for MUIMG pipeline (None = 1.5% default)
        muimg_xfail: True to mark as expected failure for MUIMG
        ci_threshold: Threshold for Core Image pipeline (unused in this test)
        ci_xfail: Core Image xfail flag (unused in this test)
    """
    # Mark as xfail if specified
    if muimg_xfail:
        pytest.xfail(f"MUIMG rendering known to fail for {file_stem}")
    
    dng_path = XMP_TEST_DIR / f"{file_stem}.dng"
    tif_path = XMP_TEST_DIR / f"{file_stem}.tif"
    
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
            rendering_params={'highlight_preserving_exposure': True},
        )
        
        assert result is not None, f"Rendering failed for {file_stem}"
    
    # Save rendered output for manual inspection
    output_name = file_stem + "_muimg.tif"
    tifffile.imwrite(str(output_dir / output_name), result)
    
    # Load Photoshop reference
    ref = tifffile.imread(str(tif_path))
    
    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(
            f"Shape mismatch for {file_stem}:\n"
            f"  MUIMG:     {result.shape}\n"
            f"  Photoshop: {ref.shape}"
        )
    
    # Compare against reference
    stats = compute_diff_stats(result, ref)
    
    # Use default threshold of 1.5% if not specified
    effective_threshold = muimg_threshold if muimg_threshold is not None else 1.5
    threshold_label = f"{effective_threshold}%" if muimg_threshold is not None else f"{effective_threshold}% (default)"
    
    # Print overall and per-channel diffs
    print(f"\n  [MUIMG] {file_stem}: diff={stats['mean']:.2f}% (threshold={threshold_label})")
    if 'mean_R' in stats:
        print(f"          Per-channel: R={stats['mean_R']:.2f}% G={stats['mean_G']:.2f}% B={stats['mean_B']:.2f}%")
    
    assert stats["mean"] < effective_threshold, (
        f"MUIMG XMP rendering diff {stats['mean']:.2f}% > {effective_threshold}% for {file_stem}"
    )


@pytest.mark.skipif(
    not core_image_available_for_tests(), reason="Core Image not available"
)
@pytest.mark.parametrize("file_stem,muimg_threshold,muimg_xfail,ci_threshold,ci_xfail", TEST_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_coreimage_xmp_rendering(file_stem, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail, output_dir):
    """Test Core Image XMP-based rendering against Photoshop reference.
    
    Validates that Core Image rendered output matches Photoshop reference within threshold.
    XMP NOOP filtering (Exposure2012=0, WhiteBalance="As Shot", linear tone curves)
    is handled automatically by the rendering pipeline.
    
    Args:
        file_stem: Base filename without extension
        muimg_threshold: MUIMG threshold (unused in this test)
        muimg_xfail: MUIMG xfail flag (unused in this test)
        ci_threshold: Threshold for Core Image pipeline (None = 2.0% default)
        ci_xfail: True to mark as expected failure for Core Image
    """
    # Mark as xfail if specified
    if ci_xfail:
        pytest.xfail(f"Core Image rendering known to differ significantly for {file_stem}")
    
    dng_path = XMP_TEST_DIR / f"{file_stem}.dng"
    tif_path = XMP_TEST_DIR / f"{file_stem}.tif"
    
    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")
    if not tif_path.exists():
        pytest.skip(f"Reference file not found: {tif_path}")
    
    # Render with Core Image
    result = muimg.decode_dng(
        file=str(dng_path),
        output_dtype=np.uint16,
        use_coreimage_if_available=True,
        use_xmp=True,
    )
    
    assert result is not None, f"Core Image rendering failed for {file_stem}"
    
    # Save rendered output for manual inspection
    output_name = file_stem + "_coreimage.tif"
    tifffile.imwrite(str(output_dir / output_name), result)
    
    # Load Photoshop reference
    ref = tifffile.imread(str(tif_path))
    
    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(
            f"Shape mismatch for {file_stem}:\n"
            f"  Core Image: {result.shape}\n"
            f"  Photoshop:  {ref.shape}"
        )
    
    # Compare against reference
    stats = compute_diff_stats(result, ref)
    
    # Use Core Image specific threshold if provided, otherwise use 2.0% default
    effective_threshold = ci_threshold if ci_threshold is not None else 2.0
    threshold_label = f"{effective_threshold}%" if ci_threshold is not None else f"{effective_threshold}% (default)"
    
    # Print overall and per-channel diffs
    print(f"\n  [CI] {file_stem}: diff={stats['mean']:.2f}% (threshold={threshold_label})")
    if 'mean_R' in stats:
        print(f"       Per-channel: R={stats['mean_R']:.2f}% G={stats['mean_G']:.2f}% B={stats['mean_B']:.2f}%")
    
    assert stats["mean"] < effective_threshold, (
        f"Core Image XMP rendering diff {stats['mean']:.2f}% > {effective_threshold}% for {file_stem}"
    )

