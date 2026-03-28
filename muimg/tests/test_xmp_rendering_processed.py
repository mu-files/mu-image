"""Tests for XMP-based rendering against Photoshop reference images.

Uses processed (down-sized) test files from mu-image-testdata/processed.
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


# Test data directory (processed/down-sized files)
TEST_DIR = Path(__file__).parent.parent.parent.parent / "mu-image-testdata" / "processed"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_xmp_rendering_processed"


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
    # Color pattern tests (300x200, 6x4 patches) - small uncompressed files
    ("asi676mc.colorpattern.none", None, False, None, False),
    ("asi676mc.colorpattern.exposure", None, False, None, False),
    ("asi676mc.colorpattern.temp-tint", None, False, None, False),
    ("asi676mc.colorpattern.rcurve", None, False, None, False),
    ("asi676mc.colorpattern.gcurve", None, False, None, False),
    ("asi676mc.colorpattern.bcurve", None, False, None, False),

    # Full-size ASI676MC tests (scaled + JXL)
    ("asi676mc.none", None, False, None, False),
    ("asi676mc.exposure", None, False, None, False),
    ("asi676mc.curve", None, False, None, False),
    ("asi676mc.rgbcurve", None, False, None, False),

    # Synthetic color patterns - small uncompressed files
    ("linear_gradient.exp-curve", None, False, None, False),
    ("rgb_ramp_test.1", None, False, None, False),
    ("rgb_ramp_test.rcurve", None, False, None, False),
    ("rgb_ramp_test.gcurve", None, False, None, False),
    ("rgb_ramp_test.bcurve", None, False, None, False),
    ("rgb_ramp_test.bcurve2", None, False, None, False),
    ("rgb_ramp_test.multi-curve", None, False, None, False),
    ("rgb_ramp_test.maincurve", None, False, None, False),
    ("rgb_ramp_test.mainrgbcurve", None, False, None, False),

    # Canon EOS R5 tests (JXL compressed)
    ("canon_eos_r5.none", None, False, None, False),
    ("canon_eos_r5.exposure", None, False, None, False),
    ("canon_eos_r5.baselineexposure-bl1", None, False, None, False),
    ("canon_eos_r5.temp-tint", None, False, None, False),

    # Camera tests (scaled + JXL)
    ("e-m10markiv.1", None, False, None, False),  # Olympus E-M10MarkIV
    ("nikon_z_8.1", None, False, None, False),  # NIKON Z 8
    ("asi676mc.1", None, False, None, False),  # test_stacking3_RCD_corrected_cfa
    ("dc-s9.1", None, False, None, False),  # Panasonic DC-S9
    ("nikon_z_9.1", None, False, None, False),  # NIKON Z 9
    ("leica_sl2-s.1", None, False, None, False),  # Leica SL2-S
    ("pentax_k-3_mark_iii.1", None, False, None, False),  # Pentax K-3 Mark III
]


@pytest.mark.parametrize(
    "file_stem,muimg_threshold,muimg_xfail,ci_threshold,ci_xfail",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_muimg_xmp_rendering(
    file_stem, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail, output_dir, request
):
    """Test MUIMG XMP-based rendering against Photoshop reference.

    Validates that rendered output matches Photoshop reference within threshold.
    """
    if muimg_xfail:
        request.applymarker(pytest.mark.xfail(reason=f"MUIMG rendering known to fail for {file_stem}"))

    dng_path = TEST_DIR / f"{file_stem}.dng"
    tif_path = TEST_DIR / f"{file_stem}.tif"

    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")

    # Render with automatic XMP extraction
    with muimg.DngFile(dng_path) as dng:
        result = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            strict=False,
            use_xmp=True,
            rendering_params={"highlight_preserving_exposure": True},
        )

        assert result is not None, f"Rendering failed for {file_stem}"

    # Save rendered output for manual inspection
    output_name = file_stem + "_muimg.tif"
    tifffile.imwrite(str(output_dir / output_name), result)

    # Skip comparison if no reference TIF (but rendering succeeded)
    if not tif_path.exists():
        print(f"\n  [MUIMG] {file_stem}: rendered OK, no reference TIF for comparison")
        return

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
    threshold_label = (
        f"{effective_threshold}%"
        if muimg_threshold is not None
        else f"{effective_threshold}% (default)"
    )

    # Print overall and per-channel diffs
    print(
        f"\n  [MUIMG] {file_stem}: diff={stats['mean']:.2f}% "
        f"(threshold={threshold_label})"
    )
    if "mean_R" in stats:
        print(
            f"          Per-channel: R={stats['mean_R']:.2f}% "
            f"G={stats['mean_G']:.2f}% B={stats['mean_B']:.2f}%"
        )

    assert stats["mean"] < effective_threshold, (
        f"MUIMG XMP rendering diff {stats['mean']:.2f}% > "
        f"{effective_threshold}% for {file_stem}"
    )


@pytest.mark.skipif(
    not core_image_available_for_tests(), reason="Core Image not available"
)
@pytest.mark.parametrize(
    "file_stem,muimg_threshold,muimg_xfail,ci_threshold,ci_xfail",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_coreimage_xmp_rendering(
    file_stem, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail, output_dir, request
):
    """Test Core Image XMP-based rendering against Photoshop reference."""
    if ci_xfail:
        request.applymarker(pytest.mark.xfail(reason=f"Core Image rendering known to differ significantly for {file_stem}"))

    dng_path = TEST_DIR / f"{file_stem}.dng"
    tif_path = TEST_DIR / f"{file_stem}.tif"

    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")

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

    # Skip comparison if no reference TIF (but rendering succeeded)
    if not tif_path.exists():
        print(f"\n  [CI] {file_stem}: rendered OK, no reference TIF for comparison")
        return

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
    threshold_label = (
        f"{effective_threshold}%"
        if ci_threshold is not None
        else f"{effective_threshold}% (default)"
    )

    # Print overall and per-channel diffs
    print(
        f"\n  [CI] {file_stem}: diff={stats['mean']:.2f}% "
        f"(threshold={threshold_label})"
    )
    if "mean_R" in stats:
        print(
            f"       Per-channel: R={stats['mean_R']:.2f}% "
            f"G={stats['mean_G']:.2f}% B={stats['mean_B']:.2f}%"
        )

    assert stats["mean"] < effective_threshold, (
        f"Core Image XMP rendering diff {stats['mean']:.2f}% > "
        f"{effective_threshold}% for {file_stem}"
    )
