# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Tests for DNG rendering against Photoshop and dng_validate references.

Validates that XMP metadata (Temperature, Tint, Exposure2012, ToneCurvePV2012)
is correctly parsed and applied during rendering, comparing output against both
Photoshop-rendered TIFF files and dng_validate output.
"""
import pytest
import numpy as np
from pathlib import Path
import tifffile
import imagecodecs
import logging

import muimg
from muimg.raw_render import DemosaicAlgorithm
from conftest import (
    compute_diff_stats,
    core_image_available_for_tests,
    run_dng_validate,
    DNG_VALIDATE_PATH,
    OutputPathManager,
)

# Suppress tifffile logging about Photoshop TIFF metadata inconsistencies
logging.getLogger('tifffile').setLevel(logging.CRITICAL)


# Test data directory
DNGFILES_DIR = Path(__file__).parent / "dngfiles"

# Test output path manager - set persistent=True to keep outputs, False for tmp_path
output_path_manager = OutputPathManager(persistent=True)


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    """Create output directory for test artifacts."""
    tmp_path = tmp_path_factory.mktemp("test_dng_render")
    return output_path_manager.get_path(tmp_path, "test_dng_render")


@pytest.fixture(scope="module")
def cli_batch_output_dir(output_dir):
    """Run CLI batch-convert once and return the output directory."""
    import subprocess
    import sys
    
    cli_output_dir = output_dir / "cli_batch_convert"
    cli_output_dir.mkdir(exist_ok=True)
    
    # Run CLI batch-convert command once for all test files
    cmd = [
        sys.executable, "-m", "muimg.cli",
        "dng", "batch-convert",
        str(DNGFILES_DIR),
        str(cli_output_dir),
        "--bit-depth", "16",
        "--format", "tif",
        "--num-workers", "4",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.fail(f"CLI batch-convert failed: {result.stderr}")
    
    return cli_output_dir


# Test files with thresholds for 4 comparisons:
# Format: (file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh)
# Thresholds >= 1.8% are automatically marked as xfail (expected to fail)
# Thresholds calculated as: 1.8 if diff >= 1.8 else min(diff * 1.1, 1.79)
# Note: Small values (0.00, 0.01, 0.02) adjusted +0.01 to avoid floating-point precision issues
TEST_CASES = [
    # Camera files
    ("asi676mc_1", 0.74, 1.23, 0.15, 1.57),  # PS: MUIMG 0.67%, CI 1.12%; dngval: MUIMG 0.14%, CI 1.43%
    ("canon_eos_r5_baselineexposure-bl1", 1.67, 1.80, 0.02, 0.39),  # PS: MUIMG 1.52%, CI 1.93%; dngval: MUIMG 0.02%, CI 0.35%
    ("canon_eos_r5_exposure", 1.80, 1.80, 0.02, 0.59),  # PS: MUIMG 2.41%, CI 4.17%; dngval: MUIMG 0.01%, CI 0.54%
    ("canon_eos_r5_none", 1.30, 1.79, 0.02, 0.59),  # PS: MUIMG 1.18%, CI 1.74%; dngval: MUIMG 0.01%, CI 0.54%
    ("canon_eos_r5_temp-tint", 1.30, 1.79, 0.02, 0.59),  # PS: MUIMG 1.18%, CI 1.77%; dngval: MUIMG 0.01%, CI 0.54%
    ("dc-s9_1", 1.64, 1.80, 0.03, 1.40),  # PS: MUIMG 1.49%, CI 1.98%; dngval: MUIMG 0.02%, CI 1.27%
    ("e-m10markiv_1", 1.80, 1.80, 0.01, 0.45),  # PS: MUIMG 2.01%, CI 2.60%; dngval: MUIMG 0.01%, CI 0.41%
    ("insta360_oners", 1.80, 1.80, 0.01, 0.48),  # PS: MUIMG 2.36%, CI 2.23%; dngval: MUIMG 0.01%, CI 0.44%
    ("iphone_15_pro_1_back_camera_lossy", 0.63, 1.08, 0.21, 0.87),  # PS: MUIMG 0.57%, CI 0.98%; dngval: MUIMG 0.19%, CI 0.79%
    ("iphone_15_pro_flowers", 1.80, 1.80, 0.23, 0.98),  # PS: MUIMG 27.83%, CI 28.67%; dngval: MUIMG 0.21%, CI 0.89%
    ("leica_sl2-s_1", 1.64, 1.62, 0.02, 0.34),  # PS: MUIMG 1.49%, CI 1.47%; dngval: MUIMG 0.01%, CI 0.31%
    ("nikon_z_8_1437", 1.53, 1.79, 0.02, 0.81),  # PS: MUIMG 1.39%, CI 1.70%; dngval: MUIMG 0.01%, CI 0.74%
    ("nikon_z_9", 1.79, 1.80, 0.03, 0.90),  # PS: MUIMG 1.76%, CI 2.24%; dngval: MUIMG 0.02%, CI 0.82%
    ("pentax_k-3_mark_iii_1", 1.61, 1.64, 0.25, 0.90),  # PS: MUIMG 1.46%, CI 1.49%; dngval: MUIMG 0.23%, CI 0.82%
    ("pixel_9_pro_fold_20240902_030456036", 1.80, 1.80, 1.80, 1.80),  # PS: MUIMG 7.67%, CI 6.89%; dngval: MUIMG 1.88%, CI 2.02%
    ("ricoh_gr_iiix_1", 1.24, 1.16, 0.21, 0.44),  # PS: MUIMG 1.13%, CI 1.05%; dngval: MUIMG 0.19%, CI 0.40%
    ("sony_dsc-rx100m7", 1.06, 1.08, 0.02, 0.55),  # PS: MUIMG 0.96%, CI 0.98%; dngval: MUIMG 0.01%, CI 0.50%
    ("sony_ilce-7c_lossy", 1.00, 1.56, 0.02, 0.68),  # PS: MUIMG 0.91%, CI 1.42%; dngval: MUIMG 0.02%, CI 0.62%
    # Test pattern files
    ("asi676mc_colorpattern_bcurve", 0.63, 1.80, 0.01, 1.45),  # PS: MUIMG 0.57%, CI 1.82%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_colorpattern_exposure", 0.72, 1.13, 0.01, 1.45),  # PS: MUIMG 0.65%, CI 1.03%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_colorpattern_gcurve", 0.73, 1.79, 0.01, 1.45),  # PS: MUIMG 0.66%, CI 1.72%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_colorpattern_none", 0.77, 1.80, 0.01, 1.45),  # PS: MUIMG 0.70%, CI 1.85%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_colorpattern_rcurve", 0.73, 1.64, 0.01, 1.45),  # PS: MUIMG 0.66%, CI 1.49%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_colorpattern_temp-tint", 0.89, 1.80, 0.01, 1.45),  # PS: MUIMG 0.81%, CI 1.85%; dngval: MUIMG 0.00%, CI 1.32%
    ("asi676mc_curve", 1.46, 1.50, 0.01, 0.28),  # PS: MUIMG 1.33%, CI 1.36%; dngval: MUIMG 0.00%, CI 0.25%
    ("asi676mc_exposure", 0.78, 0.86, 0.01, 0.28),  # PS: MUIMG 0.71%, CI 0.78%; dngval: MUIMG 0.00%, CI 0.25%
    ("asi676mc_none", 1.23, 1.28, 0.01, 0.28),  # PS: MUIMG 1.12%, CI 1.16%; dngval: MUIMG 0.00%, CI 0.25%
    ("asi676mc_rgbcurve", 1.45, 1.47, 0.01, 0.28),  # PS: MUIMG 1.32%, CI 1.34%; dngval: MUIMG 0.00%, CI 0.25%
    ("linear_gradient_test_exp-curve", 1.80, 1.80, 0.01, 0.12),  # PS: MUIMG 2.92%, CI 13.73%; dngval: MUIMG 0.00%, CI 0.11%
    ("rgb_ramp_test", 1.80, 1.80, 0.02, 1.80),  # PS: MUIMG 9.96%, CI 10.38%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_bcurve", 1.80, 1.80, 0.02, 1.80),  # PS: MUIMG 11.72%, CI 12.47%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_bcurve2", 1.01, 1.80, 0.02, 1.80),  # PS: MUIMG 0.92%, CI 3.43%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_gcurve", 0.74, 1.80, 0.02, 1.80),  # PS: MUIMG 0.67%, CI 2.05%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_maincurve", 1.56, 1.80, 0.02, 1.80),  # PS: MUIMG 1.42%, CI 2.34%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_mainrgbcurve", 1.26, 1.80, 0.02, 1.80),  # PS: MUIMG 1.15%, CI 2.40%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_multi-curve", 1.52, 1.80, 0.02, 1.80),  # PS: MUIMG 1.38%, CI 2.40%; dngval: MUIMG 0.01%, CI 2.29%
    ("rgb_ramp_test_rcurve", 0.66, 1.79, 0.02, 1.80),  # PS: MUIMG 0.60%, CI 1.68%; dngval: MUIMG 0.01%, CI 2.29%
]


@pytest.mark.parametrize(
    "file_stem,muimg_ps_thresh,ci_ps_thresh,muimg_dngval_thresh,ci_dngval_thresh",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_muimg_vs_photoshop(
    file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh,
    output_dir, request
):
    """Test MUIMG rendering against Photoshop reference."""
    if muimg_ps_thresh is None:
        pytest.skip(f"MUIMG vs Photoshop skipped for {file_stem} (shape mismatch or known issue)")
    
    threshold = muimg_ps_thresh
    if threshold >= 1.8:
        request.applymarker(pytest.mark.xfail(reason=f"MUIMG vs Photoshop diff > 1.8%"))

    dng_path = DNGFILES_DIR / f"{file_stem}.dng"
    jxl_path = DNGFILES_DIR / f"{file_stem}.jxl"

    if not dng_path.exists():
        pytest.skip(f"Test DNG not found: {dng_path}")

    if not jxl_path.exists():
        pytest.skip(f"Photoshop reference not found: {jxl_path}")

    # Render with MUIMG
    with muimg.DngFile(dng_path) as dng:
        result = dng.render_raw(
            output_dtype=np.uint16,
            demosaic_algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR,
            strict=False,
            use_xmp=True,
            rendering_params={"highlight_preserving_exposure": True},
        )

    assert result is not None, f"MUIMG rendering failed for {file_stem}"

    # Save rendered output
    tifffile.imwrite(str(output_dir / f"{file_stem}_muimg.tif"), result)

    # Load Photoshop reference from JXL
    jxl_data = jxl_path.read_bytes()
    ref = imagecodecs.jpegxl_decode(jxl_data)

    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: MUIMG={result.shape}, Photoshop={ref.shape}")

    # Compare
    stats = compute_diff_stats(result, ref)
    measured_diff = stats['mean']

    print(f"\n  [MUIMG vs PS] {file_stem}: diff={measured_diff:.2f}% (threshold={threshold}%)")

    if measured_diff >= threshold:
        pytest.fail(f"MUIMG vs Photoshop: measured {measured_diff:.2f}% > threshold {threshold}%")
    
    assert measured_diff < threshold


@pytest.mark.skipif(not core_image_available_for_tests(), reason="Core Image not available")
@pytest.mark.parametrize(
    "file_stem,muimg_ps_thresh,ci_ps_thresh,muimg_dngval_thresh,ci_dngval_thresh",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_coreimage_vs_photoshop(
    file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh,
    output_dir, request
):
    """Test Core Image rendering against Photoshop reference."""
    if ci_ps_thresh is None:
        request.applymarker(pytest.mark.xfail(reason=f"Core Image vs Photoshop shape mismatch for {file_stem}"))
        threshold = 100.0  # Set high threshold so test will fail
    else:
        threshold = ci_ps_thresh
        if threshold >= 1.8:
            request.applymarker(pytest.mark.xfail(reason=f"Core Image vs Photoshop diff > 1.8%"))

    dng_path = DNGFILES_DIR / f"{file_stem}.dng"
    jxl_path = DNGFILES_DIR / f"{file_stem}.jxl"

    if not dng_path.exists():
        pytest.skip(f"Test DNG not found: {dng_path}")

    if not jxl_path.exists():
        pytest.skip(f"Photoshop reference not found: {jxl_path}")

    # Render with Core Image (use_xmp=True, no highlight preservation)
    result, _ = muimg.decode_dng(
        file=str(dng_path),
        output_dtype=np.uint16,
        use_coreimage_if_available=True,
        use_xmp=True,
    )

    assert result is not None, f"Core Image rendering failed for {file_stem}"

    # Save rendered output
    tifffile.imwrite(str(output_dir / f"{file_stem}_coreimage.tif"), result)

    # Load Photoshop reference from JXL
    jxl_data = jxl_path.read_bytes()
    ref = imagecodecs.jpegxl_decode(jxl_data)

    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: CoreImage={result.shape}, Photoshop={ref.shape}")

    # Compare
    stats = compute_diff_stats(result, ref)
    measured_diff = stats['mean']

    print(f"\n  [CI vs PS] {file_stem}: diff={measured_diff:.2f}% (threshold={threshold}%)")

    if measured_diff >= threshold:
        pytest.fail(f"Core Image vs Photoshop: measured {measured_diff:.2f}% > threshold {threshold}%")
    
    assert measured_diff < threshold


@pytest.mark.skipif(not DNG_VALIDATE_PATH.exists(), reason="dng_validate not available")
@pytest.mark.parametrize(
    "file_stem,muimg_ps_thresh,ci_ps_thresh,muimg_dngval_thresh,ci_dngval_thresh",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_muimg_vs_dng_validate(
    file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh,
    output_dir, request
):
    """Test MUIMG rendering against dng_validate reference."""
    if muimg_dngval_thresh is None:
        pytest.skip(f"MUIMG vs dng_validate skipped for {file_stem} (shape mismatch or known issue)")
    
    threshold = muimg_dngval_thresh
    if threshold >= 1.8:
        request.applymarker(pytest.mark.xfail(reason=f"MUIMG vs dng_validate diff > 1.8%"))

    dng_path = DNGFILES_DIR / f"{file_stem}.dng"

    if not dng_path.exists():
        pytest.skip(f"Test DNG not found: {dng_path}")

    # Render with MUIMG (use_xmp=False, no highlight preservation)
    with muimg.DngFile(dng_path) as dng:
        muimg_result = dng.render_raw(
            output_dtype=np.uint16,
            demosaic_algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR,
            strict=False,
            use_xmp=False,
            rendering_params={"highlight_preserving_exposure": False},
        )

    assert muimg_result is not None, f"MUIMG rendering failed for {file_stem}"

    # Render with dng_validate
    dngval_base = output_dir / f"{file_stem}_dngvalidate"
    dngval_result = run_dng_validate(dng_path, dngval_base, timeout=60, validate=False)

    if dngval_result is None:
        pytest.skip(f"dng_validate failed for {file_stem}")

    # Ensure same shape
    if muimg_result.shape != dngval_result.shape:
        pytest.fail(f"Shape mismatch: MUIMG={muimg_result.shape}, dng_validate={dngval_result.shape}")

    # Compare
    stats = compute_diff_stats(muimg_result, dngval_result)
    measured_diff = stats['mean']

    print(f"\n  [MUIMG vs dngval] {file_stem}: diff={measured_diff:.2f}% (threshold={threshold}%)")

    if measured_diff >= threshold:
        pytest.fail(f"MUIMG vs dng_validate: measured {measured_diff:.2f}% > threshold {threshold}%")
    
    assert measured_diff < threshold


@pytest.mark.skipif(not core_image_available_for_tests(), reason="Core Image not available")
@pytest.mark.skipif(not DNG_VALIDATE_PATH.exists(), reason="dng_validate not available")
@pytest.mark.parametrize(
    "file_stem,muimg_ps_thresh,ci_ps_thresh,muimg_dngval_thresh,ci_dngval_thresh",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_coreimage_vs_dng_validate(
    file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh,
    output_dir, request
):
    """Test Core Image rendering against dng_validate reference."""
    if ci_dngval_thresh is None:
        request.applymarker(pytest.mark.xfail(reason=f"Core Image vs dng_validate shape mismatch for {file_stem}"))
        threshold = 100.0  # Set high threshold so test will fail
    else:
        threshold = ci_dngval_thresh
        if threshold >= 1.8:
            request.applymarker(pytest.mark.xfail(reason=f"Core Image vs dng_validate diff > 1.8%"))

    dng_path = DNGFILES_DIR / f"{file_stem}.dng"

    if not dng_path.exists():
        pytest.skip(f"Test DNG not found: {dng_path}")

    # Render with Core Image (use_xmp=False, no highlight preservation)
    ci_result, _ = muimg.decode_dng(
        file=str(dng_path),
        output_dtype=np.uint16,
        use_coreimage_if_available=True,
        use_xmp=False,
    )

    assert ci_result is not None, f"Core Image rendering failed for {file_stem}"

    # Render with dng_validate
    dngval_base = output_dir / f"{file_stem}_dngvalidate"
    dngval_result = run_dng_validate(dng_path, dngval_base, timeout=60, validate=False)

    if dngval_result is None:
        pytest.skip(f"dng_validate failed for {file_stem}")

    # Ensure same shape
    if ci_result.shape != dngval_result.shape:
        pytest.fail(f"Shape mismatch: CoreImage={ci_result.shape}, dng_validate={dngval_result.shape}")

    # Compare
    stats = compute_diff_stats(ci_result, dngval_result)
    measured_diff = stats['mean']

    print(f"\n  [CI vs dngval] {file_stem}: diff={measured_diff:.2f}% (threshold={threshold}%)")

    if measured_diff >= threshold:
        pytest.fail(f"Core Image vs dng_validate: measured {measured_diff:.2f}% > threshold {threshold}%")
    
    assert measured_diff < threshold


@pytest.mark.parametrize(
    "file_stem,muimg_ps_thresh,ci_ps_thresh,muimg_dngval_thresh,ci_dngval_thresh",
    TEST_CASES,
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_cli_batch_convert_vs_photoshop(
    file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh,
    cli_batch_output_dir, request
):
    """Test CLI batch-convert with multiple workers against Photoshop reference.
    
    Uses a module-scoped fixture that runs batch-convert once for all test cases.
    """
    if muimg_ps_thresh is None:
        pytest.skip(f"CLI batch-convert vs Photoshop skipped for {file_stem} (shape mismatch or known issue)")
    
    threshold = muimg_ps_thresh
    if threshold >= 1.8:
        request.applymarker(pytest.mark.xfail(reason=f"CLI batch-convert vs Photoshop diff > 1.8%"))

    jxl_path = DNGFILES_DIR / f"{file_stem}.jxl"

    if not jxl_path.exists():
        pytest.skip(f"Photoshop reference not found: {jxl_path}")

    # Load the output file from the cached batch-convert run
    output_tif = cli_batch_output_dir / f"{file_stem}.tif"
    if not output_tif.exists():
        pytest.skip(f"CLI batch-convert did not produce output: {output_tif}")

    cli_result = tifffile.imread(str(output_tif))

    # Convert to uint16 for comparison (CLI output is uint8)
    if cli_result.dtype == np.uint8:
        cli_result = (cli_result.astype(np.uint16) * 257)  # Scale 0-255 to 0-65535

    # Load Photoshop reference from JXL
    jxl_data = jxl_path.read_bytes()
    ref = imagecodecs.jpegxl_decode(jxl_data)

    # Ensure same shape
    if cli_result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: CLI={cli_result.shape}, Photoshop={ref.shape}")

    # Compare
    stats = compute_diff_stats(cli_result, ref)
    measured_diff = stats['mean']

    print(f"\n  [CLI batch-convert vs PS] {file_stem}: diff={measured_diff:.2f}% (threshold={threshold}%)")

    if measured_diff >= threshold:
        pytest.fail(f"CLI batch-convert vs Photoshop: measured {measured_diff:.2f}% > threshold {threshold}%")
    
    assert measured_diff < threshold
