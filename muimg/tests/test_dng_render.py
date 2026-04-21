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
from conftest import compute_diff_stats, core_image_available_for_tests, run_dng_validate, DNG_VALIDATE_PATH

# Suppress tifffile logging about Photoshop TIFF metadata inconsistencies
logging.getLogger('tifffile').setLevel(logging.CRITICAL)


# Test data directory
DNGFILES_DIR = Path(__file__).parent / "dngfiles"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_dng_render"


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# Test files with thresholds for 4 comparisons:
# Format: (file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh)
# Thresholds >= 1.8% are automatically marked as xfail (expected to fail)
# None = use default threshold (1.5% for muimg, 2.0% for CI)
TEST_CASES = [
    # Files from prepare script (33 files, excluding the 3 multi-IFD files)
    # Format: (file_stem, muimg_ps_thresh, ci_ps_thresh, muimg_dngval_thresh, ci_dngval_thresh)
    ("apple_iphone_15_pro.1", 0.5, 1.0, 0.3, 0.9),  # PS: MUIMG 0.44%, CI 0.90%; dngval: MUIMG 0.19%, CI 0.79%
    ("arashi_vision_insta360_oners.1", 1.8, 1.8, 0.1, 0.5),  # PS: MUIMG 2.08%, CI 1.96%; dngval: MUIMG 0.01%, CI 0.44%
    ("canon_eos_r5.baselineexposure-bl1", 1.8, 1.8, 0.1, 0.4),  # PS: MUIMG 2.28%, CI 1.92%; dngval: MUIMG 0.02%, CI 0.36%
    ("canon_eos_r5.exposure", 1.8, 1.8, 0.1, 0.6),  # PS: MUIMG 2.15%, CI 4.08%; dngval: MUIMG 0.01%, CI 0.50%
    ("canon_eos_r5.none", 1.4, 1.79, 0.1, 0.6),  # PS: MUIMG 1.21%, CI 1.71%; dngval: MUIMG 0.01%, CI 0.50%
    ("canon_eos_r5.temp-tint", 1.4, 1.79, 0.1, 0.6),  # PS: MUIMG 1.22%, CI 1.73%; dngval: MUIMG 0.01%, CI 0.50%
    ("nikon_corporation_nikon_z_8.1", 1.8, 1.8, 0.4, 1.4),  # PS: MUIMG 2.08%, CI 2.86%; dngval: MUIMG 0.27%, CI 1.24%
    ("nikon_corporation_nikon_z_9.1", 1.79, 1.8, 0.1, 1.0),  # PS: MUIMG 1.75%, CI 2.24%; dngval: MUIMG 0.06%, CI 0.84%
    ("olympus_corporation_e-m10markiv.1", 1.5, 1.79, 0.4, 0.6),  # PS: MUIMG 1.33%, CI 1.72%; dngval: MUIMG 0.30%, CI 0.54%
    ("panasonic_dc-s9.1", 1.5, 1.8, 0.3, 1.6),  # PS: MUIMG 1.35%, CI 1.87%; dngval: MUIMG 0.22%, CI 1.43%
    ("rgb_ramp_test.1", 0.9, 1.8, 0.1, 1.8),  # PS: MUIMG 0.74%, CI 2.73%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.bcurve", 0.7, 1.8, 0.1, 1.8),  # PS: MUIMG 0.60%, CI 1.83%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.bcurve2", 0.9, 1.8, 0.1, 1.8),  # PS: MUIMG 0.81%, CI 3.38%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.gcurve", 0.7, 1.8, 0.1, 1.8),  # PS: MUIMG 0.57%, CI 1.99%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.maincurve", 1.5, 1.8, 0.1, 1.8),  # PS: MUIMG 1.36%, CI 2.29%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.mainrgbcurve", 1.2, 1.8, 0.1, 1.8),  # PS: MUIMG 1.06%, CI 2.34%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.multi-curve", 1.5, 1.8, 0.1, 1.8),  # PS: MUIMG 1.32%, CI 2.34%; dngval: MUIMG 0.01%, CI 2.25%
    ("rgb_ramp_test.rcurve", 0.6, 1.79, 0.1, 1.8),  # PS: MUIMG 0.51%, CI 1.64%; dngval: MUIMG 0.01%, CI 2.25%
    ("ricoh_imaging_company_ltd_pentax_k-3_mark_iii.1", 1.4, 1.4, 0.3, 1.0),  # PS: MUIMG 1.23%, CI 1.24%; dngval: MUIMG 0.24%, CI 0.82%
    ("sony_dsc-rx100m7.1", 1.1, 1.2, 0.3, 0.7),  # PS: MUIMG 1.00%, CI 1.07%; dngval: MUIMG 0.21%, CI 0.60%
    ("sony_ilce-7c.1", 1.79, 1.8, 0.1, 0.8),  # PS: MUIMG 1.69%, CI 2.37%; dngval: MUIMG 0.04%, CI 0.64%
    ("test_linear_gradient.exp-curve", 1.8, None, 0.1, None),  # PS: MUIMG 2.92%, CI shape mismatch; dngval: MUIMG 0.00%, CI shape mismatch
    ("zwo_asi676mc.1", 0.8, 1.3, 0.2, 1.7),  # PS: MUIMG 0.69%, CI 1.12%; dngval: MUIMG 0.14%, CI 1.48%
    ("zwo_asi676mc_asi676mc.colorpattern.bcurve", 0.6, 1.8, 0.1, 1.5),  # PS: MUIMG 0.54%, CI 1.81%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.colorpattern.exposure", 0.7, 1.2, 0.1, 1.5),  # PS: MUIMG 0.62%, CI 1.02%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.colorpattern.gcurve", 0.7, 1.79, 0.1, 1.5),  # PS: MUIMG 0.63%, CI 1.70%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.colorpattern.none", 0.8, 1.8, 0.1, 1.5),  # PS: MUIMG 0.66%, CI 1.84%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.colorpattern.rcurve", 0.7, 1.7, 0.1, 1.5),  # PS: MUIMG 0.63%, CI 1.48%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.colorpattern.temp-tint", 0.9, 1.8, 0.1, 1.5),  # PS: MUIMG 0.78%, CI 1.84%; dngval: MUIMG 0.00%, CI 1.32%
    ("zwo_asi676mc_asi676mc.curve", 1.1, 1.2, 0.1, 0.3),  # PS: MUIMG 0.96%, CI 1.00%; dngval: MUIMG 0.00%, CI 0.25%
    ("zwo_asi676mc_asi676mc.exposure", 0.6, 0.7, 0.1, 0.3),  # PS: MUIMG 0.50%, CI 0.59%; dngval: MUIMG 0.00%, CI 0.25%
    ("zwo_asi676mc_asi676mc.none", 0.9, 1.0, 0.1, 0.3),  # PS: MUIMG 0.81%, CI 0.86%; dngval: MUIMG 0.00%, CI 0.25%
    ("zwo_asi676mc_asi676mc.rgbcurve", 1.1, 1.1, 0.1, 0.3),  # PS: MUIMG 0.94%, CI 0.97%; dngval: MUIMG 0.00%, CI 0.25%
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
    result = muimg.decode_dng(
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
    ci_result = muimg.decode_dng(
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
