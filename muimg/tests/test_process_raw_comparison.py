"""Tests for DngFile.render() and render_dng_coreimage().

Compares MUIMG (our Python port) and Core Image against C++ DNG SDK reference (dng_validate).
"""

import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

import muimg
from muimg.dngio import IfdSpec
from conftest import (
    TEST_FILES_DIR,
    LOCAL_TEST_FILES_DIR,
    DNG_VALIDATE_PATH,
    compute_diff_stats,
    core_image_available_for_tests,
    run_dng_validate,
)

# Output directory for comparison files
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_process_raw_comparison"

# Test cases for DNG rendering comparison
# Format: (filename, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail)
# Thresholds are 1.1x above measured values
# muimg_threshold: threshold for MUIMG/SDK pipeline (None = use default 0.15%)
# muimg_xfail: True to mark as expected failure for MUIMG pipeline
# ci_threshold: threshold for Core Image pipeline (None = use default 2.75%)
# ci_xfail: True to mark as expected failure for Core Image pipeline
TEST_CASES = [
    # ASI676MC
    ("asi676mc.cfa.jxl_lossy.1ifds.dng", 0.08, False, None, True),  # muimg: 0.07%, CI: 2.51% (>2%)
    ("asi676mc.cfa.jxl_lossy.2ifds.dng", 0.03, False, 0.75, False),  # muimg: 0.02%, CI: 0.68%
    ("asi676mc.cfa.uncomp.2ifds.dng", 0.04, False, 0.87, False),  # muimg: 0.03%, CI: 0.87%
    ("asi676mc.linearraw.jxl_lossy.1ifds.dng", 0.18, False, 1.06, False),  # muimg: 0.17%, CI: 1.06%
    
    # Canon EOS R5
    ("canon_eos_r5.baselineexposure-bl0.dng", 0.1, False, 0.76, False),  # muimg: 0.02%, CI: 0.76%
    ("canon_eos_r5.baselineexposure-bl1.dng", 0.1, False, 0.37, False),  # muimg: 0.02%, CI: 0.37%
    ("canon_eos_r5.cfa.ljpeg.6ifds.dng", 0.02, False, 1.73, False),  # muimg: 0.02%, CI: 1.73%
    ("canon_eos_r5_mark_ii.linearraw.jxl_lossy.6ifds.dng", 0.02, False, None, True),  # muimg: 0.02%, CI: 22.24%
    
    # DNG SDK test files - JXL
    ("dngsdk.01_jxl_linear_raw_integer.dng", 0.01, False, 0.74, False),  # muimg: 0.01%, CI: 0.74%
    ("dngsdk.02_jxl_linear_raw_float.dng", 0.09, False, None, True),  # muimg: 0.08%, CI: 9.74%
    ("dngsdk.03_jxl_bayer_raw_integer.dng", 0.22, False, 1.89, False),  # muimg: 0.20%, CI: 1.89%
    
    # DNG SDK test files - PGTM2
    ("dngsdk.04_PGTM2_per_profile.dng", 0.01, False, 0.67, False),  # muimg: 0.00%, CI: 0.61%
    ("dngsdk.05_PGTM2_unsigned8.dng", 0.01, False, 0.46, False),  # muimg: 0.00%, CI: 0.46%
    ("dngsdk.06_PGTM2_unsigned16.dng", 0.01, False, 0.46, False),  # muimg: 0.00%, CI: 0.46%
    ("dngsdk.07_PGTM2_float16.dng", 0.01, False, None, True),  # muimg: 0.00%, CI: 51.38%
    ("dngsdk.08_PGTM2_float32.dng", 0.01, False, 0.46, False),  # muimg: 0.00%, CI: 0.46%
    
    # DNG SDK test files - ImageSequenceInfo
    ("dngsdk.09_ImageSequenceInfo_1_of_3.dng", 0.01, False, 0.01, False),  # muimg: 0.00%, CI: 0.01%
    ("dngsdk.10_ImageSequenceInfo_2_of_3.dng", 0.01, False, 0.01, False),  # muimg: 0.00%, CI: 0.01%
    ("dngsdk.11_ImageSequenceInfo_3_of_3.dng", 0.01, False, 0.01, False),  # muimg: 0.00%, CI: 0.01%
    
    # DNG SDK test files - ImageStats
    ("dngsdk.12_ImageStats_WeightedAverage.dng", 0.02, False, None, True),  # muimg: 0.01%, CI: 33.75%
    ("dngsdk.13_ImageStats_Several.dng", 0.02, False, None, True),  # muimg: 0.01%, CI: 33.75%
    
    # DNG SDK test files - HDR/SDR
    ("dngsdk.14_hdr_sdr_profiles.dng", 0.01, False, None, True),  # muimg: 0.00%, CI: error
    
    # Insta360
    ("insta360_oners.cfa.uncomp.1ifds.dng", 0.02, False, None, True),  # muimg: 0.01%, CI: 4.69%
    
    # iPhone 16
    ("iphone16_1_back_camera.linearraw.jxl_lossy.1ifds.dng", 0.21, False, 0.87, False),  # muimg: 0.19%, CI: 0.79%
    ("iphone16_1_back_camera.linearraw.jxl_lossy.2ifds.dng", 0.72, False, 0.82, False),  # muimg: 0.65%, CI: 0.82%
    
    # Lumix S9
    ("lumixs9.dng", 0.15, False, None, True),  # muimg: 0.14%, CI: 2.90%
    
    # Nikon Z 9
    ("nikon_z_9.cfa.ljpeg.2ifds.dng", 0.15, False, None, True),  # muimg: 0.11%, CI: 2.33% (>2%)
    
    # Sony DSC-RX100M7
    ("sony_dsc-rx100m7.cfa.ljpeg.2ifds.dng", 0.28, False, None, True),  # muimg: 0.25%, CI: 4.15%
    ("sony_dsc-rx100m7.linearraw.jxl_lossy.2ifds.dng", 0.02, False, None, True),  # muimg: 0.01%, CI: 21.39%
    
    # Sony ILCE-7C
    ("sony_ilce-7c.cfa.jxl_lossy.1ifds.dng", 0.05, False, None, True),  # muimg: 0.04%, CI: 7.30%
    ("sony_ilce-7c.cfa.jxl_lossy.4ifds.dng", 0.05, False, 1.52, False),  # muimg: 0.04%, CI: 1.38%
]

MUIMG_DEFAULT_THRESHOLD = 0.15  # Fallback for unknown files
CI_DEFAULT_THRESHOLD = 2.75  # Fallback for unknown files


def get_dng_files():
    """Get list of DNG files for parametrized tests."""
    files = []
    if TEST_FILES_DIR.exists():
        files.extend(TEST_FILES_DIR.glob("*.dng"))
    if LOCAL_TEST_FILES_DIR.exists():
        files.extend(LOCAL_TEST_FILES_DIR.glob("*.dng"))
    return sorted(files)


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for comparison files."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    return OUTPUT_DIR


# Create lookup dictionaries from TEST_CASES
TEST_CASES_DICT = {filename: (muimg_threshold, muimg_xfail, ci_threshold, ci_xfail) 
                   for filename, muimg_threshold, muimg_xfail, ci_threshold, ci_xfail in TEST_CASES}


@pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
def test_muimg_vs_dngvalidate(dng_path, output_dir):
    """Test MUIMG rendering against C++ DNG SDK reference."""
    # Get test case parameters
    if dng_path.name in TEST_CASES_DICT:
        muimg_threshold, muimg_xfail, _, _ = TEST_CASES_DICT[dng_path.name]
    else:
        muimg_threshold, muimg_xfail = MUIMG_DEFAULT_THRESHOLD, False
    
    # Mark as xfail if specified
    if muimg_xfail:
        pytest.xfail(f"MUIMG rendering known to fail for {dng_path.name}")
    
    output_base = output_dir / f"{dng_path.stem}_dngvalidate"
    ref = run_dng_validate(dng_path, output_base, timeout=60)
    if ref is None:
        pytest.skip("dng_validate reference not available")
    
    t0 = time.perf_counter()
    with muimg.DngFile(dng_path) as dng:
        result = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            strict=False,
            use_xmp=False,  # Match dng_validate behavior (no XMP processing)
            rendering_params={'highlight_preserving_exposure': False},  # Use DNG SDK exposure behavior
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert result is not None, "MUIMG returned None"
    tifffile.imwrite(str(output_dir / f"{dng_path.stem}_muimg.tiff"), result)
    
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: MUIMG {result.shape} vs REF {ref.shape}")
    
    stats = compute_diff_stats(result, ref)
    threshold = muimg_threshold if muimg_threshold is not None else MUIMG_DEFAULT_THRESHOLD
    print(f"\n  [MUIMG] {dng_path.name}: {elapsed_ms:.0f}ms, diff:{stats['mean']:.2f}% (threshold:{threshold}%)")
    
    assert stats["mean"] < threshold, f"Mean diff {stats['mean']:.2f}% > {threshold}%"


@pytest.mark.skipif(
    not core_image_available_for_tests(), reason="Core Image not available"
)
@pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
def test_coreimage_vs_dngvalidate(dng_path, output_dir):
    """Test Core Image against C++ DNG SDK reference."""
    # Get test case parameters
    if dng_path.name in TEST_CASES_DICT:
        _, _, ci_threshold, ci_xfail = TEST_CASES_DICT[dng_path.name]
    else:
        ci_threshold, ci_xfail = CI_DEFAULT_THRESHOLD, False
    
    output_base = output_dir / f"{dng_path.stem}_dngvalidate"
    ref = run_dng_validate(dng_path, output_base, timeout=60)
    if ref is None:
        pytest.skip("dng_validate reference not available")
    
    t0 = time.perf_counter()
    try:
        result = muimg.decode_dng(
            file=str(dng_path),
            output_dtype=np.uint16,
            use_coreimage_if_available=True,
            use_xmp=False,  # Match dng_validate (no XMP), but still do color conversion
        )
    except Exception as e:
        if ci_xfail:
            pytest.xfail(f"Known Core Image failure for {dng_path.name}: {e}")
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert result is not None, "Core Image returned None"
    tifffile.imwrite(str(output_dir / f"{dng_path.stem}_coreimage.tiff"), result)
    
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: CI {result.shape} vs REF {ref.shape}")
    
    stats = compute_diff_stats(result, ref)
    threshold = ci_threshold if ci_threshold is not None else CI_DEFAULT_THRESHOLD
    print(f"\n  [CI] {dng_path.name}: {elapsed_ms:.0f}ms, diff:{stats['mean']:.2f}%")

    if ci_xfail and stats["mean"] >= threshold:
        pytest.xfail(
            f"Known Core Image mismatch for {dng_path.name}: mean diff {stats['mean']:.2f}% > {threshold}%"
        )

    assert stats["mean"] < threshold, f"Mean diff {stats['mean']:.2f}% > {threshold}%"


# Per-file thresholds for stripped DNG test (1.1x above measured values)
STRIPPED_THRESHOLDS = {
    "asi676mc.cfa.jxl_lossy.1ifds.dng": 0.08,  # measured 0.07%
    "asi676mc.linearraw.jxl_lossy.1ifds.dng": 0.19,  # measured 0.17%
    "iphone16_1_back_camera.linearraw.jxl_lossy.1ifds.dng": 0.13,  # measured 0.11%
}
STRIPPED_DEFAULT_THRESHOLD = 0.05


# Tags to strip for simplified DNG comparison test
# These are advanced processing features that add complexity to comparison
STRIPPED_TAGS = {
    # ProfileHueSatMap (color adjustments)
    'ProfileHueSatMapDims', 'ProfileHueSatMapData1', 'ProfileHueSatMapData2',
    # ProfileLookTable
    'ProfileLookTableDims', 'ProfileLookTableData',
    # ProfileGainTableMap (PGTM)
    'ProfileGainTableMap', 'ProfileGainTableMap2',
    # Baseline exposure
    'BaselineExposure', 'BaselineExposureOffset',
    # Opcode lists
    'OpcodeList1', 'OpcodeList2', 'OpcodeList3',
    # Orientation (rotation)
    'Orientation',
}


def get_non_sdk_dng_files():
    """Get list of non-SDK DNG files for stripped tag tests."""
    files = []
    if TEST_FILES_DIR.exists():
        for f in TEST_FILES_DIR.glob("*.dng"):
            if not f.name.startswith("dngsdk."):
                files.append(f)
    if LOCAL_TEST_FILES_DIR.exists():
        for f in LOCAL_TEST_FILES_DIR.glob("*.dng"):
            if not f.name.startswith("dngsdk."):
                files.append(f)
    return sorted(files)


@pytest.mark.parametrize("dng_path", get_non_sdk_dng_files(), ids=lambda p: p.name)
def test_stripped_dng_comparison(dng_path, output_dir):
    """Test muimg vs dng_validate on stripped DNGs (no advanced processing tags).
    
    Creates a stripped copy of each DNG without HueSatMap, LookTable, PGTM,
    BaselineExposure, and Opcode tags, then compares muimg output to dng_validate.
    This tests the core demosaic and color pipeline without advanced features.
    
    Also compares stripped output to original when no tags were actually stripped,
    to catch data corruption during copy (e.g., byte order issues).
    """
    stripped_dng = output_dir / f"{dng_path.stem}_stripped.dng"
    
    # Create stripped DNG and track which tags were actually stripped
    tags_stripped = set()
    with muimg.DngFile(dng_path) as dng:
        page = dng.get_main_page()
        if page is None:
            pytest.skip(f"No main page found in {dng_path.name}")
        
        # Check which tags exist in the original
        for tag in page.tags.values():
            if tag.name in STRIPPED_TAGS:
                tags_stripped.add(tag.name)
        if page.ifd0 is not None:
            for tag in page.ifd0.tags.values():
                if tag.name in STRIPPED_TAGS:
                    tags_stripped.add(tag.name)
        
        muimg.write_dng(
            destination_file=stripped_dng,
            subifds=[IfdSpec(data=page)],
            skip_tags=STRIPPED_TAGS,
        )
    
    # Run dng_validate on stripped DNG
    ref_base = output_dir / f"{dng_path.stem}_stripped_dngvalidate"
    ref = run_dng_validate(stripped_dng, ref_base, timeout=60)
    if ref is None:
        pytest.skip("dng_validate reference not available")
    
    # Run muimg on stripped DNG
    t0 = time.perf_counter()
    with muimg.DngFile(stripped_dng) as dng:
        result = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            strict=False,
            use_xmp=False,  # Match dng_validate behavior (no XMP processing)
            rendering_params={'highlight_preserving_exposure': False},  # Use DNG SDK exposure behavior
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert result is not None, "MUIMG returned None on stripped DNG"
    tifffile.imwrite(str(output_dir / f"{dng_path.stem}_stripped_muimg.tiff"), result)
    
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: MUIMG {result.shape} vs REF {ref.shape}")
    
    stats = compute_diff_stats(result, ref)
    threshold = STRIPPED_THRESHOLDS.get(dng_path.name, STRIPPED_DEFAULT_THRESHOLD)
    tags_info = f", stripped: {sorted(tags_stripped)}" if tags_stripped else ", no tags stripped"
    print(f"\n  [STRIPPED] {dng_path.name}: {elapsed_ms:.0f}ms, diff:{stats['mean']:.2f}% (threshold:{threshold}%){tags_info}")
    
    assert stats["mean"] < threshold, f"Mean diff {stats['mean']:.2f}% > {threshold}%"
    
    # If no tags were stripped, also compare to original file decode
    # This catches data corruption during copy (e.g., byte order issues)
    if not tags_stripped:
        with muimg.DngFile(dng_path) as dng:
            original_result = dng.render(
                output_dtype=np.uint16,
                demosaic_algorithm="DNGSDK_BILINEAR",
                strict=False,
                use_xmp=False,  # Match dng_validate behavior (no XMP processing)
                rendering_params={'highlight_preserving_exposure': False},  # Use DNG SDK exposure behavior
            )
        assert original_result is not None, "MUIMG returned None on original DNG"
        
        if result.shape == original_result.shape:
            orig_stats = compute_diff_stats(result, original_result)
            print(f"    -> vs original: diff:{orig_stats['mean']:.4f}%")
            # Should be nearly identical (only metadata differences, not pixel data)
            assert orig_stats["mean"] < 0.01, f"Stripped vs original diff {orig_stats['mean']:.4f}% > 0.01% - possible data corruption"
