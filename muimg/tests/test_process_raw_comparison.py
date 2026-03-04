"""Tests for process_raw() and process_raw_core_image().

Compares MUIMG (our Python port) and Core Image against C++ DNG SDK reference (dng_validate).
"""

import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from muimg import color
from muimg.color_mac import core_image_available, process_raw_core_image
from conftest import TEST_FILES_DIR, LOCAL_TEST_FILES_DIR, OUTPUT_COMPARISON_DIR

# Output directory for comparison files
OUTPUT_DIR = Path(__file__).parent / "output_comparison"

# Path to the C++ SDK dng_validate tool (reference)
DNG_VALIDATE_PATH = Path.home() / "Projects/C/3dparty/dng_sdk_1_7_1/dng_sdk/targets/mac/release64/dng_validate"

# Comparison thresholds (as percentage of full range)
# Per-file thresholds for MUIMG (1.1x above measured values)
MUIMG_THRESHOLDS = {
    # Original test files
    "Sony.bayer.lossy.dng": 0.05,  # measured 0.04%
    "Sony.bayer.lossy.stripped.dng": 0.05,  # measured 0.04%
    "asi676mc.cfa.dng": 0.01,  # measured 0.01%
    "asi676mc.linearraw.dng": 0.13,  # measured 0.12%
    "asi676mc.lossless.preview1.dng": 0.01,  # measured 0.00%
    "asi676mc.nopreview.lossy.dng": 0.01,  # measured 0.00%
    "asi676mc.preview0.lossy.dng": 0.01,  # measured 0.00%
    "iphone.linearRGB.lossy.dng": 0.72,  # measured 0.65%
    "iphone.linearRGB.lossy.stripped.dng": 0.21,  # measured 0.19%
    # DNG SDK test files - JXL
    "dngsdk.01_jxl_linear_raw_integer.dng": 0.01,  # measured 0.01%
    "dngsdk.02_jxl_linear_raw_float.dng": 0.08,  # measured 0.07%
    "dngsdk.03_jxl_bayer_raw_integer.dng": 0.22,  # measured 0.20%
    # DNG SDK test files - PGTM2
    "dngsdk.04_PGTM2_per_profile.dng": 0.01,  # measured 0.00%
    "dngsdk.05_PGTM2_unsigned8.dng": 0.01,  # measured 0.00%
    "dngsdk.06_PGTM2_unsigned16.dng": 0.01,  # measured 0.00%
    "dngsdk.07_PGTM2_float16.dng": 0.01,  # measured 0.00%
    "dngsdk.08_PGTM2_float32.dng": 0.01,  # measured 0.00%
    # DNG SDK test files - ImageSequenceInfo
    "dngsdk.09_ImageSequenceInfo_1_of_3.dng": 0.01,  # measured 0.00%
    "dngsdk.10_ImageSequenceInfo_2_of_3.dng": 0.01,  # measured 0.00%
    "dngsdk.11_ImageSequenceInfo_3_of_3.dng": 0.01,  # measured 0.00%
    # DNG SDK test files - ImageStats
    "dngsdk.12_ImageStats_WeightedAverage.dng": 0.02,  # measured 0.01%
    "dngsdk.13_ImageStats_Several.dng": 0.02,  # measured 0.01%
    # DNG SDK test files - HDR/SDR
    "dngsdk.14_hdr_sdr_profiles.dng": 0.01,  # measured 0.00%
    # Canon
    "CanonR5.cfa.dng": 0.02,  # measured 0.02%
    "CanonR5-II.cfa.dng": 0.02,  # measured 0.02%
    # Sony
    "SonyRX100-VII.cfa.dng": 0.02,  # measured 0.01%
    "SonyRX100-VII.cfa.uncompressed.dng": 0.28,  # measured 0.25%
    # Insta360
    "Insta360.dng": 0.02,  # measured 0.01%
}
MUIMG_DEFAULT_THRESHOLD = 0.15  # Fallback for unknown files
CI_MEAN_DIFF_THRESHOLD = 2.75  # Core Image vs dng_validate: must be < 2.75%


def get_dng_files():
    """Get list of DNG files for parametrized tests."""
    files = []
    if TEST_FILES_DIR.exists():
        files.extend(TEST_FILES_DIR.glob("*.dng"))
    if LOCAL_TEST_FILES_DIR.exists():
        files.extend(LOCAL_TEST_FILES_DIR.glob("*.dng"))
    return sorted(files)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to float [0,1] range."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def compute_diff_stats(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compute difference statistics between two images."""
    diff = np.abs(normalize_image(img1) - normalize_image(img2))
    return {
        "mean": np.mean(diff) * 100,
        "p99": np.percentile(diff, 99) * 100,
        "max": np.max(diff) * 100,
    }


def load_tiff(path: Path) -> np.ndarray | None:
    """Load TIFF and convert to interleaved format if needed."""
    if path is None or not path.exists():
        return None
    try:
        with tifffile.TiffFile(str(path)) as tif:
            img = tif.pages[0].asarray()
            # Convert planar (3,H,W) to interleaved (H,W,3)
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[0] < img.shape[1]:
                img = np.moveaxis(img, 0, -1)
            return img
    except Exception:
        return None


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for comparison files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def generate_reference(dng_path: Path, output_dir: Path) -> np.ndarray | None:
    """Generate C++ SDK reference TIFF for a single DNG file."""
    if not DNG_VALIDATE_PATH.exists():
        return None
    
    output_base = output_dir / f"{dng_path.stem}_dngvalidate"
    output_tiff = output_dir / f"{dng_path.stem}_dngvalidate.tif"
    
    try:
        subprocess.run(
            [str(DNG_VALIDATE_PATH), "-v", "-16", "-tif", str(output_base), str(dng_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
    except (subprocess.TimeoutExpired, Exception):
        return None
    
    return load_tiff(output_tiff)


@pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
def test_muimg_vs_dngvalidate(dng_path, output_dir):
    """Test MUIMG process_raw() against C++ DNG SDK reference."""
    ref = generate_reference(dng_path, output_dir)
    if ref is None:
        pytest.skip("dng_validate reference not available")
    
    t0 = time.perf_counter()
    result = color.process_raw(
        str(dng_path), 
        output_dtype=np.uint16, 
        algorithm="DNGSDK_BILINEAR", 
        strict=False
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert result is not None, "MUIMG returned None"
    tifffile.imwrite(str(output_dir / f"{dng_path.stem}_muimg.tiff"), result)
    
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: MUIMG {result.shape} vs REF {ref.shape}")
    
    stats = compute_diff_stats(result, ref)
    threshold = MUIMG_THRESHOLDS.get(dng_path.name, MUIMG_DEFAULT_THRESHOLD)
    print(f"\n  [MUIMG] {dng_path.name}: {elapsed_ms:.0f}ms, diff:{stats['mean']:.2f}% (threshold:{threshold}%)")
    
    assert stats["mean"] < threshold, f"Mean diff {stats['mean']:.2f}% > {threshold}%"


@pytest.mark.skipif(not core_image_available, reason="Core Image not available")
@pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
def test_coreimage_vs_dngvalidate(dng_path, output_dir):
    """Test Core Image against C++ DNG SDK reference."""
    ref = generate_reference(dng_path, output_dir)
    if ref is None:
        pytest.skip("dng_validate reference not available")
    
    t0 = time.perf_counter()
    result = process_raw_core_image(str(dng_path), output_dtype=np.uint16)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert result is not None, "Core Image returned None"
    tifffile.imwrite(str(output_dir / f"{dng_path.stem}_coreimage.tiff"), result)
    
    if result.shape != ref.shape:
        pytest.fail(f"Shape mismatch: CI {result.shape} vs REF {ref.shape}")
    
    stats = compute_diff_stats(result, ref)
    print(f"\n  [CI] {dng_path.name}: {elapsed_ms:.0f}ms, diff:{stats['mean']:.2f}%")
    
    assert stats["mean"] < CI_MEAN_DIFF_THRESHOLD, f"Mean diff {stats['mean']:.2f}% > {CI_MEAN_DIFF_THRESHOLD}%"
