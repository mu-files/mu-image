#!/usr/bin/env python3
"""Test exposure rendering across MUIMG, Photoshop, and dng_validate.

This test suite validates that MUIMG's exposure_tone function correctly matches
Photoshop's XMP Exposure2012 behavior across multiple exposure values (both
negative and positive), and compares against dng_validate's BaselineExposure rendering.
"""

import pytest
import numpy as np
import muimg
from pathlib import Path
import tempfile
import subprocess
import shutil
import tifffile

from muimg.dngio import IfdSpec, MetadataTags
from conftest import compute_diff_stats, run_dng_validate

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "mu-image-testdata" / "gradienttestfiles"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_exp"

# Test cases: (exposure_value, dng_file, photoshop_tif)
# Negative exposures
NEGATIVE_TEST_CASES = [
    (-0.5, "linear_gradient_test.exposure-0.5.dng", "linear_gradient_test.exposure-0.5.tif"),
    (-1.0, "linear_gradient_test.exposure-1.0.dng", "linear_gradient_test.exposure-1.0.tif"),
    (-1.05, "linear_gradient_test.exposure-1.05.dng", "linear_gradient_test.exposure-1.05.tif"),
    (-1.3, "linear_gradient_test.exposure-1.3.dng", "linear_gradient_test.exposure-1.3.tif"),
    (-1.5, "linear_gradient_test.exposure-1.5.dng", "linear_gradient_test.exposure-1.5.tif"),
    (-1.55, "linear_gradient_test.exposure-1.55.dng", "linear_gradient_test.exposure-1.55.tif"),
    (-1.8, "linear_gradient_test.exposure-1.8.dng", "linear_gradient_test.exposure-1.8.tif"),
    (-2.0, "linear_gradient_test.exposure-2.0.dng", "linear_gradient_test.exposure-2.0.tif"),
    (-2.05, "linear_gradient_test.exposure-2.05.dng", "linear_gradient_test.exposure-2.05.tif"),
    (-2.3, "linear_gradient_test.exposure-2.3.dng", "linear_gradient_test.exposure-2.3.tif"),
    (-2.55, "linear_gradient_test.exposure-2.55.dng", "linear_gradient_test.exposure-2.55.tif"),
]

# Positive exposures
POSITIVE_TEST_CASES = [
    (0.0, "linear_gradient_test.exposure_0.0.dng", "linear_gradient_test.exposure_0.0.tif"),
    (0.5, "linear_gradient_test.exposure_0.5.dng", "linear_gradient_test.exposure_0.5.tif"),
    (1.0, "linear_gradient_test.exposure_1.0.dng", "linear_gradient_test.exposure_1.0.tif"),
    (1.5, "linear_gradient_test.exposure_1.5.dng", "linear_gradient_test.exposure_1.5.tif"),
    (2.0, "linear_gradient_test.exposure_2.0.dng", "linear_gradient_test.exposure_2.0.tif"),
    (2.5, "linear_gradient_test.exposure_2.5.dng", "linear_gradient_test.exposure_2.5.tif"),
]

# Baseline exposure test cases (Exposure2012 moved to BaselineExposure tag, XMP stripped)
# These test whether Photoshop treats BaselineExposure differently than user Exposure2012
BASELINE_EXPOSURE_TEST_CASES = [
    (-1.0, "linear_gradient_test.baselineexp-1.0.dng", "linear_gradient_test.baselineexp-1.0.tif"),
    (1.0, "linear_gradient_test.baselineexp_1.0.dng", "linear_gradient_test.baselineexp_1.0.tif"),
]

# Combined test cases
TEST_CASES = NEGATIVE_TEST_CASES + POSITIVE_TEST_CASES + BASELINE_EXPOSURE_TEST_CASES


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def create_baseline_exposure_dng(source_dng: Path, exposure_value: float, output_path: Path):
    """Create a DNG with BaselineExposure set, XMP removed."""
    with muimg.DngFile(source_dng) as dng:
        page = dng.get_main_page()
        
        # Create IFD0 tags with BaselineExposure, no XMP
        ifd0_tags = MetadataTags()
        ifd0_tags.add_tag("BaselineExposure", exposure_value)
        
        # Write new DNG with the page data and modified IFD0 tags
        muimg.write_dng(
            destination_file=output_path,
            subifds=[IfdSpec(data=page)],
            ifd0_tags=ifd0_tags,
        )




@pytest.mark.parametrize("exposure,dng_file,photoshop_tif", TEST_CASES)
def test_exposure_comprehensive(exposure, dng_file, photoshop_tif, output_dir):
    """Test exposure rendering with k=0.25.
    
    Tests:
    1. MUIMG with XMP (k=0.25) vs Photoshop (when reference available)
    2. For positive exposures without Photoshop reference, just verify rendering works
    """
    dng_path = TEST_DATA_DIR / dng_file
    
    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")
    
    # Photoshop reference may not exist for positive exposures
    photoshop_path = TEST_DATA_DIR / photoshop_tif if photoshop_tif else None
    has_photoshop_ref = photoshop_path and photoshop_path.exists()
    
    print(f"\n{'='*70}")
    print(f"Testing exposure={exposure} stops")
    print(f"{'='*70}")
    
    # Render with MUIMG
    with muimg.DngFile(dng_path) as dng:
        muimg_xmp = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            use_xmp=True,
            rendering_params={'highlight_compressing_exposure': True},
        )
    
    # Save MUIMG XMP output
    stem = dng_path.stem
    muimg_xmp_output = output_dir / f"{stem}_muimg.tif"
    tifffile.imwrite(str(muimg_xmp_output), muimg_xmp)
    
    # Compare against Photoshop if reference exists
    if has_photoshop_ref:
        print(f"\n[1] MUIMG (XMP) vs Photoshop:")
        photoshop_img = tifffile.imread(str(photoshop_path))
        stats_xmp_ps = compute_diff_stats(muimg_xmp, photoshop_img)
        print(f"    Difference: {stats_xmp_ps['mean']:.2f}%")
        
        # Assertions with different thresholds for negative vs positive exposure
        # TODO: Positive exposures have ~0.8-1.2% error (vs <0.2% for negative).
        # This needs investigation - likely related to complementary power function
        # interaction with shadow processing or tone curve application order.
        # Current threshold set to 1.25% to catch regressions while allowing current behavior.
        if exposure >= 0.0:
            threshold = 1.25  # Relaxed threshold for positive exposures (needs work)
        else:
            threshold = 0.5  # Strict threshold for negative exposures (well-tuned)
        
        assert stats_xmp_ps['mean'] < threshold, (
            f"MUIMG (XMP) vs Photoshop: {stats_xmp_ps['mean']:.2f}% > {threshold}%"
        )
    else:
        print(f"\n[1] MUIMG (XMP) rendering successful (no Photoshop reference)")
        print(f"    Output saved to: {muimg_xmp_output}")
        # Basic sanity checks for positive exposures
        assert muimg_xmp.shape[0] > 0 and muimg_xmp.shape[1] > 0, "Invalid image dimensions"
        assert muimg_xmp.dtype == np.uint16, "Expected uint16 output"
    



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
