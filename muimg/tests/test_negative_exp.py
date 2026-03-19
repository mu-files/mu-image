#!/usr/bin/env python3
"""Test negative exposure rendering across MUIMG, Photoshop, and dng_validate.

This test suite validates that MUIMG's exposure_tone function correctly matches
Photoshop's XMP Exposure2012 behavior across multiple negative exposure values,
and compares against dng_validate's BaselineExposure rendering.
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
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_negative_exp"

# Test cases: (exposure_value, dng_file, photoshop_tif)
TEST_CASES = [
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
def test_negative_exposure_comprehensive(exposure, dng_file, photoshop_tif, output_dir):
    """Test negative exposure rendering with k=0.25.
    
    Tests:
    1. MUIMG with XMP (k=0.25) vs Photoshop
    2. Optional: Compare MUIMG vs dng_validate (k=1.0) to show difference
    """
    dng_path = TEST_DATA_DIR / dng_file
    photoshop_path = TEST_DATA_DIR / photoshop_tif
    
    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")
    if not photoshop_path.exists():
        pytest.skip(f"Photoshop reference not found: {photoshop_path}")
    
    print(f"\n{'='*70}")
    print(f"Testing exposure={exposure} stops")
    print(f"{'='*70}")
    
    # Load Photoshop reference
    photoshop_img = tifffile.imread(str(photoshop_path))
    
    # Test 1: MUIMG with XMP vs Photoshop
    print(f"\n[1] MUIMG (XMP) vs Photoshop:")
    with muimg.DngFile(dng_path) as dng:
        muimg_xmp = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            use_xmp=True
        )
    
    stats_xmp_ps = compute_diff_stats(muimg_xmp, photoshop_img)
    print(f"    Difference: {stats_xmp_ps['mean']:.2f}%")
    
    # Save MUIMG XMP output
    stem = dng_path.stem
    muimg_xmp_output = output_dir / f"{stem}_muimg.tif"
    tifffile.imwrite(str(muimg_xmp_output), muimg_xmp)
    
    # Assertions
    threshold = 0.5
    assert stats_xmp_ps['mean'] < threshold, (
        f"MUIMG (XMP) vs Photoshop: {stats_xmp_ps['mean']:.2f}% > {threshold}%"
    )
    



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
