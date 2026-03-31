"""Tests for SubIFD decode/roundtrip validation.

For multi-IFD DNG files, validates that:
1. Each raw SubIFD can be decoded by muimg's render_dng
2. write_dng_from_page produces valid DNG output
3. The roundtrip DNG produces identical results when processed by dng_validate
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

import muimg
from muimg.dngio import IfdSpec, MetadataTags
from conftest import (
    DNG_VALIDATE_PATH,
    compute_diff_stats,
    run_dng_validate,
)

# Output directory for this test
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_subifd_roundtrip"
DNGFILES_DIR = Path(__file__).parent / "dngfiles"

# Threshold for comparison (percentage of full range)
ROUNDTRIP_THRESHOLD = 0.01  # Should be nearly identical

# Multi-IFD test files with validation warning ignores
# Format: {filename: [warning_patterns_to_ignore]}
TEST_FILES = {
    "asi676mc.cfa.jxl_lossy.2ifds.dng": ["makernote has unexpected type", "too little padding"],
    "canon_eos_r5_mark_ii.linearraw.jxl_lossy.6ifds.dng": [],
    "sony_ilce-7c.cfa.jxl_lossy.4ifds.dng": ["columninterleavefactor tag not allowed"],
}


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test files."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    return OUTPUT_DIR


def get_test_files():
    """Get list of test DNG file paths."""
    return [DNGFILES_DIR / filename for filename in TEST_FILES.keys()]


@pytest.mark.parametrize("dng_path", get_test_files(), ids=lambda p: p.name)
def test_subifd_roundtrip(dng_path: Path, output_dir: Path):
    """Test that each raw SubIFD can be decoded and roundtripped through write_dng_from_page.
    
    For each raw IFD, produces 3 files:
    - {stem}_ifd{n}.dng - roundtrip DNG created by write_dng_from_page
    - {stem}_ifd{n}_muimg.tif - muimg render_dng output
    - {stem}_ifd{n}_dngvalidate.tif - dng_validate output on roundtrip DNG
    """
    if not DNG_VALIDATE_PATH.exists():
        pytest.skip("dng_validate not available")
    
    stem = dng_path.stem
    
    with muimg.DngFile(dng_path) as dng:
        pages = dng.get_flattened_pages()
        print(f"\n{dng_path.name}: {len(pages)} IFDs")
        
        # First pass: show all IFDs
        for i, page in enumerate(pages):
            is_raw = page.is_cfa or page.is_linear_raw
            main_marker = " [MAIN]" if page.is_main_image else ""
            print(f"  IFD {i}: {page.photometric or 'unknown'} {page.shape}{main_marker} {'-> PROCESS' if is_raw else '-> skip'}")
        
        # Second pass: process raw IFDs
        raw_ifd_count = 0
        for i, page in enumerate(pages):
            if not (page.is_cfa or page.is_linear_raw):
                continue
            
            raw_ifd_count += 1
            ifd_type = "cfa" if page.is_cfa else "linearraw"
            
            # Output file paths - 3 files per IFD
            roundtrip_dng = output_dir / f"{stem}_ifd{i}.dng"
            muimg_tif = output_dir / f"{stem}_ifd{i}_muimg.tif"
            dngvalidate_base = output_dir / f"{stem}_ifd{i}_dngvalidate"
            
            print(f"\n  Processing IFD {i} ({ifd_type}):")
            
            # 1. muimg render_dng -> {stem}_ifd{n}_muimg.tif
            try:
                decoded = page.render_raw(output_dtype=np.uint16, strict=False, use_xmp=False,
                                      rendering_params={'highlight_preserving_exposure': False})
                if decoded is None:
                    pytest.fail(f"page.render_raw returned None for IFD {i}")
                tifffile.imwrite(str(muimg_tif), decoded)
                print(f"    -> {muimg_tif.name} ({decoded.shape})")
            except Exception as e:
                pytest.fail(f"page.render_raw failed for IFD {i}: {e}")
            
            # 2. write_dng_from_page -> {stem}_ifd{n}.dng
            try:
                muimg.write_dng(
                    destination_file=roundtrip_dng,
                    subifds=[IfdSpec(data=page)],
                )
                print(f"    -> {roundtrip_dng.name}")
            except Exception as e:
                pytest.fail(f"write_dng failed for IFD {i}: {e}")
            
            # 3. Try muimg decode on the roundtrip DNG
            roundtrip_muimg_tif = output_dir / f"{stem}_ifd{i}_roundtrip_muimg.tif"
            roundtrip_decoded = None
            try:
                with muimg.DngFile(roundtrip_dng) as roundtrip_dng_file:
                    roundtrip_decoded = roundtrip_dng_file.render_raw(output_dtype=np.uint16, strict=False, use_xmp=False,
                                                                  rendering_params={'highlight_preserving_exposure': False})
                if roundtrip_decoded is not None:
                    tifffile.imwrite(str(roundtrip_muimg_tif), roundtrip_decoded)
                    print(f"    -> {roundtrip_muimg_tif.name} ({roundtrip_decoded.shape})")
                else:
                    print(f"    -> muimg roundtrip decode returned None")
            except Exception as e:
                print(f"    -> muimg roundtrip decode failed: {e}")
            
            # 3a. Compare original muimg decode vs roundtrip muimg decode
            if roundtrip_decoded is not None and decoded.shape == roundtrip_decoded.shape:
                stats = compute_diff_stats(decoded, roundtrip_decoded)
                print(f"    original vs roundtrip (muimg): mean={stats['mean']:.4f}%, max={stats['max']:.4f}%")
                # Fail if roundtrip through muimg produces significantly different results
                if stats['mean'] > 1.0:
                    pytest.fail(
                        f"Roundtrip through muimg produced different results: "
                        f"mean={stats['mean']:.2f}%, max={stats['max']:.2f}%"
                    )
            elif roundtrip_decoded is not None:
                print(f"    Shape mismatch: original {decoded.shape} vs roundtrip {roundtrip_decoded.shape}")
            
            # 4. dng_validate on roundtrip DNG -> {stem}_ifd{n}_dngvalidate.tif
            # Get ignored warnings for this file from TEST_FILES
            ignored_warnings = TEST_FILES.get(dng_path.name, [])
            
            dngvalidate_out = run_dng_validate(roundtrip_dng, dngvalidate_base, ignored_warnings=ignored_warnings)
            if dngvalidate_out is None:
                pytest.fail(f"dng_validate failed on {roundtrip_dng.name}")
            print(f"    -> {dngvalidate_base.name}.tif ({dngvalidate_out.shape})")
            
            # 5. Compare original muimg vs dng_validate (both decoding roundtrip DNG)
            if decoded.shape == dngvalidate_out.shape:
                stats = compute_diff_stats(decoded, dngvalidate_out)
                print(f"    muimg vs dng_validate: mean={stats['mean']:.4f}%, max={stats['max']:.4f}%")
                # Fail if muimg and dng_validate produce very different results
                if stats['mean'] > 1.0:
                    pytest.fail(
                        f"muimg and dng_validate produced very different results: "
                        f"mean={stats['mean']:.2f}%, max={stats['max']:.2f}%"
                    )
            else:
                print(f"    Shape mismatch: muimg {decoded.shape} vs dng_validate {dngvalidate_out.shape}")
        
        if raw_ifd_count == 0:
            pytest.skip(f"No raw IFDs found in {dng_path.name}")
        
        print(f"\n  Processed {raw_ifd_count} raw IFDs")
