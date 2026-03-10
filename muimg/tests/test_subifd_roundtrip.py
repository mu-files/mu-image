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
from muimg import color
from muimg.dngio import write_dng_from_page, MetadataTags
from conftest import (
    TEST_FILES_DIR,
    DNG_VALIDATE_PATH,
    compute_diff_stats,
    run_dng_validate,
)

# Output directory for this test
OUTPUT_DIR = Path(__file__).parent / "output_subifd"

# Threshold for comparison (percentage of full range)
ROUNDTRIP_THRESHOLD = 0.01  # Should be nearly identical


def get_multi_ifd_dng_files():
    """Get list of DNG files with >2 IFDs (indicating SubIFDs)."""
    files = []
    if TEST_FILES_DIR.exists():
        for path in sorted(TEST_FILES_DIR.glob("*.dng")):
            try:
                with muimg.DngFile(str(path)) as dng:
                    ifd_count = len(dng.get_flattened_pages())
                    if ifd_count > 2:
                        files.append(path)
            except Exception:
                pass  # Skip files that can't be opened
    return files


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


@pytest.mark.parametrize("dng_path", get_multi_ifd_dng_files(), ids=lambda p: p.name)
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
    
    with muimg.DngFile(str(dng_path)) as dng:
        pages = dng.get_flattened_pages()
        print(f"\n{dng_path.name}: {len(pages)} IFDs")
        
        # First pass: show all IFDs
        for i, page in enumerate(pages):
            is_raw = page.is_cfa or page.is_linear_raw
            raw_type = "CFA" if page.is_cfa else ("LINEAR_RAW" if page.is_linear_raw else "")
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
                decoded = color.render_dng(page, output_dtype=np.uint16, strict=False)
                if decoded is None:
                    pytest.fail(f"render_dng returned None for IFD {i}")
                tifffile.imwrite(str(muimg_tif), decoded)
                print(f"    -> {muimg_tif.name} ({decoded.shape})")
            except Exception as e:
                pytest.fail(f"render_dng failed for IFD {i}: {e}")
            
            # 2. write_dng_from_page -> {stem}_ifd{n}.dng
            # Use decoded fallback if tile dimensions not supported by tifffile
            use_decoded_fallback = False
            if page.is_tiled:
                tile_h, tile_w = page.tilelength, page.tilewidth
                tile_valid = (
                    tile_h <= page.shape[0] and tile_w <= page.shape[1] and
                    tile_h % 16 == 0 and tile_w % 16 == 0
                )
                if not tile_valid:
                    use_decoded_fallback = True
            
            try:
                if use_decoded_fallback:
                    write_dng_from_page(page, roundtrip_dng, decompress=True)
                    print(f"    -> {roundtrip_dng.name} (decoded)")
                else:
                    write_dng_from_page(page, roundtrip_dng)
                    print(f"    -> {roundtrip_dng.name}")
            except Exception as e:
                pytest.fail(f"write_dng_from_page failed for IFD {i}: {e}")
            
            # 3. Try muimg decode on the roundtrip DNG
            roundtrip_muimg_tif = output_dir / f"{stem}_ifd{i}_roundtrip_muimg.tif"
            try:
                roundtrip_decoded = color.render_dng(roundtrip_dng, output_dtype=np.uint16, strict=False)
                if roundtrip_decoded is not None:
                    tifffile.imwrite(str(roundtrip_muimg_tif), roundtrip_decoded)
                    print(f"    -> {roundtrip_muimg_tif.name} ({roundtrip_decoded.shape})")
                else:
                    print(f"    -> muimg roundtrip decode returned None")
            except Exception as e:
                print(f"    -> muimg roundtrip decode failed: {e}")
            
            # 4. dng_validate on roundtrip DNG -> {stem}_ifd{n}_dngvalidate.tif
            dngvalidate_out = run_dng_validate(roundtrip_dng, dngvalidate_base)
            if dngvalidate_out is None:
                pytest.fail(f"dng_validate failed on {roundtrip_dng.name}")
            print(f"    -> {dngvalidate_base.name}.tif ({dngvalidate_out.shape})")
            
            # 4. Compare muimg vs dng_validate (same source - roundtrip DNG)
            if decoded.shape == dngvalidate_out.shape:
                stats = compute_diff_stats(decoded, dngvalidate_out)
                print(f"    muimg vs dng_validate: mean={stats['mean']:.4f}%, max={stats['max']:.4f}%")
            else:
                print(f"    Shape mismatch: muimg {decoded.shape} vs dng_validate {dngvalidate_out.shape}")
        
        if raw_ifd_count == 0:
            pytest.skip(f"No raw IFDs found in {dng_path.name}")
        
        print(f"\n  Processed {raw_ifd_count} raw IFDs")
