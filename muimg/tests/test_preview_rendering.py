# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Tests for DNG preview page rendering and decoding.

Validates that:
1. Preview pages (RGB/YCBCR and LINEAR_RAW) match the main rendered image at their resolutions
2. API correctly raises errors when methods are called on inappropriate page types
"""
import pytest
import numpy as np
from pathlib import Path
import cv2
import tifffile

import muimg
from conftest import compute_diff_stats

# Test data directory
DNGFILES_DIR = Path(__file__).parent / "dngfiles"
TESTDATA_DIR = Path.home() / "Projects/python/mu-image-testdata/dngtestfiles"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_preview_rendering"

# Threshold table for canon_eos_r5_cfa_ljpeg_6ifds.dng
# Baseline measurements * 1.02 for regression detection
# 
# LINEAR_RAW previews vs main render:
#   IFD 3 (2048x1366): mean 0.637%, p99 5.138%, max 32.676%
#   IFD 4 (512x342):   mean 0.583%, p99 5.768%, max 25.960%
#   IFD 5 (256x171):   mean 0.716%, p99 7.994%, max 19.792%
# 
# RGB/YCBCR previews vs each other:
#   IFD 0 vs IFD 2:    mean 0.504%, p99 2.939%, max 11.032%
THRESHOLDS = {
    'linear_raw_vs_main': {
        'mean': 0.73,    # max observed 0.716% * 1.02
        'p99': 8.315,    # max observed 7.994% * 1.02
        'max': 33.33,    # max observed 32.676% * 1.02
    },
    'rgb_ycbcr_vs_each_other': {
        'mean': 0.514,   # observed 0.504% * 1.02
        'p99': 2.998,    # observed 2.939% * 1.02
        'max': 11.253,   # observed 11.032% * 1.02
    }
}


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_resolution_pyramid(image: np.ndarray, min_dimension: int = 128) -> list[np.ndarray]:
    """Build resolution pyramid by progressively downsampling by 2x.
    
    Args:
        image: Source image to downsample
        min_dimension: Stop when smallest dimension reaches this size
        
    Returns:
        List of images at progressively smaller resolutions, starting with original
    """
    pyramid = [image]
    current = image
    
    while True:
        h, w = current.shape[:2]
        if min(h, w) <= min_dimension:
            break
        
        # Downsample by 2x using INTER_AREA (best for downsampling)
        next_level = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        pyramid.append(next_level)
        current = next_level
    
    return pyramid


def find_or_create_pyramid_level(pyramid: list[np.ndarray], target_shape: tuple[int, int]) -> np.ndarray:
    """Find pyramid level matching target shape, or create it by resizing.
    
    Args:
        pyramid: List of pyramid levels (largest to smallest)
        target_shape: (height, width) to match
        
    Returns:
        Image at target resolution
    """
    target_h, target_w = target_shape
    
    # Check if we have exact match in pyramid
    for level in pyramid:
        if level.shape[:2] == (target_h, target_w):
            return level
    
    # No exact match, resize from largest level that's bigger
    source = pyramid[0]  # Start with full resolution
    for level in pyramid:
        if level.shape[0] >= target_h and level.shape[1] >= target_w:
            source = level
        else:
            break
    
    # Resize to exact target size
    return cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA)


def test_preview_pages_match_main_render(output_dir: Path):
    """Test that preview pages match downsampled main render at their resolutions."""
    
    test_file = DNGFILES_DIR / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    
    if not test_file.exists():
        pytest.skip(f"Test file not available: {test_file}")
    
    # Create output subdirectory for this test
    test_output_dir = output_dir / test_file.stem
    test_output_dir.mkdir(exist_ok=True)
    
    print(f"\nTesting preview rendering with: {test_file.name}")
    
    with muimg.DngFile(test_file) as dng:
        pages = dng.get_flattened_pages()
        main_page = dng.get_main_page()
        
        assert main_page is not None, "No main page found"
        
        # Step 1: Render main page at full resolution
        print("\n1. Rendering main raw page...")
        main_render = dng.render_raw(output_dtype=np.uint16, strict=False)
        assert main_render is not None, "Failed to render main page"
        
        print(f"   Main render: {main_render.shape}, dtype={main_render.dtype}")
        
        # Save main render
        tifffile.imwrite(str(test_output_dir / "main_render.tif"), main_render)
        
        # Step 2: Build resolution pyramid
        print("\n2. Building resolution pyramid...")
        pyramid = build_resolution_pyramid(main_render, min_dimension=128)
        
        print(f"   Pyramid levels: {len(pyramid)}")
        for i, level in enumerate(pyramid):
            print(f"     Level {i}: {level.shape}")
            tifffile.imwrite(str(test_output_dir / f"pyramid_level_{i}.tif"), level)
        
        # Step 3: Categorize preview pages
        print("\n3. Categorizing preview pages...")
        linear_raw_previews = []
        rgb_ycbcr_previews = []
        
        for i, page in enumerate(pages):
            if page.is_main_image:
                continue
            
            if page.is_linear_raw:
                linear_raw_previews.append((i, page))
                print(f"   IFD {i}: LINEAR_RAW preview - {page.imagewidth}x{page.imagelength}")
            elif page.photometric_name in ("RGB", "YCBCR"):
                rgb_ycbcr_previews.append((i, page))
                print(f"   IFD {i}: {page.photometric_name} preview - {page.imagewidth}x{page.imagelength}")
        
        # Step 4: Compare LINEAR_RAW previews against main render
        print(f"\n4. Comparing {len(linear_raw_previews)} LINEAR_RAW preview(s) against main render...")
        
        for i, page in linear_raw_previews:
            size = f"{page.imagewidth}x{page.imagelength}"
            print(f"\n   IFD {i}: LINEAR_RAW, {size}")
            
            try:
                # Decode LINEAR_RAW preview
                preview = page.render_raw(output_dtype=np.uint16, strict=False)
                assert preview is not None, f"render_raw returned None for IFD {i}"
                
                print(f"     ✓ Decoded: {preview.shape}, dtype={preview.dtype}")
                
                # Save preview
                preview_path = test_output_dir / f"ifd{i}_linear_raw_preview.tif"
                tifffile.imwrite(str(preview_path), preview)
                
                # Find matching pyramid level
                expected = find_or_create_pyramid_level(pyramid, preview.shape[:2])
                
                # Save expected
                expected_path = test_output_dir / f"ifd{i}_linear_raw_expected.tif"
                tifffile.imwrite(str(expected_path), expected)
                
                # Compute difference statistics
                stats = compute_diff_stats(preview, expected)
                
                print(f"     Comparison vs downsampled main render:")
                print(f"       Mean diff:  {stats['mean']:.3f}%")
                print(f"       P99 diff:   {stats['p99']:.3f}%")
                print(f"       Max diff:   {stats['max']:.3f}%")
                
                # Assert thresholds (baseline * 1.02)
                thresh = THRESHOLDS['linear_raw_vs_main']
                assert stats['mean'] < thresh['mean'], \
                    f"Mean diff {stats['mean']:.3f}% exceeds threshold {thresh['mean']:.3f}%"
                assert stats['p99'] < thresh['p99'], \
                    f"P99 diff {stats['p99']:.3f}% exceeds threshold {thresh['p99']:.3f}%"
                assert stats['max'] < thresh['max'], \
                    f"Max diff {stats['max']:.3f}% exceeds threshold {thresh['max']:.3f}%"
                
                # Save difference visualization
                diff = np.abs(preview.astype(np.float32) - expected.astype(np.float32))
                diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
                diff_path = test_output_dir / f"ifd{i}_linear_raw_diff.tif"
                tifffile.imwrite(str(diff_path), diff_vis)
                
                print(f"     ✓ Passed thresholds")
                
            except Exception as e:
                pytest.fail(f"LINEAR_RAW preview comparison failed for IFD {i}: {e}")
        
        # Step 5: Compare RGB/YCBCR previews against each other
        print(f"\n5. Processing {len(rgb_ycbcr_previews)} RGB/YCBCR preview(s)...")
        
        # Decode all RGB/YCBCR previews
        decoded_previews = []
        for i, page in rgb_ycbcr_previews:
            size = f"{page.imagewidth}x{page.imagelength}"
            print(f"\n   IFD {i}: {page.photometric_name}, {size}")
            
            try:
                preview = page.decode_to_rgb(output_dtype=np.uint16)
                assert preview is not None, f"decode_to_rgb returned None for IFD {i}"
                
                print(f"     ✓ Decoded: {preview.shape}, dtype={preview.dtype}")
                
                # Save preview
                preview_path = test_output_dir / f"ifd{i}_{page.photometric_name.lower()}_preview.tif"
                tifffile.imwrite(str(preview_path), preview)
                
                decoded_previews.append((i, page, preview))
                
            except Exception as e:
                pytest.fail(f"RGB/YCBCR preview decoding failed for IFD {i}: {e}")
        
        # Compare previews against each other if multiple exist
        if len(decoded_previews) > 1:
            print(f"\n   Comparing RGB/YCBCR previews against each other...")
            
            # Find largest preview
            largest_idx = max(range(len(decoded_previews)), 
                            key=lambda idx: decoded_previews[idx][2].shape[0] * decoded_previews[idx][2].shape[1])
            largest_ifd, largest_page, largest_preview = decoded_previews[largest_idx]
            
            print(f"     Using IFD {largest_ifd} as reference (largest at {largest_preview.shape[1]}x{largest_preview.shape[0]})")
            
            # Compare each smaller preview against downsampled largest
            for i, page, preview in decoded_previews:
                if i == largest_ifd:
                    continue
                
                print(f"\n     Comparing IFD {i} vs IFD {largest_ifd}:")
                
                # Downsample largest to match this preview's size
                expected = cv2.resize(largest_preview, 
                                    (preview.shape[1], preview.shape[0]), 
                                    interpolation=cv2.INTER_AREA)
                
                # Save expected
                expected_path = test_output_dir / f"ifd{i}_vs_ifd{largest_ifd}_expected.tif"
                tifffile.imwrite(str(expected_path), expected)
                
                # Compute difference statistics
                stats = compute_diff_stats(preview, expected)
                
                print(f"       Mean diff:  {stats['mean']:.3f}%")
                print(f"       P99 diff:   {stats['p99']:.3f}%")
                print(f"       Max diff:   {stats['max']:.3f}%")
                
                # Assert thresholds (baseline * 1.02)
                thresh = THRESHOLDS['rgb_ycbcr_vs_each_other']
                assert stats['mean'] < thresh['mean'], \
                    f"Mean diff {stats['mean']:.3f}% exceeds threshold {thresh['mean']:.3f}%"
                assert stats['p99'] < thresh['p99'], \
                    f"P99 diff {stats['p99']:.3f}% exceeds threshold {thresh['p99']:.3f}%"
                assert stats['max'] < thresh['max'], \
                    f"Max diff {stats['max']:.3f}% exceeds threshold {thresh['max']:.3f}%"
                
                # Save difference visualization
                diff = np.abs(preview.astype(np.float32) - expected.astype(np.float32))
                diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
                diff_path = test_output_dir / f"ifd{i}_vs_ifd{largest_ifd}_diff.tif"
                tifffile.imwrite(str(diff_path), diff_vis)
                
                print(f"       ✓ Passed thresholds")
        else:
            print(f"     Only one RGB/YCBCR preview found, skipping inter-preview comparison")
        
        # Step 6: Test DngFile.get_preview_rgb() API
        print(f"\n6. Testing DngFile.get_preview_rgb() API...")
        
        ifd0 = pages[0]
        
        try:
            # Always call get_preview_rgb() to test API behavior
            api_preview = dng.get_preview_rgb(output_dtype=np.uint16)
            
            if api_preview is None:
                # Should only return None if IFD0 is not a preview
                print(f"   get_preview_rgb() returned None - verifying IFD0 is not a preview...")
                assert ifd0.photometric_name not in ("RGB", "YCBCR"), \
                    f"get_preview_rgb() returned None but IFD0 is {ifd0.photometric_name} (should be a preview)"
                print(f"     ✓ Correct: IFD0 is {ifd0.photometric_name} (not RGB/YCBCR)")
            else:
                # Returned a preview - verify IFD0 is actually RGB/YCBCR
                print(f"   get_preview_rgb() returned preview - verifying IFD0 is RGB/YCBCR...")
                assert ifd0.photometric_name in ("RGB", "YCBCR"), \
                    f"get_preview_rgb() returned data but IFD0 is {ifd0.photometric_name} (not a preview)"
                
                print(f"     ✓ Correct: IFD0 is {ifd0.photometric_name} preview")
                print(f"     ✓ DngFile.get_preview_rgb(): {api_preview.shape}, dtype={api_preview.dtype}")
                
                # Get preview using page API for comparison
                page_preview = ifd0.decode_to_rgb(output_dtype=np.uint16)
                assert page_preview is not None, "decode_to_rgb returned None"
                
                print(f"     ✓ DngPage.decode_to_rgb(): {page_preview.shape}, dtype={page_preview.dtype}")
                
                # Compare the two methods - should be identical
                assert api_preview.shape == page_preview.shape, \
                    f"Shape mismatch: API {api_preview.shape} vs Page {page_preview.shape}"
                
                # Compute difference (should be zero or near-zero)
                diff = np.abs(api_preview.astype(np.float32) - page_preview.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"     Comparison: mean diff={mean_diff:.6f}, max diff={max_diff:.6f}")
                
                # Should be identical (allow tiny floating point error)
                assert max_diff < 1.0, f"Max difference {max_diff} too large - APIs should return identical results"
                
                # Save comparison
                api_path = test_output_dir / "api_get_preview_rgb.tif"
                page_path = test_output_dir / "page_decode_to_rgb.tif"
                tifffile.imwrite(str(api_path), api_preview)
                tifffile.imwrite(str(page_path), page_preview)
                
                print(f"     ✓ APIs return identical results")
                
        except Exception as e:
            pytest.fail(f"DngFile.get_preview_rgb() API test failed: {e}")
        
        print(f"\n✓ Test completed. Results saved to: {test_output_dir}")


def test_decode_to_rgb_on_raw_page():
    """Test that decode_to_rgb() works on CFA/LINEAR_RAW pages by rendering them."""
    
    test_file = DNGFILES_DIR / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    
    if not test_file.exists():
        pytest.skip(f"Test file not available: {test_file}")
    
    with muimg.DngFile(test_file) as dng:
        pages = dng.get_flattened_pages()
        
        # Find a CFA page (main page)
        main_page = dng.get_main_page()
        assert main_page is not None
        assert main_page.is_cfa or main_page.is_linear_raw
        
        # Should successfully decode (render) the raw page
        result = main_page.decode_to_rgb(output_dtype=np.uint8)
        assert result is not None
        assert result.shape[2] == 3  # RGB output
        assert result.dtype == np.uint8


def test_render_raw_on_preview_page():
    """Test that render_raw raises ValueError on RGB/YCBCR pages."""
    
    test_file = DNGFILES_DIR / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    
    if not test_file.exists():
        pytest.skip(f"Test file not available: {test_file}")
    
    with muimg.DngFile(test_file) as dng:
        pages = dng.get_flattened_pages()
        
        # Find an RGB preview page (IFD0)
        ifd0 = pages[0]
        assert ifd0.photometric_name in ("RGB", "YCBCR")
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="render_raw.*requires CFA or LINEAR_RAW"):
            ifd0.render_raw(output_dtype=np.uint16)


def test_dngfile_decode_to_rgb_on_raw_ifd0():
    """Test that DngFile.decode_to_rgb raises ValueError when IFD0 is raw."""
    
    # Need a DNG file where IFD0 is raw (not common, but possible)
    # For now, we'll look for such a file in the test directory
    
    # Check if we have any DNG files where IFD0 is raw
    test_files = list(DNGFILES_DIR.glob("*.dng"))
    
    found_raw_ifd0 = False
    for test_file in test_files:
        try:
            with muimg.DngFile(test_file) as dng:
                pages = dng.get_flattened_pages()
                if pages and (pages[0].is_cfa or pages[0].is_linear_raw):
                    # Found a file with raw IFD0
                    found_raw_ifd0 = True
                    
                    # Should raise ValueError
                    with pytest.raises(ValueError, match="IFD0 is a raw page"):
                        dng.decode_to_rgb(output_dtype=np.uint8)
                    
                    print(f"✓ Tested with {test_file.name} (IFD0 is {pages[0].photometric_name})")
                    break
        except Exception:
            continue
    
    if not found_raw_ifd0:
        pytest.skip("No DNG files found with raw IFD0 for testing")
