#!/usr/bin/env python3
"""Compare process_raw (DNG SDK pipeline) vs C++ SDK reference (dng_validate).

This script processes DNG files from the rawfiles test folder using:
1. C++ SDK dng_validate tool (generates reference TIFF)
2. Python process_raw() implementation
3. macOS Core Image (for additional comparison)

Outputs are compared against the C++ SDK reference.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

# Configure logging - suppress verbose timing from muimg.color
logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')
logging.getLogger('muimg.color').setLevel(logging.WARNING)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from muimg import color
from muimg.color_mac import process_raw_core_image

# Path to the C++ SDK dng_validate tool
DNG_VALIDATE_PATH = Path.home() / "Projects/C/3dparty/dng_sdk_1_7_1/dng_sdk/targets/mac/release64/dng_validate"


def generate_cpp_sdk_reference(dng_path: Path, output_dir: Path) -> Path | None:
    """Generate reference TIFF using C++ SDK dng_validate tool.
    
    Returns the path to the generated TIFF, or None if generation failed.
    """
    if not DNG_VALIDATE_PATH.exists():
        print(f"  [REF] ERROR: dng_validate not found at {DNG_VALIDATE_PATH}")
        return None
    
    # dng_validate appends .tif to the path, so don't include extension
    output_base = output_dir / f"{dng_path.stem}_cppsdk"
    output_tiff = output_dir / f"{dng_path.stem}_cppsdk.tif"
    
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            [str(DNG_VALIDATE_PATH), "-v", "-16", "-tif", str(output_base), str(dng_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed = time.perf_counter() - t0
        
        if result.returncode == 0 and output_tiff.exists():
            print(f"  [REF] Generated C++ SDK reference in {elapsed*1000:.1f}ms")
            print(f"  [REF] Saved: {output_tiff.name}")
            return output_tiff
        else:
            print(f"  [REF] ERROR: dng_validate failed (code {result.returncode})")
            if result.stderr:
                print(f"  [REF] stderr: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  [REF] ERROR: dng_validate timed out")
        return None
    except Exception as e:
        print(f"  [REF] ERROR: {e}")
        return None


def load_reference_tiff(ref_path: Path) -> np.ndarray | None:
    """Load the C++ SDK reference TIFF for comparison."""
    if ref_path is None or not ref_path.exists():
        return None
    try:
        with tifffile.TiffFile(str(ref_path)) as tif:
            # Debug: print TIFF structure
            page = tif.pages[0]
            print(f"  [REF] TIFF info: shape={page.shape}, dtype={page.dtype}, "
                  f"photometric={page.photometric.name if page.photometric else 'None'}, "
                  f"planar={page.planarconfig.name if page.planarconfig else 'None'}")
            
            # Read the image data
            img = page.asarray()
            print(f"  [REF] Loaded array: shape={img.shape}, dtype={img.dtype}")
            return img
    except Exception as e:
        print(f"  [REF] ERROR loading reference: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_and_save(dng_path: Path, output_dir: Path):
    """Process a DNG file with all implementations and compare against C++ SDK reference."""
    stem = dng_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {dng_path.name}")
    print(f"{'='*60}")
    
    # Step 1: Generate C++ SDK reference
    print("  [REF] Generating C++ SDK reference with dng_validate...")
    ref_path = generate_cpp_sdk_reference(dng_path, output_dir)
    ref_image = load_reference_tiff(ref_path)
    
    # Process with Python SDK implementation
    print("  [SDK] Processing with process_raw()...")
    t0 = time.perf_counter()
    try:
        sdk_result = color.process_raw(str(dng_path), output_dtype=np.uint16, algorithm="DNGSDK_BILINEAR", strict=False)
        sdk_time = time.perf_counter() - t0
        if sdk_result is not None:
            sdk_out = output_dir / f"{stem}_sdk.tiff"
            tifffile.imwrite(str(sdk_out), sdk_result)
            print(f"  [SDK] Success: {sdk_result.shape}, {sdk_result.dtype} in {sdk_time*1000:.1f}ms")
            print(f"  [SDK] Saved: {sdk_out.name}")
        else:
            print(f"  [SDK] FAILED: returned None")
            sdk_result = None
    except Exception as e:
        print(f"  [SDK] ERROR: {e}")
        sdk_result = None
        sdk_time = 0
    
    # Process with Core Image
    print("  [CI]  Processing with process_raw_core_image()...")
    t0 = time.perf_counter()
    try:
        ci_result = process_raw_core_image(str(dng_path), output_dtype=np.uint16)
        ci_time = time.perf_counter() - t0
        if ci_result is not None:
            ci_out = output_dir / f"{stem}_coreimage.tiff"
            tifffile.imwrite(str(ci_out), ci_result)
            print(f"  [CI]  Success: {ci_result.shape}, {ci_result.dtype} in {ci_time*1000:.1f}ms")
            print(f"  [CI]  Saved: {ci_out.name}")
        else:
            print(f"  [CI]  FAILED: returned None")
            ci_result = None
    except Exception as e:
        print(f"  [CI]  ERROR: {e}")
        ci_result = None
        ci_time = 0
    
    # Compare results against C++ SDK reference
    if ref_image is not None:
        print("  [CMP] Comparing against C++ SDK reference:")
        
        # Handle potential planar vs interleaved difference
        ref_img = ref_image
        if ref_img.ndim == 3 and ref_img.shape[0] == 3 and ref_img.shape[0] < ref_img.shape[1]:
            print("  [CMP]   Converting REF from planar (3,H,W) to interleaved (H,W,3)")
            ref_img = np.moveaxis(ref_img, 0, -1)
        
        # Normalize to float [0,1] for comparison regardless of bit depth
        if ref_img.dtype == np.uint8:
            ref_normalized = ref_img.astype(np.float32) / 255.0
        elif ref_img.dtype == np.uint16:
            ref_normalized = ref_img.astype(np.float32) / 65535.0
        else:
            ref_normalized = ref_img.astype(np.float32)
        
        if sdk_result is not None:
            if sdk_result.shape == ref_img.shape:
                if sdk_result.dtype == np.uint16:
                    sdk_normalized = sdk_result.astype(np.float32) / 65535.0
                elif sdk_result.dtype == np.uint8:
                    sdk_normalized = sdk_result.astype(np.float32) / 255.0
                else:
                    sdk_normalized = sdk_result.astype(np.float32)
                
                diff = np.abs(sdk_normalized - ref_normalized)
                mean_diff = np.mean(diff) * 100  # as percentage
                max_diff = np.max(diff) * 100
                # Count pixels with >50% difference
                extreme_count = np.sum(diff > 0.5)
                total_pixels = diff.size
                p99_diff = np.percentile(diff, 99) * 100
                print(f"  [CMP]   Python SDK vs C++ REF: mean={mean_diff:.2f}%, p99={p99_diff:.2f}%, max={max_diff:.2f}% ({extreme_count}/{total_pixels} pixels >50%)")
            else:
                print(f"  [CMP]   Python SDK shape mismatch: {sdk_result.shape} vs REF {ref_img.shape}")
        
        if ci_result is not None:
            if ci_result.shape == ref_img.shape:
                if ci_result.dtype == np.uint16:
                    ci_normalized = ci_result.astype(np.float32) / 65535.0
                elif ci_result.dtype == np.uint8:
                    ci_normalized = ci_result.astype(np.float32) / 255.0
                else:
                    ci_normalized = ci_result.astype(np.float32)
                
                diff = np.abs(ci_normalized - ref_normalized)
                mean_diff = np.mean(diff) * 100
                max_diff = np.max(diff) * 100
                extreme_count = np.sum(diff > 0.5)
                total_pixels = diff.size
                p99_diff = np.percentile(diff, 99) * 100
                print(f"  [CMP]   Core Image vs C++ REF: mean={mean_diff:.2f}%, p99={p99_diff:.2f}%, max={max_diff:.2f}% ({extreme_count}/{total_pixels} pixels >50%)")
            else:
                print(f"  [CMP]   Core Image shape mismatch: {ci_result.shape} vs REF {ref_img.shape}")
    
    return ref_image is not None, sdk_result is not None, ci_result is not None


def main():
    # Import paths from conftest (auto-clones fixtures if needed)
    from conftest import TEST_FILES_DIR
    
    # Test files directory
    test_dir = TEST_FILES_DIR
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1
    
    # Output directory
    output_dir = Path(__file__).parent / "output_comparison"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all DNG files
    dng_files = sorted(test_dir.glob("*.dng"))
    if not dng_files:
        print(f"Error: No DNG files found in {test_dir}")
        return 1
    
    print(f"Found {len(dng_files)} DNG files to process")
    
    # Process each file
    results = []
    for dng_path in dng_files:
        ref_ok, sdk_ok, ci_ok = process_and_save(dng_path, output_dir)
        results.append((dng_path.name, ref_ok, sdk_ok, ci_ok))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'File':<40} {'C++ REF':>8} {'Py SDK':>8} {'CI':>8}")
    print("-" * 70)
    for name, ref_ok, sdk_ok, ci_ok in results:
        ref_str = "OK" if ref_ok else "FAIL"
        sdk_str = "OK" if sdk_ok else "FAIL"
        ci_str = "OK" if ci_ok else "FAIL"
        print(f"{name:<40} {ref_str:>8} {sdk_str:>8} {ci_str:>8}")
    
    ref_total = sum(1 for _, ok, _, _ in results if ok)
    sdk_total = sum(1 for _, _, ok, _ in results if ok)
    ci_total = sum(1 for _, _, _, ok in results if ok)
    print("-" * 70)
    print(f"{'TOTAL':<40} {ref_total:>5}/{len(results)} {sdk_total:>5}/{len(results)} {ci_total:>5}/{len(results)}")
    
    print(f"\nOutput files saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
