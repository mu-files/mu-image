"""Test that copy_dng produces equivalent decoded output regardless of source format."""

import numpy as np
import pytest
from pathlib import Path

from muimg.imgio import decode_dng
from conftest import TEST_FILES_DIR, compute_diff_stats, run_dng_validate, DNG_VALIDATE_PATH


# Test files - original, pure copy, and demosaiced copy
TESTDATA_DIR = Path("/Users/anonymized/Projects/python/mu-image-testdata/xmptestfiles")
ORIGINAL_DNG = TESTDATA_DIR / "canon_eos_r5.cfa.ljpeg.6ifds.dng"
COPIED_DNG = TESTDATA_DIR / "canon_eos_r5.cfa.copied.dng"
DEMOSAICED_DNG = TESTDATA_DIR / "canon_eos_r5.cfa.demosaic.dng"

# Output directory for dng_validate comparison files
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_copy_dng_equivalence"


@pytest.mark.skipif(
    not all([ORIGINAL_DNG.exists(), COPIED_DNG.exists(), DEMOSAICED_DNG.exists()]),
    reason="Test DNG files not found"
)
def test_copy_dng_decode_equivalence():
    """Test that original, copied, and demosaiced DNGs decode to identical images.
    
    This verifies that:
    1. Pure copy (page copy API) preserves the raw data exactly
    2. Demosaic copy (get_camera_rgb + strip tags) produces equivalent output
    
    All files should decode to the same rendered image when using use_xmp=False.
    """
    # Decode all three files with use_xmp=False to ensure consistent rendering
    original_img = decode_dng(
        file=str(ORIGINAL_DNG),
        output_dtype=np.uint16,
        use_xmp=False,
    )
    assert original_img is not None, "Failed to decode original DNG"
    
    copied_img = decode_dng(
        file=str(COPIED_DNG),
        output_dtype=np.uint16,
        use_xmp=False,
    )
    assert copied_img is not None, "Failed to decode copied DNG"
    
    demosaiced_img = decode_dng(
        file=str(DEMOSAICED_DNG),
        output_dtype=np.uint16,
        use_xmp=False,
    )
    assert demosaiced_img is not None, "Failed to decode demosaiced DNG"
    
    print(f"\nOriginal shape: {original_img.shape}")
    print(f"Copied shape: {copied_img.shape}")
    print(f"Demosaiced shape: {demosaiced_img.shape}")
    
    # Compare original vs copied (should be identical)
    print("\n=== Original vs Copied ===")
    if original_img.shape != copied_img.shape:
        pytest.fail(
            f"Shape mismatch: original={original_img.shape}, copied={copied_img.shape}"
        )
    
    stats_copied = compute_diff_stats(original_img, copied_img)
    print(f"Mean diff: {stats_copied['mean']:.4f}%")
    print(f"P99 diff: {stats_copied['p99']:.4f}%")
    print(f"Max diff: {stats_copied['max']:.4f}%")
    
    # Pure copy should be pixel-perfect
    assert np.array_equal(original_img, copied_img), (
        f"Copied DNG differs from original: mean={stats_copied['mean']:.4f}%, "
        f"max={stats_copied['max']:.4f}%"
    )
    
    # Compare original vs demosaiced (should be identical)
    print("\n=== Original vs Demosaiced ===")
    if original_img.shape != demosaiced_img.shape:
        pytest.fail(
            f"Shape mismatch: original={original_img.shape}, demosaiced={demosaiced_img.shape}"
        )
    
    stats_demosaiced = compute_diff_stats(original_img, demosaiced_img)
    print(f"Mean diff: {stats_demosaiced['mean']:.4f}%")
    print(f"P99 diff: {stats_demosaiced['p99']:.4f}%")
    print(f"Max diff: {stats_demosaiced['max']:.4f}%")
    
    # Demosaiced copy will have small differences due to float32 conversion
    # Allow small tolerance for rounding differences
    assert stats_demosaiced['mean'] < 0.02, (
        f"Demosaiced DNG differs from original: mean={stats_demosaiced['mean']:.4f}% "
        f"exceeds threshold 0.02%"
    )
    assert stats_demosaiced['max'] < 0.5, (
        f"Demosaiced DNG differs from original: max={stats_demosaiced['max']:.4f}% "
        f"exceeds threshold 0.5%"
    )
    
    print("\n✓ All three DNGs decode to identical images")


@pytest.mark.skipif(
    not all([ORIGINAL_DNG.exists(), COPIED_DNG.exists(), DEMOSAICED_DNG.exists()]),
    reason="Test DNG files not found"
)
@pytest.mark.skipif(not DNG_VALIDATE_PATH.exists(), reason="dng_validate not available")
def test_copy_dng_vs_dng_validate():
    """Compare muimg and dng_validate decoding for all three DNG variants.
    
    This verifies that both pipelines produce equivalent results for:
    1. Original CFA DNG
    2. Copied CFA DNG
    3. Demosaiced LINEAR_RAW DNG
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("original", ORIGINAL_DNG),
        ("copied", COPIED_DNG),
        ("demosaiced", DEMOSAICED_DNG),
    ]
    
    for label, dng_path in files:
        print(f"\n=== {label.upper()}: {dng_path.name} ===")
        
        # Decode with muimg
        muimg_img = decode_dng(
            file=str(dng_path),
            output_dtype=np.uint16,
            use_xmp=False,
        )
        assert muimg_img is not None, f"Failed to decode {label} with muimg"
        print(f"muimg shape: {muimg_img.shape}")
        
        # Decode with dng_validate
        dngvalidate_base = OUTPUT_DIR / f"canon_eos_r5.{label}.dngvalidate"
        dngvalidate_img = run_dng_validate(dng_path, dngvalidate_base, timeout=60)
        assert dngvalidate_img is not None, f"Failed to decode {label} with dng_validate"
        print(f"dng_validate shape: {dngvalidate_img.shape}")
        
        # Compare
        if muimg_img.shape != dngvalidate_img.shape:
            pytest.fail(
                f"{label}: Shape mismatch - muimg={muimg_img.shape}, "
                f"dng_validate={dngvalidate_img.shape}"
            )
        
        stats = compute_diff_stats(muimg_img, dngvalidate_img)
        print(f"muimg vs dng_validate: mean={stats['mean']:.4f}%, "
              f"p99={stats['p99']:.4f}%, max={stats['max']:.4f}%")
        
        # Allow differences due to different demosaic implementations
        # muimg uses RCD, dng_validate uses bilinear
        # RCD is higher quality but produces different results than bilinear
        assert stats['mean'] < 1.0, (
            f"{label}: Mean difference {stats['mean']:.4f}% exceeds threshold 1.0%"
        )
        assert stats['p99'] < 6.0, (
            f"{label}: P99 difference {stats['p99']:.4f}% exceeds threshold 6.0%"
        )
    
    print("\n✓ All three DNGs decode consistently across muimg and dng_validate")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
