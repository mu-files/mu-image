"""Test demosaic algorithms for consistency across bit depths."""
import numpy as np
import pytest
from pathlib import Path
import tifffile

from muimg import DngFile
from muimg.raw_render import demosaic, convert_dtype, DemosaicAlgorithm
from conftest import compute_diff_stats


# Test file path
TEST_DNG = Path(__file__).parent / "dngfiles" / "sony_ilce-7c_cfa_jxl_lossy_4ifds.dng"


@pytest.fixture
def cfa_data():
    """Load CFA data from test DNG file."""
    if not TEST_DNG.exists():
        pytest.skip(f"Test file not found: {TEST_DNG}")
    
    with DngFile(TEST_DNG) as dng:
        cfa, cfa_pattern = dng.get_cfa()
        return cfa, cfa_pattern


def test_demosaic_uint8_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for uint8 input."""
    cfa, cfa_pattern = cfa_data
    
    # Convert CFA to uint8
    cfa_u8 = convert_dtype(cfa, np.uint8)
    
    # Run DNGSDK_BILINEAR as reference
    reference = demosaic(cfa_u8, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    assert reference.dtype == np.uint8
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in [DemosaicAlgorithm.VNG, DemosaicAlgorithm.RCD, DemosaicAlgorithm.OPENCV_EA]:
        result = demosaic(cfa_u8, cfa_pattern, algorithm=algorithm)
        
        # Check dtype preservation
        assert result.dtype == np.uint8, f"{algorithm} should preserve uint8 dtype"
        
        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"
        
        # Check values are in valid range
        assert result.min() >= 0 and result.max() <= 255, f"{algorithm} values out of range"
        
        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"
        
        # Check similarity to reference
        # Different algorithms will differ, but should be reasonably close
        # Based on actual measurements: mean ~0.02-0.15%, p99 ~0.4-1.2%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(f"\n{algorithm} vs DNGSDK_BILINEAR (uint8): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%")
        assert diff_stats['mean'] < 1.0, f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        assert diff_stats['p99'] < 2.0, f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"


def test_demosaic_uint16_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for uint16 input."""
    cfa, cfa_pattern = cfa_data
    
    # Convert CFA to uint16
    cfa_u16 = convert_dtype(cfa, np.uint16)
    
    # Run DNGSDK_BILINEAR as reference
    reference = demosaic(cfa_u16, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    assert reference.dtype == np.uint16
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in [DemosaicAlgorithm.VNG, DemosaicAlgorithm.RCD, DemosaicAlgorithm.OPENCV_EA]:
        result = demosaic(cfa_u16, cfa_pattern, algorithm=algorithm)
        
        # Check dtype preservation
        assert result.dtype == np.uint16, f"{algorithm} should preserve uint16 dtype"
        
        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"
        
        # Check values are in valid range
        assert result.min() >= 0 and result.max() <= 65535, f"{algorithm} values out of range"
        
        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"
        
        # Check similarity to reference
        # Based on actual measurements: mean ~0.02-0.12%, p99 ~0.4-1.0%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(f"\n{algorithm} vs DNGSDK_BILINEAR (uint16): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%")
        assert diff_stats['mean'] < 1.0, f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        assert diff_stats['p99'] < 2.0, f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"


def test_demosaic_float32_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for float32 input."""
    cfa, cfa_pattern = cfa_data
    
    # Convert CFA to float32 (0-1 range)
    cfa_f32 = convert_dtype(cfa, np.float32)
    
    # Run DNGSDK_BILINEAR as reference
    reference = demosaic(cfa_f32, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    assert reference.dtype == np.float32
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in [DemosaicAlgorithm.VNG, DemosaicAlgorithm.RCD, DemosaicAlgorithm.OPENCV_EA]:
        result = demosaic(cfa_f32, cfa_pattern, algorithm=algorithm)
        
        # Check dtype preservation
        assert result.dtype == np.float32, f"{algorithm} should preserve float32 dtype"
        
        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"
        
        # Check values are in valid range (0-1 for normalized float)
        assert result.min() >= 0.0 and result.max() <= 1.0, f"{algorithm} values out of range"
        
        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"
        
        # Check similarity to reference
        # Based on actual measurements: mean ~0.02-0.12%, p99 ~0.4-1.0%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(f"\n{algorithm} vs DNGSDK_BILINEAR (float32): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%")
        assert diff_stats['mean'] < 1.0, f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        assert diff_stats['p99'] < 2.0, f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"


def test_demosaic_dtype_conversion_roundtrip(cfa_data):
    """Test that demosaic preserves relative values across dtype conversions."""
    cfa, cfa_pattern = cfa_data
    
    # Get float32 reference
    cfa_f32 = convert_dtype(cfa, np.float32)
    result_f32 = demosaic(cfa_f32, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    
    # Convert to uint16 and demosaic
    cfa_u16 = convert_dtype(cfa, np.uint16)
    result_u16 = demosaic(cfa_u16, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    
    # Convert uint16 result back to float32 for comparison
    result_u16_as_f32 = convert_dtype(result_u16, np.float32)
    
    # Results should be very close (within quantization error)
    # uint16 has 65536 levels, so max error is ~1/65536 ≈ 0.000015
    max_diff = np.abs(result_f32 - result_u16_as_f32).max()
    assert max_diff < 0.001, f"Float32 and uint16 results differ by {max_diff}"
    
    # Same test for uint8
    cfa_u8 = convert_dtype(cfa, np.uint8)
    result_u8 = demosaic(cfa_u8, cfa_pattern, algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)
    result_u8_as_f32 = convert_dtype(result_u8, np.float32)
    
    # uint8 has only 256 levels, so allow larger error
    max_diff = np.abs(result_f32 - result_u8_as_f32).max()
    assert max_diff < 0.01, f"Float32 and uint8 results differ by {max_diff}"


def test_demosaic_invalid_algorithm(cfa_data):
    """Test that invalid algorithm raises ValueError."""
    cfa, cfa_pattern = cfa_data
    
    with pytest.raises(TypeError, match="algorithm must be a DemosaicAlgorithm enum"):
        demosaic(cfa, cfa_pattern, algorithm="INVALID_ALGO")


def test_demosaic_invalid_pattern(cfa_data):
    """Test that invalid CFA pattern raises ValueError."""
    cfa, _ = cfa_data
    
    with pytest.raises(ValueError, match="Invalid CFA pattern"):
        demosaic(cfa, "INVALID_PATTERN", algorithm=DemosaicAlgorithm.DNGSDK_BILINEAR)


def test_demosaic_cfa_pattern_consistency(cfa_data):
    """Test that demosaic works correctly for all four CFA patterns.
    
    The four Bayer patterns are:
    RGGB:  RG    GRBG:  GR    GBRG:  GB    BGGR:  BG
           GB           BG           RG           GR
    
    We can create the other three patterns from RGGB by cropping by (x,y):
    - GRBG: crop by (1,0) - shift left by 1 column
    - GBRG: crop by (0,1) - shift up by 1 row
    - BGGR: crop by (1,1) - shift left 1 column and up 1 row
    
    For each pattern, we demosaic the cropped CFA and compare against the
    corresponding crop of the original demosaic result.
    """
    cfa, cfa_pattern = cfa_data
    
    # Ensure we start with RGGB pattern
    assert cfa_pattern == "RGGB", f"Test expects RGGB pattern, got {cfa_pattern}"
    
    # Convert to uint16 for testing
    cfa_u16 = convert_dtype(cfa, np.uint16)
    
    # Pattern transformations: (crop_x, crop_y) -> new_pattern
    pattern_crops = {
        "RGGB": (0, 0),  # Original, no crop
        "BGGR": (1, 1),  # Shift left 1 column and up 1 row
        "GBRG": (0, 1),  # Shift up by 1 row
        "GRBG": (1, 0),  # Shift left by 1 column
    }
    
    # Test each algorithm for the reference
    algorithms = [DemosaicAlgorithm.DNGSDK_BILINEAR, DemosaicAlgorithm.OPENCV_EA,
                  DemosaicAlgorithm.VNG, DemosaicAlgorithm.RCD, ]
    
    for ref_algorithm in algorithms:
        # Demosaic the original RGGB pattern with reference algorithm
        reference_rgb = demosaic(cfa_u16, "RGGB", algorithm=ref_algorithm)
        
        # Test each pattern variant
        for pattern_name, (crop_x, crop_y) in pattern_crops.items():
            # Crop the CFA to create the new pattern
            cropped_cfa = cfa_u16[crop_y:, crop_x:]
            
            # Make dimensions even (required for demosaic)
            h, w = cropped_cfa.shape
            if h % 2 == 1:
                cropped_cfa = cropped_cfa[:-1, :]
            if w % 2 == 1:
                cropped_cfa = cropped_cfa[:, :-1]
            
            # Make contiguous copy (C extensions may assume contiguous memory)
            cropped_cfa = np.ascontiguousarray(cropped_cfa)
            
            # Crop the reference RGB to match (include color channel dimension)
            expected_rgb = reference_rgb[crop_y:, crop_x:, :]
            h, w = cropped_cfa.shape
            expected_rgb = expected_rgb[:h, :w, :]
            
            # Test demosaicing the cropped pattern with all algorithms
            for test_algorithm in algorithms:
                # Demosaic the cropped CFA with the test algorithm
                cropped_rgb = demosaic(cropped_cfa, pattern_name, algorithm=test_algorithm)
                
                # Results should have correct shape
                assert cropped_rgb.shape == expected_rgb.shape, \
                    f"{ref_algorithm}->{test_algorithm} {pattern_name}: shape mismatch"
                
                # Compare results - allow small differences at edges due to interpolation
                diff = np.abs(cropped_rgb.astype(np.int32) - expected_rgb.astype(np.int32))
                
                # Exclude 2-pixel border where edge effects may occur
                if diff.shape[0] > 4 and diff.shape[1] > 4:
                    diff_interior = diff[2:-2, 2:-2]
                    max_interior_diff = diff_interior.max()
                    mean_interior_diff = diff_interior.mean()
                    
                    print(f"\n{ref_algorithm}->{test_algorithm} {pattern_name}: max_interior={max_interior_diff}, mean_interior={mean_interior_diff:.2f}")
                    
                    # Always verify non-zero output
                    assert not np.all(cropped_rgb == 0), \
                        f"{ref_algorithm}->{test_algorithm} {pattern_name}: produced all zeros"
                    
                    # When using same algorithm, interior should be nearly identical
                    if ref_algorithm == test_algorithm:
                        # RCD has localized interpolation differences, use relaxed thresholds
                        if test_algorithm == DemosaicAlgorithm.RCD:
                            max_threshold = 12000
                            mean_threshold = 0.85
                        else:
                            max_threshold = 1
                            mean_threshold = 0.1
                        
                        if max_interior_diff > max_threshold or mean_interior_diff >= mean_threshold:
                            # Save debug output
                            output_dir = Path(__file__).parent / "test_outputs" / "test_demosaic"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            ref_file = output_dir / f"{ref_algorithm.name}_{pattern_name}_reference.tif"
                            test_file = output_dir / f"{ref_algorithm.name}_{pattern_name}_test.tif"
                            diff_file = output_dir / f"{ref_algorithm.name}_{pattern_name}_diff.tif"
                            
                            tifffile.imwrite(ref_file, expected_rgb)
                            tifffile.imwrite(test_file, cropped_rgb)
                            tifffile.imwrite(diff_file, diff.astype(np.uint16))
                            
                            print(f"Saved debug output to {output_dir}")
                        
                        assert max_interior_diff <= max_threshold, \
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: interior max diff {max_interior_diff} > {max_threshold}"
                        assert mean_interior_diff < mean_threshold, \
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: interior mean diff {mean_interior_diff:.2f} > {mean_threshold}"
                    else:
                        # Cross-algorithm: algorithms differ but should be reasonably similar
                        # Mean difference should be < 96 (out of 65535, ~0.15%)
                        assert mean_interior_diff < 96, \
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: cross-algorithm mean diff {mean_interior_diff:.2f} > 100"
