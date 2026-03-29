"""Test demosaic algorithms for consistency across bit depths."""
import numpy as np
import pytest
from pathlib import Path

from muimg import DngFile
from muimg.raw_render import demosaic, convert_dtype
from conftest import compute_diff_stats


# Test file path
TEST_DNG = Path(__file__).parent / "dngfiles" / "asi676mc.cfa.jxl_lossy.2ifds.dng"


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
    reference = demosaic(cfa_u8, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    assert reference.dtype == np.uint8
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in ["VNG", "RCD", "OPENCV_EA"]:
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
    reference = demosaic(cfa_u16, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    assert reference.dtype == np.uint16
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in ["VNG", "RCD", "OPENCV_EA"]:
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
    reference = demosaic(cfa_f32, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    assert reference.dtype == np.float32
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)
    
    # Test other algorithms produce same dtype and similar results
    for algorithm in ["VNG", "RCD", "OPENCV_EA"]:
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
    result_f32 = demosaic(cfa_f32, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    
    # Convert to uint16 and demosaic
    cfa_u16 = convert_dtype(cfa, np.uint16)
    result_u16 = demosaic(cfa_u16, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    
    # Convert uint16 result back to float32 for comparison
    result_u16_as_f32 = convert_dtype(result_u16, np.float32)
    
    # Results should be very close (within quantization error)
    # uint16 has 65536 levels, so max error is ~1/65536 ≈ 0.000015
    max_diff = np.abs(result_f32 - result_u16_as_f32).max()
    assert max_diff < 0.001, f"Float32 and uint16 results differ by {max_diff}"
    
    # Same test for uint8
    cfa_u8 = convert_dtype(cfa, np.uint8)
    result_u8 = demosaic(cfa_u8, cfa_pattern, algorithm="DNGSDK_BILINEAR")
    result_u8_as_f32 = convert_dtype(result_u8, np.float32)
    
    # uint8 has only 256 levels, so allow larger error
    max_diff = np.abs(result_f32 - result_u8_as_f32).max()
    assert max_diff < 0.01, f"Float32 and uint8 results differ by {max_diff}"


def test_demosaic_invalid_algorithm(cfa_data):
    """Test that invalid algorithm raises ValueError."""
    cfa, cfa_pattern = cfa_data
    
    with pytest.raises(ValueError, match="Invalid algorithm"):
        demosaic(cfa, cfa_pattern, algorithm="INVALID_ALGO")


def test_demosaic_invalid_pattern(cfa_data):
    """Test that invalid CFA pattern raises ValueError."""
    cfa, _ = cfa_data
    
    with pytest.raises(ValueError, match="Invalid CFA pattern"):
        demosaic(cfa, "INVALID_PATTERN", algorithm="DNGSDK_BILINEAR")
