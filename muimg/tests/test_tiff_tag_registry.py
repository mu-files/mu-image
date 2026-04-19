"""Tests for TIFF_TAG_TYPE_REGISTRY validation.

Validates that:
1. Tags in real DNG files have dtypes compatible with our registry
2. TagSpec type inference works correctly
3. Value conversion produces correct TIFF types
4. All registry tag names resolve in tifffile (except those with explicit codes)
"""

from pathlib import Path
from typing import Set

import numpy as np
import pytest

import muimg
from muimg.tiff_metadata import (
    TIFF_TAG_TYPE_REGISTRY,
    TagSpec,
    DTYPE_CATEGORY,
    encode_tag_value,
    LOCAL_TIFF_TAGS,
)


DNGFILES_DIR = Path(__file__).parent.parent.parent.parent / "mu-image-testdata" / "dngtestfiles"


def get_dng_files():
    """Get list of DNG files for parametrized tests (searches recursively)."""
    files = []
    if DNGFILES_DIR.exists():
        files.extend(DNGFILES_DIR.rglob("*.dng"))
    
    # Warn if no files found
    if not files:
        import warnings
        warnings.warn(
            f"No DNG files found in {DNGFILES_DIR}. "
            f"Tests will be skipped. Please ensure test files are available.",
            UserWarning
        )
    
    return sorted(files)


def dtype_compatible(file_dtype: str, spec: TagSpec) -> bool:
    """Check if a file's dtype is compatible with a TagSpec.
    
    Args:
        file_dtype: The dtype string from tifffile (e.g., 'H', 'I', '2I')
        spec: The TagSpec from our registry
        
    Returns:
        True if compatible, False otherwise
    """
    # Get allowed dtypes from spec
    if isinstance(spec.dtype, str):
        allowed = {spec.dtype}
    else:
        allowed = set(spec.dtype)
    
    # Direct match
    if file_dtype in allowed:
        return True
    
    # Check category compatibility (int types are interchangeable, float types are interchangeable)
    file_category = DTYPE_CATEGORY.get(file_dtype)
    if file_category:
        for dt in allowed:
            if DTYPE_CATEGORY.get(dt) == file_category:
                return True
    
    return False


class TestTagRegistryCompatibility:
    """Test that real DNG file tags are compatible with our registry."""
    
    @pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
    def test_dng_tags_compatible_with_registry(self, dng_path: Path):
        """Validate tags in DNG file match registry dtypes."""
        incompatible_tags = []
        unknown_tags: Set[str] = set()
        total_tags = 0
        validated_tags = 0
        
        with muimg.DngFile(dng_path) as dng:
            for page in dng.get_flattened_pages():
                for tag in page.tags.values():
                    tag_name = tag.name
                    total_tags += 1
                    
                    # Skip unknown numeric tags
                    if tag_name.isdigit():
                        continue
                    
                    if tag_name not in TIFF_TAG_TYPE_REGISTRY:
                        unknown_tags.add(tag_name)
                        continue
                    
                    validated_tags += 1
                    spec = TIFF_TAG_TYPE_REGISTRY[tag_name]
                    
                    # Get file's dtype
                    file_dtype = self._tifffile_dtype_to_string(tag.dtype)
                    
                    if not dtype_compatible(file_dtype, spec):
                        incompatible_tags.append(
                            f"{tag_name}: file={file_dtype}, registry={spec.dtype}"
                        )
        
        # Print summary
        print(f"\n  {dng_path.name}: {validated_tags}/{total_tags} tags validated", end="")
        if unknown_tags:
            print(f", {len(unknown_tags)} unknown: {sorted(unknown_tags)}", end="")
        
        # Fail if any incompatible tags found
        assert not incompatible_tags, (
            f"Incompatible tags in {dng_path.name}:\n  " + 
            "\n  ".join(incompatible_tags)
        )
        
        # Fail if any unknown tags found (ensures registry stays complete)
        assert not unknown_tags, (
            f"Tags in {dng_path.name} missing from TIFF_TAG_TYPE_REGISTRY:\n  " +
            "\n  ".join(sorted(unknown_tags))
        )
    
    @staticmethod
    def _tifffile_dtype_to_string(dtype_code) -> str:
        """Convert tifffile dtype code to our string format."""
        # tifffile uses numpy dtype or TIFF type codes
        dtype_map = {
            1: 'B',   # BYTE
            2: 's',   # ASCII
            3: 'H',   # SHORT
            4: 'I',   # LONG
            5: '2I',  # RATIONAL
            6: 'b',   # SBYTE
            7: 'B',   # UNDEFINED (treat as bytes)
            8: 'h',   # SSHORT
            9: 'i',   # SLONG
            10: '2i', # SRATIONAL
            11: 'f',  # FLOAT
            12: 'd',  # DOUBLE
            13: 'I',  # IFD (treat as LONG)
            16: 'Q',  # LONG8
            17: 'q',  # SLONG8
            18: 'Q',  # IFD8 (treat as LONG8)
        }
        if isinstance(dtype_code, int):
            return dtype_map.get(dtype_code, str(dtype_code))
        return str(dtype_code)


class TestTagNameResolution:
    """Test that all registry tag names can be resolved by tifffile."""
    
    def test_all_registry_tags_resolvable(self):
        """Verify all tags in registry can be resolved via LOCAL_TIFF_TAGS."""
        unresolvable = []
        
        for tag_name in TIFF_TAG_TYPE_REGISTRY.keys():
            # Try to look up in LOCAL_TIFF_TAGS (extends TIFF.TAGS with extra tags)
            code = LOCAL_TIFF_TAGS.get(tag_name)
            if code is None:
                unresolvable.append(tag_name)
            else:
                assert isinstance(code, int), f"{tag_name} resolved to non-int: {code}"
        
        assert not unresolvable, (
            f"Tags not resolvable via LOCAL_TIFF_TAGS:\n  " +
            "\n  ".join(unresolvable)
        )


class TestTagSpecTypeInference:
    """Test TagSpec.get_dtype_for_value() type inference."""
    
    def test_single_dtype_returns_itself(self):
        """Single dtype spec always returns that dtype."""
        spec = TagSpec("H", 1)
        assert spec.get_dtype_for_value(42) == "H"
        assert spec.get_dtype_for_value(3.14) == "H"
        assert spec.get_dtype_for_value([1, 2, 3]) == "H"
    
    def test_multi_dtype_int_value(self):
        """Int value selects first int dtype from list."""
        spec = TagSpec(["H", "I", "2I"], None)
        assert spec.get_dtype_for_value(42) == "H"
        assert spec.get_dtype_for_value(np.int32(100)) == "H"
        assert spec.get_dtype_for_value([1, 2, 3]) == "H"
    
    def test_multi_dtype_float_value(self):
        """Float value selects first float/rational dtype from list."""
        spec = TagSpec(["H", "I", "2I"], None)
        assert spec.get_dtype_for_value(3.14) == "2I"
        assert spec.get_dtype_for_value(np.float64(1.5)) == "2I"
        assert spec.get_dtype_for_value([1.0, 2.0]) == "2I"
    
    def test_fallback_to_first(self):
        """Unknown value type falls back to first dtype."""
        spec = TagSpec(["H", "2I"], None)
        assert spec.get_dtype_for_value("string") == "H"
        assert spec.get_dtype_for_value(None) == "H"


class TestValueConversion:
    """Test encode_tag_value() produces correct output."""
    
    def test_string_conversion(self):
        """String values are null-terminated."""
        spec = TagSpec("s", None)
        dtype, count, value = encode_tag_value("TestTag", "Hello", spec)
        assert dtype == "s"
        assert value.endswith("\x00")
        assert count == 6  # "Hello" + null
    
    def test_int_to_short(self):
        """Integer converts to SHORT."""
        spec = TagSpec("H", 1)
        dtype, count, value = encode_tag_value("TestTag", 42, spec)
        assert dtype == "H"
        assert count == 1
        assert value == 42
    
    def test_float_to_rational(self):
        """Float converts to rational tuple."""
        spec = TagSpec("2I", 1)
        dtype, count, value = encode_tag_value("TestTag", 0.5, spec)
        assert dtype == "2I"
        assert count == 1
        # Value should be (numerator, denominator) representing 0.5
        assert isinstance(value, tuple)
        assert len(value) == 2
        assert value[0] / value[1] == pytest.approx(0.5)
    
    def test_array_to_rationals(self):
        """Float array converts to rational array."""
        spec = TagSpec("2I", 3)
        values = [1.0, 0.5, 0.25]
        dtype, count, result = encode_tag_value("TestTag", values, spec)
        assert dtype == "2I"
        assert count == 3
        # Result should be flat tuple of (num, denom, num, denom, num, denom)
        assert len(result) == 6
    
    def test_numpy_array_handling(self):
        """NumPy arrays are flattened correctly."""
        spec = TagSpec("2i", None)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        dtype, count, result = encode_tag_value("TestTag", matrix, spec)
        assert dtype == "2i"
        assert count == 4  # 2x2 matrix = 4 elements
    
    def test_bytes_passthrough(self):
        """Bytes pass through unchanged."""
        spec = TagSpec("B", None)
        data = b"\x01\x02\x03\x04"
        dtype, count, value = encode_tag_value("TestTag", data, spec)
        assert dtype == "B"
        assert value == data


class TestMatrixTagCount:
    """Test that matrix tags have correct count in registry for proper reshaping."""
    
    # 3x3 matrix tags that MUST have count=9
    MATRIX_3X3_TAGS = [
        "ColorMatrix1", "ColorMatrix2", "ColorMatrix3",
        "ForwardMatrix1", "ForwardMatrix2", "ForwardMatrix3",
        "CameraCalibration1", "CameraCalibration2", "CameraCalibration3",
    ]
    
    def test_matrix_tags_have_count_9(self):
        """Verify all 3x3 matrix tags have count=9 for proper reshaping."""
        missing_count = []
        wrong_count = []
        
        for tag_name in self.MATRIX_3X3_TAGS:
            if tag_name not in TIFF_TAG_TYPE_REGISTRY:
                continue  # Skip if not in registry
            
            spec = TIFF_TAG_TYPE_REGISTRY[tag_name]
            if spec.count is None:
                missing_count.append(f"{tag_name}: count=None (should be 9)")
            elif spec.count != 9:
                wrong_count.append(f"{tag_name}: count={spec.count} (should be 9)")
        
        errors = missing_count + wrong_count
        assert not errors, (
            f"Matrix tags with incorrect count (causes reshape failures):\n  " +
            "\n  ".join(errors)
        )
    
    @pytest.mark.parametrize("dng_path", get_dng_files(), ids=lambda p: p.name)
    def test_matrix_tag_count_matches_file(self, dng_path: Path):
        """Verify registry count matches actual tag count in DNG files."""
        mismatches = []
        
        with muimg.DngFile(dng_path) as dng:
            for page in dng.get_flattened_pages():
                for tag in page.tags.values():
                    tag_name = tag.name
                    
                    if tag_name not in TIFF_TAG_TYPE_REGISTRY:
                        continue
                    
                    spec = TIFF_TAG_TYPE_REGISTRY[tag_name]
                    if spec.count is None:
                        continue  # Variable count, skip
                    
                    # For RATIONAL/SRATIONAL, count is number of rationals, not raw values
                    file_count = tag.count
                    if spec.count != file_count:
                        mismatches.append(
                            f"{tag_name}: registry={spec.count}, file={file_count}"
                        )
        
        assert not mismatches, (
            f"Tag count mismatches in {dng_path.name}:\n  " +
            "\n  ".join(mismatches)
        )
