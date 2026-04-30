# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Test endianness handling for array tags in MetadataTags.

This test creates TIFF files with both big-endian and little-endian byte orders,
writes various array tags to them, and verifies that MetadataTags correctly
preserves and handles byte order information.

Uses private tag codes (65000-65535) to avoid conflicts with standard TIFF tags.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import struct
import tifffile

import sys
import muimg
from muimg.tiff_metadata import MetadataTags
from muimg.dngio import write_dng_from_array, DngFile, IfdDataSpec


# Private tag codes (65000-65535 are reserved for private use)
TAG_TEST_BYTE_ARRAY = 65000
TAG_TEST_FLOAT_ARRAY = 65001
TAG_TEST_UINT32_ARRAY = 65002


def read_tags_with_normalization(tiff_page):
    """Helper to read tags from a tifffile page and normalize to system byte order.
    
    This simulates what DngPage.get_page_tags() does.
    """
    from muimg.tiff_metadata import normalize_array_to_target_byteorder
    
    tags = MetadataTags()
    for tag in tiff_page.tags.values():
        # Normalize value to system byte order
        normalized_value = normalize_array_to_target_byteorder(tag.value, '=')
        tags.add_raw_tag(tag.code, tag.dtype, tag.count, normalized_value)
    return tags


def create_test_tiff(path: Path, byteorder: str):
    """Create a test TIFF file with byte, float, and uint32 array tags."""
    # Create test data
    data = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    
    # Test values for byte array
    byte_array_values = struct.pack(f'{byteorder}IIII', 0x12345678, 0xABCDEF00, 0xDEADBEEF, 0xCAFEBABE)
    
    # Test values for float array (100 elements)
    float_array = np.arange(100, dtype=f'{byteorder}f4') + 1.5
    
    # Test values for uint32 array (100 elements)
    # Use struct format 'I' instead of 'u4' to get correct dtype.char on Windows
    uint32_array = np.arange(100, dtype=f'{byteorder}I') + 100
    
    # Write TIFF file with specified byte order
    with tifffile.TiffWriter(path, byteorder=byteorder) as tif:
        tif.write(
            data,
            photometric='minisblack',
            metadata=None,
            extratags=[
                (TAG_TEST_BYTE_ARRAY, 1, len(byte_array_values), byte_array_values, False),  # BYTE
                (TAG_TEST_FLOAT_ARRAY, 11, len(float_array), float_array, False),  # FLOAT
                (TAG_TEST_UINT32_ARRAY, 4, len(uint32_array), uint32_array, False),  # LONG
            ]
        )


@pytest.mark.parametrize("dtype_name,np_dtype,tiff_dtype,small_size,large_size", [
    ("bytes", "u1", 1, 5, 2),
    ("int16", "u2", 3, 5, 1025),
    ("int32", "u4", 4, 5, 1025),
    ("float", "f4", 11, 5, 1025),
])
@pytest.mark.parametrize("array_size_type", ["small", "large"])
@pytest.mark.parametrize("input_format", ["tuple", "array_native", "array_le", "array_be"])
@pytest.mark.parametrize("source_order,dest_order,direction", [
    (">", "<", "BE→LE"),
    ("<", ">", "LE→BE"),
    (">", ">", "BE→BE"),
    ("<", "<", "LE→LE"),
])
def test_endian_roundtrip_matrix(
    dtype_name, np_dtype, tiff_dtype, small_size, large_size,
    array_size_type, input_format, source_order, dest_order, direction
):
    """Comprehensive test matrix for endianness handling across all data types and scenarios.
    
    Tests all combinations of:
    - Data types: bytes, int16, int32, float
    - Array sizes: small (returns tuples) and large (returns numpy arrays)
    - Input formats: tuple, array with '=', '<', '>' byteorder
    - Byte order conversions: BE→LE, LE→BE, BE→BE, LE→LE
    
    Expected failures (current implementation):
    - Arrays with byteorder='=' when source_order != dest_order
    - Byte arrays in cross-endian scenarios
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source_name = 'BE' if source_order == '>' else 'LE'
        dest_name = 'BE' if dest_order == '>' else 'LE'
        source_path = tmpdir / f"source_{source_name}.tif"
        dest_path = tmpdir / f"dest_{dest_name}.tif"
        
        # Determine array size
        size = small_size if array_size_type == "small" else large_size
        
        # Create test values
        base_values = np.arange(size, dtype=f'={np_dtype}') + 1
        expected_values = base_values[:5]  # First 5 for comparison
        
        # Create input array in specified format
        if input_format == "tuple":
            input_array = tuple(base_values.tolist())
        elif input_format == "array_native":
            input_array = np.array(base_values, dtype=f'={np_dtype}')
        elif input_format == "array_le":
            input_array = np.array(base_values, dtype=f'<{np_dtype}')
        elif input_format == "array_be":
            input_array = np.array(base_values, dtype=f'>{np_dtype}')
        
        # Helper to prepare tags for writing (convert to target byte order)
        def prepare_tags_for_write(tags: MetadataTags, target_byteorder: str):
            from muimg.tiff_metadata import normalize_array_to_target_byteorder
            for code in sorted(tags._tags.keys()):
                tag = tags._tags[code]
                value = normalize_array_to_target_byteorder(tag.value, target_byteorder)
                yield (tag.code, tag.dtype, tag.count, value, False)
        
        # Write to source file
        data = np.arange(100, dtype=np.uint16).reshape(10, 10)
        with tifffile.TiffWriter(source_path, byteorder=source_order) as tif:
            # Prepare tags with correct byte order for source file
            source_tags_obj = MetadataTags()
            source_tags_obj.add_raw_tag(65001, tiff_dtype, size, input_array)
            source_extratags = list(prepare_tags_for_write(source_tags_obj, source_order))
            tif.write(
                data,
                photometric='minisblack',
                extratags=source_extratags,
            )
        
        # Read via MetadataTags (filter out managed tags to avoid warnings)
        from tifffile import TIFF
        from muimg.dngio import _TIFFWRITER_MANAGED_TAGS
        managed_codes = {TIFF.TAGS.get(name) for name in _TIFFWRITER_MANAGED_TAGS if name in TIFF.TAGS}
        
        with tifffile.TiffFile(source_path) as source_tif:
            source_tags = MetadataTags()
            for tag in source_tif.pages[0].tags.values():
                if tag.code not in managed_codes:
                    source_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
        
        # Write to dest file
        with tifffile.TiffWriter(dest_path, byteorder=dest_order) as tif:
            # Prepare tags with correct byte order for dest file
            dest_extratags = list(prepare_tags_for_write(source_tags, dest_order))
            tif.write(
                data,
                photometric='minisblack',
                extratags=dest_extratags
            )
        
        # Read back via tifffile
        with tifffile.TiffFile(dest_path) as dest_tif:
            tifffile_value = dest_tif.pages[0].tags[65001].value
            
            # Read back via MetadataTags
            dest_tags = MetadataTags()
            for tag in dest_tif.pages[0].tags.values():
                dest_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
            metadata_value = None
            for code, dtype, count, value, _ in dest_tags:
                if code == 65001:
                    metadata_value = value
                    break
        
        # Convert to arrays for comparison
        if isinstance(tifffile_value, bytes):
            # Byte arrays returned as bytes objects
            tifffile_array = np.frombuffer(tifffile_value[:5], dtype=np.uint8)
        elif isinstance(tifffile_value, tuple):
            tifffile_array = np.array(tifffile_value[:5], dtype=f'={np_dtype}')
        else:
            tifffile_array = tifffile_value[:5]
        
        if isinstance(metadata_value, bytes):
            # Byte arrays returned as bytes objects
            metadata_array = np.frombuffer(metadata_value[:5], dtype=np.uint8)
        elif isinstance(metadata_value, tuple):
            metadata_array = np.array(metadata_value[:5], dtype=f'={np_dtype}')
        else:
            metadata_array = metadata_value[:5]
        
        # Assert values match expected
        test_id = f"{dtype_name}/{array_size_type}/{input_format}/{direction}"
        
        # Verify byte order if array
        # Note: tifffile converts arrays to native byte order when reading,
        # so we expect '<' on LE systems or '>' on BE systems, or '='
        # Single-byte types (uint8) have byteorder='|' (not applicable)
        if isinstance(tifffile_array, np.ndarray):
            system_byteorder = '<' if sys.byteorder == 'little' else '>'
            actual_byteorder = tifffile_array.dtype.byteorder
            assert actual_byteorder in (system_byteorder, '=', '|'), \
                f"[{test_id}] Expected byteorder {system_byteorder}, '=', or '|', got {actual_byteorder}"
        
        # Verify values
        np.testing.assert_array_almost_equal(tifffile_array, expected_values, decimal=5,
            err_msg=f"[{test_id}] tifffile values corrupted")
        np.testing.assert_array_almost_equal(metadata_array, expected_values, decimal=5,
            err_msg=f"[{test_id}] MetadataTags values corrupted")


def test_float_array_native_byteorder_bug():
    """Test that writing float arrays with byteorder='=' to opposite-endian files works correctly.
    
    This verifies the fix for the Canon DNG bug:
    1. Create a float array with byteorder='=' (like tifffile returns for offset-stored arrays)
    2. Write it to a big-endian file via MetadataTags.prepare_for_write()
    3. Read it back and verify values are preserved
    
    Note: tifffile has a threshold at 1024 elements for multi-byte types:
    - Multi-byte arrays (float32, float64, uint32, uint16) with ≤1024 elements: 
      stored inline in IFD, returned as tuples
    - Multi-byte arrays with ≥1025 elements: 
      stored at file offset, returned as numpy arrays with byteorder='='
    - Single-byte arrays (uint8/BYTE): always stored at offset, returned as ndarray
    
    We use 8100 elements to match the Canon DNG file and trigger the numpy array path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dest_path = tmpdir / "test_be.tif"
        
        # Create float array with native byte order (byteorder='=')
        # This simulates what tifffile returns when reading large float arrays from DNG files
        # Use 8100 elements to match Canon DNG file size
        original_values = np.arange(8100, dtype='=f4') + 1.5
        
        print(f"\nOriginal array: dtype={original_values.dtype}, byteorder={original_values.dtype.byteorder}, shape={original_values.shape}")
        print(f"First 5 values: {original_values[:5]}")
        
        # Verify it has native byte order
        assert original_values.dtype.byteorder == '=', \
            f"Array should have native byteorder '=', got '{original_values.dtype.byteorder}'"
        
        # Helper to prepare tags for writing (convert to target byte order)
        def prepare_tags_for_write(tags: MetadataTags, target_byteorder: str):
            from muimg.tiff_metadata import normalize_array_to_target_byteorder
            for code in sorted(tags._tags.keys()):
                tag = tags._tags[code]
                value = normalize_array_to_target_byteorder(tag.value, target_byteorder)
                yield (tag.code, tag.dtype, tag.count, value, False)
        
        # Create MetadataTags and add the float array
        tags = MetadataTags()
        tags.add_raw_tag(65001, 11, len(original_values), original_values)  # dtype=11 is FLOAT
        
        # Create simple image data
        data = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        
        # Write to big-endian file using helper
        with tifffile.TiffWriter(dest_path, byteorder='>') as tif:
            tif.write(
                data,
                photometric='minisblack',
                metadata=None,
                extratags=list(prepare_tags_for_write(tags, '>'))
            )
        
        print(f"Wrote to big-endian file: {dest_path}")

        with tifffile.TiffFile(dest_path) as dest_tif:
            dest_page = dest_tif.pages[0]
            
            # Verify file is big-endian
            assert dest_tif.byteorder == '>', "Destination file should be big-endian"
            print(f"File byteorder: {dest_tif.byteorder}")
            
            # Get the float array back
            read_value = dest_page.tags[65001].value
            
            print(f"\nRead back: type={type(read_value)}")
            if isinstance(read_value, np.ndarray):
                print(f"  dtype={read_value.dtype}, byteorder={read_value.dtype.byteorder}")
            
            if isinstance(read_value, tuple):
                print(f"  Converting tuple to array")
                read_value = np.array(read_value, dtype=np.float32)
            
            if isinstance(read_value, np.ndarray):
                print(f"Read values (first 5): {read_value[:5]}")
            else:
                print(f"Read values (first 5): {read_value[:5]}")
            
            # This should FAIL because the values will be corrupted
            # Original first 5: [1.5, 2.5, 3.5, 4.5, 5.5]
            # Corrupted: [8.96831e-44, 4.60060e-41, ...] (garbage)
            print(f"\nComparing first 5 values:")
            print(f"  Expected: {original_values[:5]}")
            if isinstance(read_value, np.ndarray):
                print(f"  Got:      {read_value[:5]}")
            else:
                print(f"  Got:      {np.array(read_value[:5])}")
            
            # Compare first 5 values
            expected_first_5 = original_values[:5]
            if isinstance(read_value, np.ndarray):
                actual_first_5 = read_value[:5]
            else:
                actual_first_5 = np.array(read_value[:5], dtype=np.float32)
            
            np.testing.assert_array_almost_equal(actual_first_5, expected_first_5, decimal=5,
                err_msg=f"Float array corrupted when writing byteorder='=' to big-endian file. "
                        f"Expected {expected_first_5}, got {actual_first_5}")


def test_byte_array_endianness():
    """Test that byte arrays with structured data do NOT preserve byte order.
    
    This test verifies the known limitation: generic BYTE arrays are opaque
    binary data without type information, so we cannot automatically byte-swap them.
    Only special tags like ProfileGainTableMap are handled explicitly.
    
    This test expects byte arrays to have byteorder='|' (byte-order agnostic).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create big-endian TIFF
        be_path = tmpdir / "test_be.tif"
        create_test_tiff(be_path, '>')
        
        # Create little-endian TIFF
        le_path = tmpdir / "test_le.tif"
        create_test_tiff(le_path, '<')
        
        # Read big-endian file
        with tifffile.TiffFile(be_path) as be_tif:
            be_page = be_tif.pages[0]
            
            # Get tags through MetadataTags
            be_tags = MetadataTags()
            for tag in be_page.tags.values():
                be_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
            
            # Get byte array
            be_byte_array = be_tags.get_tag(TAG_TEST_BYTE_ARRAY)
        
        # Byte arrays should be returned as bytes or ndarray with byteorder='|'
        if isinstance(be_byte_array, bytes):
            # Bytes are opaque - we can't know the internal structure
            # This is expected behavior
            pass
        elif isinstance(be_byte_array, np.ndarray):
            # Should have byte-order agnostic marker
            assert be_byte_array.dtype.byteorder == '|', \
                f"Byte array should have byteorder='|', got '{be_byte_array.dtype.byteorder}'"
        else:
            pytest.fail(f"Unexpected type for byte array: {type(be_byte_array)}")
        
        # Read little-endian file
        with tifffile.TiffFile(le_path) as le_tif:
            le_page = le_tif.pages[0]
            
            le_tags = MetadataTags()
            for tag in le_page.tags.values():
                le_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
            
            le_byte_array = le_tags.get_tag(TAG_TEST_BYTE_ARRAY)
        
        # Byte arrays should be returned as bytes or ndarray with byteorder='|'
        if isinstance(le_byte_array, bytes):
            # Bytes are opaque - we can't know the internal structure
            # This is expected behavior
            pass
        elif isinstance(le_byte_array, np.ndarray):
            # Should have byte-order agnostic marker
            assert le_byte_array.dtype.byteorder == '|', \
                f"Byte array should have byteorder='|', got '{le_byte_array.dtype.byteorder}'"
        else:
            pytest.fail(f"Unexpected type for byte array: {type(le_byte_array)}")


def test_cross_endian_detection():
    """Test that byte arrays do NOT have byte order information (known limitation).
    
    This test verifies that byte arrays from big-endian and little-endian files
    are NOT distinguishable because they're opaque binary data.
    Only special tags like ProfileGainTableMap are handled explicitly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create both files
        be_path = tmpdir / "test_be.tif"
        create_test_tiff(be_path, '>')
        
        le_path = tmpdir / "test_le.tif"
        create_test_tiff(le_path, '<')
        
        # Read both
        with tifffile.TiffFile(be_path) as be_tif, tifffile.TiffFile(le_path) as le_tif:
            be_tags = MetadataTags()
            for tag in be_tif.pages[0].tags.values():
                be_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
            
            le_tags = MetadataTags()
            for tag in le_tif.pages[0].tags.values():
                le_tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
            
            # Get byte arrays
            be_bytes = be_tags.get_tag(TAG_TEST_BYTE_ARRAY)
            le_bytes = le_tags.get_tag(TAG_TEST_BYTE_ARRAY)
        
        # Byte arrays should NOT have byte order information (expected limitation)
        if isinstance(be_bytes, np.ndarray) and isinstance(le_bytes, np.ndarray):
            # If they're numpy arrays, they should both be byte-order agnostic
            assert be_bytes.dtype.byteorder == '|', \
                f"Byte array should have byteorder='|', got '{be_bytes.dtype.byteorder}'"
            assert le_bytes.dtype.byteorder == '|', \
                f"Byte array should have byteorder='|', got '{le_bytes.dtype.byteorder}'"
        elif isinstance(be_bytes, bytes) and isinstance(le_bytes, bytes):
            # Plain bytes - expected behavior, they're opaque
            pass
        else:
            pytest.fail(f"Unexpected types: be={type(be_bytes)}, le={type(le_bytes)}")


@pytest.mark.parametrize("dtype_name,np_dtype,tiff_dtype", [
    ("int16", "u2", 3),
    ("int32", "u4", 4),
    ("float", "f4", 11),
])
@pytest.mark.parametrize("input_format", ["tuple", "array_native", "array_le", "array_be"])
def test_dng_cross_endian_write(dtype_name, np_dtype, tiff_dtype, input_format):
    """Test DngFile API cross-endian writes with array tags.
    
    This test uses write_dng_from_array() to write DNGs with custom tags
    containing array data with various input formats. Since write_dng() 
    hardcodes byteorder='>' and we're on a little-endian system, this is a 
    cross-endian write.
    
    Tests that add_raw_tag() can handle arrays with any input format (tuple, 
    native/LE/BE byte order arrays).
    
    Byte arrays are excluded from this test as they cannot be automatically byte-swapped.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dng_path = tmpdir / f"test_{dtype_name}_{input_format}.dng"
        
        # Create test array data in specified format (matching test_endian_roundtrip_matrix)
        size = 10
        base_values = np.arange(size, dtype=f'={np_dtype}') + 1
        
        if input_format == "tuple":
            test_values = tuple(base_values.tolist())
        elif input_format == "array_native":
            test_values = np.array(base_values, dtype=f'={np_dtype}')
        elif input_format == "array_le":
            test_values = np.array(base_values, dtype=f'<{np_dtype}')
        elif input_format == "array_be":
            test_values = np.array(base_values, dtype=f'>{np_dtype}')
        
        expected_values = base_values[:5]  # First 5 for comparison
        
        # Create MetadataTags and add the test array
        metadata = MetadataTags()
        metadata.add_raw_tag(65001, tiff_dtype, size, test_values)
        
        # Create minimal RGB image data (10x10x3 uint16) for LINEAR_RAW
        image_data = np.arange(300, dtype=np.uint16).reshape(10, 10, 3)
        
        # Write DNG using write_dng_from_array
        # write_dng() hardcodes byteorder='>' so this is a cross-endian write on LE systems
        data_spec = IfdDataSpec(
            data=image_data,
            photometric='LINEAR_RAW',
            extratags=metadata,
        )
        write_dng_from_array(destination_file=dng_path, data_spec=data_spec)
        
        # Read back with DngFile
        with DngFile(dng_path) as dng:
            # Extract tag 65001 value from IFD0
            ifd0_tags = dng.ifd0.get_ifd0_tags()
            tag_value = ifd0_tags._tags[65001].value
            
            # Convert to array if needed
            if isinstance(tag_value, tuple):
                readback_array = np.array(tag_value[:5], dtype=f'={np_dtype}')
            else:
                readback_array = tag_value[:5]
            
            # Verify values match expected (not byte-swapped garbage)
            np.testing.assert_array_almost_equal(
                readback_array, 
                expected_values, 
                decimal=5,
                err_msg=f"[{dtype_name}] DngFile cross-endian write corrupted values"
            )
