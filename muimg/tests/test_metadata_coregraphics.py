# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Test Core Graphics metadata writing for all TIFF tag types."""

import pytest
import numpy as np
import tifffile
from pathlib import Path

try:
    from Foundation import NSNumber, NSString, NSData, NSArray, NSMutableDictionary
    from Quartz import (
        CGImageDestinationCreateWithURL,
        CGImageDestinationAddImage,
        CGImageDestinationFinalize,
        CGImageCreate,
        CGColorSpaceCreateWithName,
        CGDataProviderCreateWithData,
        kCGImagePropertyTIFFDictionary,
        kCGImagePropertyExifDictionary,
        kCGColorSpaceSRGB,
    )
    from CoreFoundation import CFURLCreateFromFileSystemRepresentation, kCFAllocatorDefault
    COREGRAPHICS_AVAILABLE = True
except ImportError:
    COREGRAPHICS_AVAILABLE = False

from muimg.tiff_metadata import TIFF_TAG_TYPE_REGISTRY, TiffType

# Import test utilities
from conftest import OutputPathManager

# Create output path manager for this test file (persistent outputs for debugging)
output_path_manager = OutputPathManager(persistent=False)


def generate_synthetic_foundation_value(tag_spec):
    """Generate synthetic Foundation object for a given tag specification.
    
    Args:
        tag_spec: TagSpec defining the tag type and count
        
    Returns:
        Foundation object (NSNumber, NSString, NSArray, etc.)
    """
    if not COREGRAPHICS_AVAILABLE:
        pytest.skip("Core Graphics not available")
    
    # Get the dtype (handle both single type and list of types)
    dtype = tag_spec.dtype if isinstance(tag_spec.dtype, TiffType) else tag_spec.dtype[0]
    # For variable-length tags (count=None), use a reasonable default
    if tag_spec.count is None:
        # Use larger defaults for variable-length data
        count = 16 if dtype in (TiffType.BYTE, TiffType.UNDEFINED) else 4
    else:
        count = tag_spec.count
    
    if dtype == TiffType.BYTE:
        # BYTE arrays should be NSData, not NSArray
        if count > 1:
            data = bytes(range(42, 42 + count))
            ns_data = NSData.dataWithBytes_length_(data, len(data))
            return NSData.alloc().initWithData_(ns_data)
        return NSNumber.numberWithInt_(42)
    
    elif dtype == TiffType.ASCII:
        return NSString.stringWithString_("TestValue123")
    
    elif dtype == TiffType.SHORT:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(100 + i) for i in range(count)])
        return NSNumber.numberWithInt_(100)
    
    elif dtype == TiffType.LONG:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(1000 + i) for i in range(count)])
        return NSNumber.numberWithInt_(1000)
    
    elif dtype == TiffType.RATIONAL:
        # Core Graphics expects rationals as floats
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithDouble_(10.0 + i) for i in range(count)])
        return NSNumber.numberWithDouble_(10.0)
    
    elif dtype == TiffType.SBYTE:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(-10 + i) for i in range(count)])
        return NSNumber.numberWithInt_(-10)
    
    elif dtype == TiffType.UNDEFINED:
        # UNDEFINED is a byte buffer with undefined interpretation
        # Special handling for specific EXIF tags
        
        # Get tag name if available (check parent scope)
        # For now, use count as a heuristic
        if count >= 16:
            # Variable-length UNDEFINED - likely MakerNote or UserComment
            # UserComment requires 8-byte encoding prefix
            prefix = b"ASCII\x00\x00\x00"
            data = prefix + bytes([42] * (count - 8))
        else:
            data = bytes([42] * count)
        ns_data = NSData.dataWithBytes_length_(data, len(data))
        return NSData.alloc().initWithData_(ns_data)
    
    elif dtype == TiffType.SSHORT:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(-100 + i) for i in range(count)])
        return NSNumber.numberWithInt_(-100)
    
    elif dtype == TiffType.SLONG:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(-1000 + i) for i in range(count)])
        return NSNumber.numberWithInt_(-1000)
    
    elif dtype == TiffType.SRATIONAL:
        # Core Graphics expects signed rationals as floats
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithDouble_(-10.0 - i) for i in range(count)])
        return NSNumber.numberWithDouble_(-10.0)
    
    elif dtype == TiffType.FLOAT:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithDouble_(float(i) + 0.5) for i in range(count)])
        return NSNumber.numberWithDouble_(1.5)
    
    elif dtype == TiffType.DOUBLE:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithDouble_(float(i) + 0.25) for i in range(count)])
        return NSNumber.numberWithDouble_(2.25)
    
    elif dtype == TiffType.LONG8:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(10000 + i) for i in range(count)])
        return NSNumber.numberWithInt_(10000)
    
    elif dtype == TiffType.SLONG8:
        if count > 1:
            return NSArray.arrayWithArray_([NSNumber.numberWithInt_(-10000 + i) for i in range(count)])
        return NSNumber.numberWithInt_(-10000)
    
    else:
        return NSString.stringWithString_("Unknown")


def create_test_image():
    """Create a synthetic RGB gradient image for testing.
    
    Returns:
        numpy array of shape (256, 256, 3) with uint8 dtype
    """
    # Create horizontal gradient
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)
    r = np.tile(gradient, (256, 1))
    g = np.tile(gradient[::-1], (256, 1))
    b = np.ones((256, 256), dtype=np.uint8) * 128
    
    return np.stack([r, g, b], axis=-1)


def write_image_with_coregraphics(image, output_path, tiff_dict=None, exif_dict=None):
    """Write image using Core Graphics with metadata dictionaries.
    
    Args:
        image: numpy array (H, W, 3) uint8
        output_path: Path to output file
        tiff_dict: NSMutableDictionary for TIFF tags
        exif_dict: NSMutableDictionary for EXIF tags
        
    Returns:
        bool: Success status
    """
    if not COREGRAPHICS_AVAILABLE:
        return False
    
    # Create CGImage from numpy array
    height, width = image.shape[:2]
    bytes_per_row = width * 3
    
    # Create data provider from image data
    image_data = image.tobytes()
    data_provider = CGDataProviderCreateWithData(None, image_data, len(image_data), None)
    
    # Create color space
    color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB)
    
    # Create CGImage
    cg_image = CGImageCreate(
        width, height,
        8, 24, bytes_per_row,
        color_space,
        0,  # No alpha
        data_provider,
        None, False, 0
    )
    
    # Create URL for output
    output_bytes = bytes(str(output_path), 'utf-8')
    url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        output_bytes,
        len(output_bytes),
        False
    )
    
    # Determine format from file extension
    if str(output_path).endswith('.jpg') or str(output_path).endswith('.jpeg'):
        uti = 'public.jpeg'
    else:
        uti = 'public.tiff'
    
    # Create image destination
    dest = CGImageDestinationCreateWithURL(url, uti, 1, None)
    if not dest:
        return False
    
    # Build properties dictionary
    properties = NSMutableDictionary.dictionary()
    if tiff_dict and tiff_dict.count() > 0:
        properties.setObject_forKey_(tiff_dict, kCGImagePropertyTIFFDictionary)
    if exif_dict and exif_dict.count() > 0:
        properties.setObject_forKey_(exif_dict, kCGImagePropertyExifDictionary)
    
    # Add image with properties
    CGImageDestinationAddImage(dest, cg_image, properties)
    
    # Finalize
    success = CGImageDestinationFinalize(dest)
    return bool(success)


@pytest.mark.xfail(reason="have not fully debugged coregraphics metadata")
def test_all_tag_types_coregraphics(tmp_path):
    """Test Core Graphics metadata writing for all TIFF tag types.
    
    For each tag type in the registry with dng_ifd='exif' or 'ifd0',
    write synthetic data and verify it can be read back.
    """
    if not COREGRAPHICS_AVAILABLE:
        pytest.skip("Core Graphics not available")
    
    print("\n=== Starting Core Graphics metadata test ===")
    
    # Create synthetic test image
    test_image = create_test_image()
    print(f"Created test image: {test_image.shape}")
    
    results = {
        "tested": 0,
        "written": 0,
        "read_back": 0,
        "failed": []
    }
    
    # Track results per type
    results_by_type = {}
    
    print(f"Total TiffType enum values: {len(list(TiffType))}")
    
    # Iterate through each TiffType
    for tiff_type in TiffType:
        print(f"\nProcessing type: {tiff_type.name}")
        # Find all tags with this type in exif or ifd0
        for tag_name, tag_spec in TIFF_TAG_TYPE_REGISTRY.items():
            # Skip if not in exif or ifd0
            if tag_spec.dng_ifd not in ("exif", "ifd0"):
                continue
            
            # Skip SubIFD pointer tags (Core Graphics manages these internally)
            if tag_name in ("ExifTag", "GPSTag"):
                continue
            
            # Get dtype (handle both single type and list of types)
            dtype = tag_spec.dtype if isinstance(tag_spec.dtype, TiffType) else tag_spec.dtype[0]
            
            # Skip if not the current type we're testing
            if dtype != tiff_type:
                continue
            
            results["tested"] += 1
            
            # Initialize type stats if needed
            if dtype.name not in results_by_type:
                results_by_type[dtype.name] = {
                    "tested": 0,
                    "written": 0,
                    "read_back": 0,
                    "failed": 0
                }
            results_by_type[dtype.name]["tested"] += 1
            
            # Generate synthetic Foundation value
            try:
                foundation_value = generate_synthetic_foundation_value(tag_spec)
            except Exception as e:
                results["failed"].append({
                    "tag": tag_name,
                    "type": dtype.name,
                    "stage": "generate_value",
                    "error": str(e)
                })
                results_by_type[dtype.name]["failed"] += 1
                continue
            
            # Create metadata dictionaries
            tiff_dict = NSMutableDictionary.dictionary()
            exif_dict = NSMutableDictionary.dictionary()
            
            # Add a "trigger" EXIF tag to force Core Graphics to create the EXIF sub-IFD
            # Without this, Core Graphics may skip the EXIF sub-IFD entirely for TIFFs
            exif_dict.setObject_forKey_(NSString.stringWithString_("2026:04:23 14:40:30"), "DateTimeOriginal")
            
            # Add tag to appropriate dictionary
            if tag_spec.dng_ifd == "ifd0":
                tiff_dict.setObject_forKey_(foundation_value, tag_name)
            elif tag_spec.dng_ifd == "exif":
                exif_dict.setObject_forKey_(foundation_value, tag_name)
            
            # Write image with metadata
            output_dir = output_path_manager.get_path(tmp_path, "test_all_tag_types_coregraphics")
            output_file = output_dir / f"{tag_name}.jpg"
            try:
                success = write_image_with_coregraphics(test_image, output_file, tiff_dict, exif_dict)
                if not success:
                    results["failed"].append({
                        "tag": tag_name,
                        "type": dtype.name,
                        "stage": "write_image",
                        "error": "write_image_with_coregraphics returned False"
                    })
                    results_by_type[dtype.name]["failed"] += 1
                    continue
                results["written"] += 1
                results_by_type[dtype.name]["written"] += 1
            except Exception as e:
                results["failed"].append({
                    "tag": tag_name,
                    "type": dtype.name,
                    "stage": "write_image",
                    "error": str(e)
                })
                results_by_type[dtype.name]["failed"] += 1
                continue
            
            # Read back and verify using exiftool (works for both TIFF and JPEG)
            try:
                import subprocess
                result = subprocess.run(
                    ['/usr/local/bin/exiftool', '-s', '-G1', str(output_file)],
                    capture_output=True,
                    text=True
                )
                
                # Check if tag name appears in exiftool output (not in filename)
                # Look for lines that start with [Group] and contain the tag name
                found = False
                for line in result.stdout.split('\n'):
                    if line.startswith('[') and tag_name in line:
                        # Make sure it's the tag name, not just part of the value
                        parts = line.split(':', 1)
                        if len(parts) == 2 and tag_name in parts[0]:
                            found = True
                            break
                
                if found:
                    results["read_back"] += 1
                    results_by_type[dtype.name]["read_back"] += 1
                else:
                    results["failed"].append({
                        "tag": tag_name,
                        "type": dtype.name,
                        "stage": "read_back",
                        "error": "Tag not found in output"
                    })
                    results_by_type[dtype.name]["failed"] += 1
            except Exception as e:
                results["failed"].append({
                    "tag": tag_name,
                    "type": dtype.name,
                    "stage": "read_back",
                    "error": str(e)
                })
                results_by_type[dtype.name]["failed"] += 1
    
    # Print summary
    print(f"\n=== TIFF Test Results ===")
    print(f"Tags tested: {results['tested']}")
    print(f"Successfully written: {results['written']}")
    print(f"Successfully read back: {results['read_back']}")
    print(f"Failed: {len(results['failed'])}")
    
    # Print per-type summary
    print(f"\n=== Results by Type ===")
    for type_name in sorted(results_by_type.keys()):
        stats = results_by_type[type_name]
        print(f"{type_name:12s}: {stats['tested']:3d} tested, {stats['written']:3d} written, {stats['read_back']:3d} read back, {stats['failed']:3d} failed")
    
    if results["failed"]:
        print("\nFailures:")
        for failure in results["failed"]:
            print(f"  {failure['tag']} ({failure['type']}): {failure['stage']} - {failure['error']}")
    
    # Assert that ALL tags were successfully written and read back
    assert len(results["failed"]) == 0, f"{len(results['failed'])} tags failed to roundtrip correctly"
