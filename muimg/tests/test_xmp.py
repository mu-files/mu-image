from __future__ import annotations

from pathlib import Path

import pytest

import muimg

from conftest import TEST_FILES_DIR


DNG_FILENAME = "asi676mc.cfa.jxl_lossy.2ifds.dng"


@pytest.mark.parametrize("filename", [DNG_FILENAME])
def test_dump_xmp_properties(filename: str):
    dng_path = TEST_FILES_DIR / filename
    if not dng_path.exists():
        pytest.skip(f"Test file not available: {filename}")

    with muimg.DngFile(dng_path) as dng:
        main_page = dng.get_main_page()
        assert main_page is not None

        xmp = main_page.get_xmp()
        assert xmp is not None
        props = getattr(xmp, "_attributes", None)
        assert isinstance(props, dict)

        print(f"XMP property count: {len(props)}")
        for key in sorted(props.keys()):
            value = props[key]
            if isinstance(value, list):
                print(f"{key} (list, len={len(value)}):")
                for item in value:
                    print(f"  {item}")
            else:
                print(f"{key}={value}")


def test_xmp_roundtrip_metadatatags():
    """Test round-tripping XMP values through MetadataTags.
    
    Creates XMP from dict, adds to MetadataTags, retrieves it, converts back to dict,
    and validates values match the original.
    """
    # Create test data with various XMP property types
    original_dict = {
        # Supported properties (with pixel processing)
        'Temperature': 5800,
        'Tint': 10,
        'Exposure2012': -0.5,
        'ToneCurvePV2012': [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)],
        
        # Unsupported properties (no pixel processing yet, but should still round-trip)
        'Contrast2012': 25,
        'Highlights2012': -50,
        'Shadows2012': 30,
        'Vibrance': 15,
        'Saturation': 10,
        
        # Qualified properties
        'dc:subject': ['key_in=0.18', 'cloud_okta=3.5', 'test_tag'],
        'tiff:Orientation': '1',
    }
    
    # Step 1: Create XmpMetadata from dict
    xmp_metadata = muimg.supported_xmp_from_dict(original_dict)
    
    # Validate XmpMetadata was created
    assert xmp_metadata is not None
    assert isinstance(xmp_metadata, muimg.XmpMetadata)
    
    # Step 2: Add to MetadataTags
    tags = muimg.MetadataTags()
    tags.add_xmp(xmp_metadata)
    
    # Step 3: Retrieve XMP from MetadataTags
    retrieved_xmp = tags.get_xmp()
    assert retrieved_xmp is not None
    assert isinstance(retrieved_xmp, muimg.XmpMetadata)
    
    # Step 4: Convert back to dict
    result_dict = muimg.supported_xmp_to_dict(retrieved_xmp)
    
    # Step 5: Validate values match
    # Check supported properties
    assert result_dict['Temperature'] == str(original_dict['Temperature'])
    assert result_dict['Tint'] == str(original_dict['Tint'])
    assert result_dict['Exposure2012'] == str(original_dict['Exposure2012'])
    
    # Check tone curve (should be list of tuples)
    assert 'ToneCurvePV2012' in result_dict
    tone_curve = result_dict['ToneCurvePV2012']
    assert isinstance(tone_curve, list)
    assert len(tone_curve) == len(original_dict['ToneCurvePV2012'])
    for i, (orig_point, result_point) in enumerate(zip(original_dict['ToneCurvePV2012'], tone_curve)):
        assert isinstance(result_point, tuple)
        assert len(result_point) == 2
        # Allow small floating point differences
        assert abs(result_point[0] - orig_point[0]) < 0.01
        assert abs(result_point[1] - orig_point[1]) < 0.01
    
    # Check dc:subject (should be list of strings)
    assert 'dc:subject' in result_dict
    assert isinstance(result_dict['dc:subject'], list)
    assert result_dict['dc:subject'] == original_dict['dc:subject']
    
    # Check tiff:Orientation
    assert result_dict['tiff:Orientation'] == original_dict['tiff:Orientation']
    
    # Note: Unsupported properties (Contrast2012, etc.) won't be in result_dict
    # because supported_xmp_to_dict only extracts the supported subset.
    # This is expected behavior - the full XMP is preserved in the XMP tag,
    # but the convenience dict conversion only includes supported properties.
    
    print(f"Original dict keys: {sorted(original_dict.keys())}")
    print(f"Result dict keys: {sorted(result_dict.keys())}")
    print("XMP round-trip test passed!")


def test_xmp_roundtrip_dng_file():
    """Test round-tripping XMP values through DNG file write/read.
    
    Creates a small color ramp image, writes it to a DNG file with XMP metadata,
    then reads it back and validates the XMP values match.
    """
    import numpy as np
    
    # Create a small 64x64 color ramp image (linear raw RGB, uint16)
    height, width = 64, 64
    
    # Create color ramps with different patterns:
    # Red: left to right (horizontal gradient)
    # Blue: up to down (vertical gradient)
    # Green: diagonal (top-left to bottom-right)
    linear_rgb = np.zeros((height, width, 3), dtype=np.uint16)
    
    # Red channel: horizontal gradient (left to right)
    for x in range(width):
        linear_rgb[:, x, 0] = int(65535 * x / (width - 1))
    
    # Blue channel: vertical gradient (top to bottom)
    for y in range(height):
        linear_rgb[y, :, 2] = int(65535 * y / (height - 1))
    
    # Green channel: diagonal gradient (top-left to bottom-right)
    for y in range(height):
        for x in range(width):
            diagonal_position = (x + y) / (width + height - 2)
            linear_rgb[y, x, 1] = int(65535 * diagonal_position)
    
    # Create XMP metadata with reasonable supported values
    # Use a more dramatic S-curve
    xmp_dict = {
        'Temperature': 5800,
        'Tint': 10,
        'Exposure2012': -0.5,
        'ToneCurvePV2012': [(0.0, 0.0), (0.25, 0.1), (0.5, 0.5), (0.75, 0.9), (1.0, 1.0)],
        'dc:subject': ['test_image', 'color_ramp', 'xmp_roundtrip'],
        'tiff:Orientation': '1',
    }
    
    # Create MetadataTags and add XMP
    tags = muimg.MetadataTags()
    muimg.add_supported_xmp_from_dict(tags, xmp_dict)
    
    # Write DNG to test_outputs directory
    output_dir = Path(__file__).parent / "test_outputs" / "test_xmp_roundtrip"
    output_dir.mkdir(parents=True, exist_ok=True)
    dng_path = output_dir / "test_xmp_roundtrip.dng"
    muimg.write_dng_from_array(
        destination_file=str(dng_path),
        data=linear_rgb,
        ifd0_tags=tags,
        photometric="linear_raw",
        bits_per_pixel=16,
    )
    
    # Verify file was created
    assert dng_path.exists()
    print(f"Created DNG file: {dng_path} ({dng_path.stat().st_size} bytes)")
    
    # Read back the DNG file
    with muimg.DngFile(dng_path) as dng:
        # Get XMP metadata directly from DngFile
        retrieved_xmp = dng.get_xmp()
        assert retrieved_xmp is not None
        assert isinstance(retrieved_xmp, muimg.XmpMetadata)
        
        # Convert to dict for comparison
        result_dict = muimg.supported_xmp_to_dict(retrieved_xmp)
        
        # Validate values match (accounting for type conversions)
        assert result_dict['Temperature'] == str(xmp_dict['Temperature'])
        assert result_dict['Tint'] == str(xmp_dict['Tint'])
        assert result_dict['Exposure2012'] == str(xmp_dict['Exposure2012'])
        
        # Check tone curve
        assert 'ToneCurvePV2012' in result_dict
        tone_curve = result_dict['ToneCurvePV2012']
        assert isinstance(tone_curve, list)
        assert len(tone_curve) == len(xmp_dict['ToneCurvePV2012'])
        for orig_point, result_point in zip(xmp_dict['ToneCurvePV2012'], tone_curve):
            assert isinstance(result_point, tuple)
            assert len(result_point) == 2
            # Allow small floating point differences from serialization
            assert abs(result_point[0] - orig_point[0]) < 0.01
            assert abs(result_point[1] - orig_point[1]) < 0.01
        
        # Check dc:subject
        assert 'dc:subject' in result_dict
        assert isinstance(result_dict['dc:subject'], list)
        assert result_dict['dc:subject'] == xmp_dict['dc:subject']
        
        # Check tiff:Orientation
        assert result_dict['tiff:Orientation'] == xmp_dict['tiff:Orientation']
        
        print(f"Successfully round-tripped XMP through DNG file!")
        print(f"Original XMP keys: {sorted(xmp_dict.keys())}")
        print(f"Retrieved XMP keys: {sorted(result_dict.keys())}")
        print(f"DNG file available for validation: {dng_path}")
