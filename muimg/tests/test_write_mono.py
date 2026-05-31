# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Test writing gray/monochrome DNG images.

Validates that single-channel (monochrome) DNGs can be written and validated.
"""

import logging
import numpy as np
import pytest
from pathlib import Path
from tifffile import COMPRESSION

# Suppress tifffile warnings about "shaped series shape does not match page shape"
logging.getLogger('tifffile').setLevel(logging.CRITICAL)

from muimg.dngio import (
    write_dng_from_array,
    DngFile,
    IfdDataSpec,
    PageEncoding,
    PreviewParams,
    PyramidParams,
)
from muimg.tiff_metadata import MetadataTags
from conftest import run_dng_validate, OutputPathManager, compute_diff_stats

# Test output path manager - set persistent=True to keep outputs, False for tmp_path
output_path_manager = OutputPathManager(persistent=True)


# Test configurations: (label, dtype, compression, compression_args, mean_thresh, max_thresh, tile_size, preview, pyramid)
MONO_CONFIGS = [
    # Basic compression tests
    ("uint16_uncompressed", np.uint16, COMPRESSION.NONE, None, 0.0, 0.0, None, False, False),
    ("uint16_lossless_jxl", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, None, False, False),
    ("uint8_uncompressed", np.uint8, COMPRESSION.NONE, None, 0.0, 0.0, None, False, False),
    ("uint8_lossless_jpeg", np.uint8, COMPRESSION.JPEG, {'lossless': True}, 0.0, 0.0, None, False, False),
    # Lossy JXL
    ("uint16_lossy_jxl_1.0", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 1.0, 'effort': 4}, 0.95, 15.0, None, False, False),
    # Tile sizes (must be multiples of 16 for compression alignment)
    ("uint16_tiles_256x256", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, (256, 256), False, False),
    ("uint16_tiles_192x128", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, (192, 128), False, False),
    ("uint16_tiles_320x416", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, (320, 416), False, False),
    # Preview
    ("uint16_with_preview", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, None, True, False),
    # Pyramid only (no preview)
    ("uint16_pyramid_only", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, None, False, True),
    # Preview + Pyramid
    ("uint16_preview_pyramid", np.uint16, COMPRESSION.JPEGXL_DNG, {'distance': 0.0, 'effort': 4}, 0.0, 0.0, (128, 144), True, True),
]


def _test_write_mono(
    tmp_path,
    label: str,
    dtype,
    compression,
    compression_args,
    mean_thresh: float,
    max_thresh: float,
    tile_size: tuple[int, int] | None,
    use_preview: bool,
    use_pyramid: bool
):
    """Test writing monochrome DNG with specified configuration.
    
    Args:
        tmp_path: pytest tmp_path fixture
        label: Test case label for output filenames
        dtype: numpy dtype for test data
        compression: tifffile COMPRESSION enum
        compression_args: dict of compression arguments or None
        mean_thresh: Maximum allowed mean diff percentage
        max_thresh: Maximum allowed max diff percentage
    """
    output_path = output_path_manager.get_path(tmp_path, "test_write_mono")
    
    # Create grayscale ramp
    width, height = 640, 480
    gray_ramp = _generate_gray_ramp(width, height, dtype=dtype)
    
    # Write as monochrome LINEAR_RAW
    dng_path = output_path / f"mono_{label}.dng"
    
    # Add required metadata
    metadata = MetadataTags()
    metadata.add_tag("UniqueCameraModel", "Test Monochrome Camera")
    
    # Create main encoding with optional tile size
    main_encoding = PageEncoding(
        compression=compression,
        compression_args=compression_args,
        tile_size=tile_size,
    ) if compression != COMPRESSION.NONE else None
    
    # Build preview and pyramid params
    preview_params = None
    pyramid_params = None
    
    if use_preview:
        preview_params = PreviewParams()
    
    if use_pyramid:
        pyramid_encoding = PageEncoding(
            compression=compression,
            compression_args=compression_args,
            tile_size=tile_size,
        ) if compression != COMPRESSION.NONE else None
        pyramid_params = PyramidParams(levels=2, encoding=pyramid_encoding)
    
    # Use write_dng_from_array with optional preview and pyramid
    data_spec = IfdDataSpec(
        data=gray_ramp,
        photometric="LINEAR_RAW",
        encoding=main_encoding,
        extratags=metadata,
    )
    write_dng_from_array(
        destination_file=dng_path,
        data_spec=data_spec,
        preview=preview_params,
        pyramid=pyramid_params,
        num_compression_workers=4,
    )
    
    # Validate with dng_validate
    output_base = output_path / f"mono_{label}_validate"
    run_dng_validate(
        dng_path=dng_path,
        output_base=output_base,
        validate=True,
        indent="  ",
    )
    
    # Read back and verify
    with DngFile(dng_path) as dng:
        main_page = dng.get_main_page()
        
        # Check photometric interpretation
        assert main_page.photometric_name == "LINEAR_RAW"
        assert main_page.samplesperpixel == 1
        
        # Read back the raw data using our API
        raw_data = dng.get_linear_raw()
        assert raw_data is not None, f"Failed to get LINEAR_RAW from {label}"
        
        # Verify shape - should be 2D for monochrome (H, W) or 3D (H, W, 1)
        if raw_data.ndim == 3:
            assert raw_data.shape[2] == 1
            raw_data = raw_data.squeeze()  # Convert to 2D for comparison
        
        # Verify data matches original
        if mean_thresh == 0.0 and max_thresh == 0.0:
            # Exact match expected for lossless/uncompressed
            assert np.array_equal(raw_data, gray_ramp)
        else:
            # Lossy compression - check thresholds
            stats = compute_diff_stats(raw_data, gray_ramp)
            print(f"  {label}: mean={stats['mean']:.4f}%, max={stats['max']:.4f}%")
            assert stats['mean'] < mean_thresh, (
                f"{label}: Mean diff {stats['mean']:.4f}% exceeds {mean_thresh}%"
            )
            assert stats['max'] < max_thresh, (
                f"{label}: Max diff {stats['max']:.4f}% exceeds {max_thresh}%"
            )
        
        # Render and compare against dng_validate output
        # Render our output
        our_rendered = dng.render_raw(output_dtype=np.uint8)
        assert our_rendered is not None, f"Failed to render {label}"
        
        # Compare against dng_validate rendering (if available)
        output_tiff = Path(str(output_base) + '.tif')
        if output_tiff.exists():
            from tifffile import imread
            validated_render = imread(output_tiff)
            # dng_validate outputs 3-channel TIFF for monochrome, we output 1-channel
            if validated_render.ndim == 3 and our_rendered.ndim == 2:
                validated_render = validated_render[:, :, 0]  # Take first channel
            render_stats = compute_diff_stats(our_rendered, validated_render)
            print(f"    Render vs dng_validate: mean={render_stats['mean']:.4f}%, max={render_stats['max']:.4f}%")
            assert render_stats['mean'] < 0.6, (
                f"{label}: Render vs dng_validate diff {render_stats['mean']:.4f}% exceeds 0.6%"
            )
    
    print(f"  ✓ Monochrome {label} DNG validated successfully")


def _generate_gray_ramp(width: int, height: int, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Generate a grayscale ramp test image.
    
    Creates a diagonal gradient from black (top-left) to white (bottom-right).
    """
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    # Diagonal gradient from 0.0 (top-left) to 1.0 (bottom-right)
    gray = (xx / width + yy / height) / 2.0
    
    # Convert to target dtype
    if dtype == np.uint8:
        return (gray * 255).astype(np.uint8)
    elif dtype == np.uint16:
        return (gray * 65535).astype(np.uint16)
    else:
        return gray.astype(dtype)


@pytest.mark.parametrize("label,dtype,compression,compression_args,mean_thresh,max_thresh,tile_size,use_preview,use_pyramid",
                         MONO_CONFIGS,
                         ids=[c[0] for c in MONO_CONFIGS])
def test_write_mono(tmp_path, label, dtype, compression, compression_args, mean_thresh, max_thresh, tile_size, use_preview, use_pyramid):
    """Test writing monochrome DNG with various configurations."""
    _test_write_mono(tmp_path, label, dtype, compression, compression_args, mean_thresh, max_thresh, tile_size, use_preview, use_pyramid)


def test_write_mono_3d_shape(tmp_path):
    """Test writing monochrome DNG with 3D array shape (H, W, 1)."""
    output_path = output_path_manager.get_path(tmp_path, "test_write_mono")
    
    # Create 16-bit grayscale ramp as 3D array (H, W, 1)
    width, height = 640, 480
    gray_ramp_2d = _generate_gray_ramp(width, height, dtype=np.uint16)
    gray_ramp_3d = gray_ramp_2d.reshape(height, width, 1)
    
    dng_path = output_path / "mono_3d_shape.dng"
    
    metadata = MetadataTags()
    metadata.add_tag("UniqueCameraModel", "Test Monochrome Camera")
    
    data_spec = IfdDataSpec(
        data=gray_ramp_3d,
        photometric="LINEAR_RAW",
        encoding=None,
        extratags=metadata,
    )
    
    write_dng_from_array(destination_file=dng_path, data_spec=data_spec)
    
    output_base = output_path / "mono_3d_shape_validate"
    run_dng_validate(dng_path=dng_path, output_base=output_base, validate=True, indent="  ")
    
    with DngFile(dng_path) as dng:
        main_page = dng.get_main_page()
        raw_data = main_page.asarray()
        
        # Data should come back as 2D
        assert raw_data.ndim == 2
        assert raw_data.shape == (height, width)
        assert np.array_equal(raw_data, gray_ramp_2d)
    
    print(f"  ✓ Monochrome 3D shape DNG validated successfully")


def test_write_mono_with_metadata(tmp_path):
    """Test writing monochrome DNG with metadata tags."""
    output_path = output_path_manager.get_path(tmp_path, "test_write_mono")
    
    width, height = 640, 480
    gray_ramp = _generate_gray_ramp(width, height, dtype=np.uint16)
    
    metadata = MetadataTags()
    metadata.add_tag("UniqueCameraModel", "Test Monochrome Camera")
    
    dng_path = output_path / "mono_with_metadata.dng"
    
    data_spec = IfdDataSpec(
        data=gray_ramp,
        photometric="LINEAR_RAW",
        encoding=None,
        extratags=metadata,
    )
    
    write_dng_from_array(destination_file=dng_path, data_spec=data_spec)
    
    output_base = output_path / "mono_with_metadata_validate"
    run_dng_validate(dng_path=dng_path, output_base=output_base, validate=True, indent="  ")
    
    with DngFile(dng_path) as dng:
        main_page = dng.get_main_page()
        
        camera_model = main_page.tags.get('UniqueCameraModel')
        assert camera_model is not None
        assert camera_model.value == "Test Monochrome Camera"
        
        raw_data = main_page.asarray()
        assert np.array_equal(raw_data, gray_ramp)
    
    print(f"  ✓ Monochrome DNG with metadata validated successfully")
