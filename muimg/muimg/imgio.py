# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import IO

import numpy as np
import tifffile

from .dngio import DngFile, DngPage, decode_dng
from .raw_render import DemosaicAlgorithm, convert_dtype
from .tiff_metadata import MetadataTags, filter_tags_by_ifd_category, resolve_tag, TiffType

logger = logging.getLogger(__name__)

def write_image(
    image: np.ndarray,
    output: str | Path | IO[bytes],
    output_format_stream: str = "jpg",
    metadata: MetadataTags | None = None,
) -> bool:
    """
    Save a decoded RGB image to file or stream with optional metadata.
    
    Helper function used by convert_imgformat and convert_dng.
    
    Args:
        image: RGB image array with shape (height, width, 3)
        output: Output file path (str/Path) or stream (IO[bytes])
        output_format_stream: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
        metadata: Optional metadata tags to embed in output (for JPG/TIFF)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine output format
    if isinstance(output, (str, Path)):
        output_path = Path(output)
        format_ext = output_path.suffix.lower()
    else:
        format_ext = f".{output_format_stream.lstrip('.')}"
    
    # Handle TIFF using tifffile
    if format_ext in ('.tif', '.tiff'):
        
        # Prepare metadata if provided
        extratags = None
        software = "muimg"  # Default
        if metadata is not None:
            # Extract Software tag before filtering (if present, use it; otherwise use default)
            software = metadata.get_tag('Software') or "muimg"
            
            # Filter to only ifd0 and exif tags
            filtered_metadata = filter_tags_by_ifd_category(metadata, ["ifd0", "exif"])
            
            # Remove SubIFD pointer tags and others that tifffile manages itself
            for tag_name in ("ExifTag", "GPSTag", "ProfileIFD", "Software", "ImageDescription"):
                filtered_metadata.remove_tag(tag_name)
            
            extratags = filtered_metadata
        
        # Write TIFF directly to output
        tifffile.imwrite(output, image, extratags=extratags, software=software)
        msg = "with metadata" if extratags is not None else "without metadata"
        if isinstance(output, (str, Path)):
            logger.info(f"Successfully saved image {msg} to {output}")
        else:
            logger.info(f"Successfully wrote image {msg} to stream")
        return True
    
    # Handle JXL format
    elif format_ext in ('.jxl',):
        import imagecodecs
        
        # Encode to JXL with lossless compression
        jxl_data = imagecodecs.jpegxl_encode(
            image,
            lossless=True,
            effort=5,
        )
        
        # Write to output
        if isinstance(output, (str, Path)):
            with open(output_path, 'wb') as f:
                f.write(jxl_data)
            logger.info(f"Successfully saved image to {output_path} (metadata not saved)")
        else:
            output.write(jxl_data)
            logger.info(f"Successfully wrote {len(jxl_data)} bytes to stream (metadata not saved)")
        return True
    
    # Handle other formats with OpenCV
    else:
        # Convert RGB to BGR for OpenCV
        import cv2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Encode with OpenCV
        success, encoded_buffer = cv2.imencode(format_ext, bgr_image)
        if not success:
            logger.error(f"Failed to encode image as {format_ext}")
            return False
        image_bytes = encoded_buffer.tobytes()
    
        # Write to output
        if isinstance(output, (str, Path)):
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"Successfully saved image to {output_path} (metadata not saved)")
        else:
            output.write(image_bytes)
            logger.info(f"Successfully wrote {len(image_bytes)} bytes to stream (metadata not saved)")
    
    return True


def decode_image(
    file: str | Path | IO[bytes],
    output_dtype: type = np.uint8,
) -> np.ndarray:
    """
    Decode an image file to a numpy array.
    
    Supports DNG files (with default raw processing) and standard image formats
    (JPEG, PNG, TIFF, etc.) via OpenCV.
    
    Args:
        file: Path to image file or file-like object
        output_dtype: Output data type (np.uint8 or np.uint16)
        **processing_params: Ignored. Kept for API compatibility.
            For DNG files with custom rendering parameters, use the 
            'muimg dng convert' CLI command instead.
        
    Returns:
        RGB image array with shape (height, width, 3) and specified dtype
    """
    # Try to open as DNG - DngFile handles str, Path, and IO[bytes]
    dng_file = None
    is_dng = False
    try:
        dng_file = DngFile(file)
        # Check if file has DNG version tag in IFD0
        is_dng = "DNGVersion" in dng_file.get_ifd0_tags()
    except Exception:
        pass
    
    if is_dng:
        # For advanced control with custom parameters, use 'muimg dng convert' CLI command
        return dng_file.get_main_page().decode_to_rgb(output_dtype)
    
    # Check if it's a JXL file
    if isinstance(file, (str, Path)) and str(file).lower().endswith('.jxl'):
        # Decode JXL using imagecodecs
        import imagecodecs
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as f:
                jxl_data = f.read()
        else:
            file.seek(0)
            jxl_data = file.read()
        
        img = imagecodecs.jpegxl_decode(jxl_data)
        if img is None:
            raise ValueError("Failed to decode JXL image")
        
        # JXL already returns RGB, just convert dtype if needed
        return convert_dtype(img, output_dtype)
    
    # Check if it's a TIFF file - use tifffile to avoid OpenCV warnings about EXIF tags
    if isinstance(file, (str, Path)) and str(file).lower().endswith(('.tif', '.tiff')):
        import tifffile
        img = tifffile.imread(file)
        if img is None:
            raise ValueError("Failed to decode TIFF image")
        
        # tifffile returns RGB, just convert dtype if needed
        return convert_dtype(img, output_dtype)
    
    # Fall back to cv2 for other formats
    import cv2
    if isinstance(file, (str, Path)):
        img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    else:
        # File-like input - read and decode
        file.seek(0)
        data = file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # OpenCV returns BGR, convert to RGB and handle grayscale/alpha
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported decoded image shape: {img.shape}")

    return convert_dtype(img, output_dtype)

def convert_imgformat(
    file: str | Path | IO[bytes],
    output: str | Path | IO[bytes],
    output_dtype: type = np.uint8,
    output_format_stream: str = "jpg",
) -> bool:
    """
    Convert an image file to another format, saving to file or stream.
    
    For DNG files, uses default rendering parameters. For custom DNG rendering,
    use the 'muimg dng convert' CLI command instead.
    
    Args:
        file: Path to image file or file-like object
        output: Output file path (str/Path) or stream (IO[bytes])
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        output_format_stream: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Decode image to numpy array
        image = decode_image(file=file, output_dtype=output_dtype)
        
        # Save using shared helper
        return write_image(image, output, output_format_stream)
                
    except Exception as e:
        logger.error(f"Error converting {file} ({type(e).__name__}): {e}", exc_info=True)
        raise


def convert_imgformat_to_stream(
    file: str | Path | IO[bytes],
    output_format_stream: str = "jpg",
    output_dtype: type = np.uint8,
) -> IO[bytes]:
    """
    Encode an image file to a BytesIO stream.
    
    Helper function that creates a BytesIO stream and calls convert_imgformat.
    
    Args:
        file: Path to image file or file-like object
        output_format_stream: Output format ("jpg", "png", "tiff", etc.)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        
    Returns:
        BytesIO: Stream containing encoded image data
    """
    from io import BytesIO
    stream = BytesIO()
    convert_imgformat(
        file=file, output=stream, output_dtype=output_dtype, output_format_stream=output_format_stream)
    stream.seek(0)
    return stream


def convert_dng(
    file: str | Path | IO[bytes] | "DngFile" | "DngPage",
    output: str | Path | IO[bytes],
    output_dtype: type = np.uint16,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    strict: bool = True,
    use_xmp: bool = True,
    rendering_params: dict = None,
    use_coreimage_if_available: bool = False,
    output_format_stream: str = "jpg",
) -> bool:
    """
    Convert a DNG file or page to an image file or stream with custom rendering parameters.
    
    This function provides programmatic access to DNG conversion with full control
    over rendering parameters. For simple default conversion, use convert_imgformat().
    
    Args:
        file: DNG file path, file-like object, DngFile instance, or DngPage instance
        output: Output file path (str/Path) or stream (IO[bytes])
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        demosaic_algorithm: Demosaic algorithm for CFA pages ("RCD", "VNG", etc.)
        strict: If True, raise error on unsupported DNG tags
        use_xmp: Whether to read XMP metadata for rendering defaults
        rendering_params: Optional dict to override rendering parameters. Supported keys:
            - 'Temperature': White balance temperature in Kelvin (float)
            - 'Tint': White balance tint adjustment (float)
            - 'Exposure2012': Exposure compensation in stops (float)
            - 'ToneCurvePV2012': Main tone curve
            - 'orientation': EXIF orientation code
            See decode_dng() for full list of supported parameters.
        use_coreimage_if_available: Use Core Image pipeline on macOS if available.
            Note: Only supported when passing file path/IO/DngFile, not DngPage.
        output_format_stream: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
        
    Returns:
        bool: True if successful, False otherwise
        
    Note:
        When passing a DngPage for a preview page (RGB/YCBCR), rendering_params
        are not supported and will cause the function to return False.
        Core Image is not available when passing a DngPage instance.
    """
    try:
        # Decode DNG to numpy array with metadata
        image, metadata = decode_dng(
            file=file,
            output_dtype=output_dtype,
            demosaic_algorithm=demosaic_algorithm,
            use_coreimage_if_available=use_coreimage_if_available,
            use_xmp=use_xmp,
            rendering_params=rendering_params,
            strict=strict,
        )
        
        # Save using shared helper with metadata
        return write_image(image, output, output_format_stream, metadata=metadata)
                
    except ValueError as e:
        # Handle validation errors (e.g., rendering params on preview pages)
        logger.error(f"Validation error: ({type(e).__name__}): {e}")
        return False
    except Exception as e:
        logger.error(f"Error converting DNG ({type(e).__name__}): {e}", exc_info=True)
        raise


def convert_dng_to_stream(
    file: str | Path | IO[bytes] | "DngFile" | "DngPage",
    output_format_stream: str = "jpg",
    output_dtype: type = np.uint16,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    strict: bool = True,
    use_xmp: bool = True,
    rendering_params: dict = None,
    use_coreimage_if_available: bool = False,
) -> IO[bytes]:
    """
    Encode a DNG file to a BytesIO stream with custom rendering parameters.
    
    Helper function that creates a BytesIO stream and calls convert_dng.
    
    Args:
        file: DNG file path, file-like object, DngFile instance, or DngPage instance
        output_format_stream: Output format ("jpg", "png", "tiff", etc.)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        demosaic_algorithm: Demosaic algorithm for CFA pages ("RCD", "VNG", etc.)
        strict: If True, raise error on unsupported DNG tags
        use_xmp: Whether to read XMP metadata for rendering defaults
        rendering_params: Optional dict to override rendering parameters
        use_coreimage_if_available: Use Core Image pipeline on macOS if available
        
    Returns:
        BytesIO: Stream containing encoded image data
    """
    from io import BytesIO
    stream = BytesIO()
    convert_dng(
        file=file,
        output=stream,
        output_dtype=output_dtype,
        demosaic_algorithm=demosaic_algorithm,
        strict=strict,
        use_xmp=use_xmp,
        rendering_params=rendering_params,
        use_coreimage_if_available=use_coreimage_if_available,
        output_format_stream=output_format_stream,
    )
    stream.seek(0)
    return stream