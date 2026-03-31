import io
import logging
from pathlib import Path
from typing import IO, Union

import numpy as np

from .dngio import DngFile, DngPage, decode_dng

logger = logging.getLogger(__name__)

def _coerce_decoded_image(img: np.ndarray, output_dtype: type) -> np.ndarray:
    import cv2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported decoded image shape: {img.shape}")

    if output_dtype is np.uint8:
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8, copy=False)
        elif img.dtype != np.uint8:
            raise ValueError(
                f"Unsupported decoded dtype {img.dtype} for output_dtype=np.uint8"
            )
        return img

    if output_dtype is np.uint16:
        if img.dtype == np.uint8:
            return (img.astype(np.uint16) << 8)
        if img.dtype != np.uint16:
            raise ValueError(
                f"Unsupported decoded dtype {img.dtype} for output_dtype=np.uint16"
            )
        return img

    raise ValueError(f"Unsupported output_dtype: {output_dtype}")


def decode_image(
    file: Union[str, Path, IO[bytes]],
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
        is_dng = "DNGVersion" in dng_file.pages[0].tags
    except Exception:
        pass
    
    if is_dng:
        # Use default decode_dng parameters
        # For advanced control with custom parameters, use 'muimg dng convert' CLI command
        return decode_dng(file=dng_file, output_dtype=output_dtype)
    
    # Fall back to cv2 for other formats (processing_params ignored)
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

    return _coerce_decoded_image(img, output_dtype=output_dtype)


def _save_decoded_image(
    image: np.ndarray,
    output: Union[str, Path, IO[bytes]],
    output_format: str = "jpg",
) -> bool:
    """
    Save a decoded RGB image to file or stream.
    
    Helper function used by convert_imgformat and convert_dng.
    
    Args:
        image: RGB image array with shape (height, width, 3)
        output: Output file path (str/Path) or stream (IO[bytes])
        output_format: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert RGB to BGR for OpenCV
    import cv2
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if isinstance(output, (str, Path)):
        # Save to file
        output_path = Path(output)
        success = cv2.imwrite(str(output_path), bgr_image)
        
        if success:
            logger.info(f"Successfully saved image to {output_path}")
            return True
        else:
            logger.error(f"Failed to save file: {output_path}")
            return False
    else:
        # Write to stream
        format_ext = f".{output_format.lstrip('.')}"
        success, encoded_buffer = cv2.imencode(format_ext, bgr_image)
        
        if success:
            encoded_bytes = encoded_buffer.tobytes()
            output.write(encoded_bytes)
            logger.info(f"Successfully wrote {len(encoded_bytes)} bytes to stream as {output_format}")
            return True
        else:
            logger.error(f"Failed to encode image as {output_format}")
            return False


def convert_imgformat(
    file: Union[str, Path, IO[bytes]],
    output: Union[str, Path, IO[bytes]],
    output_dtype: type = np.uint8,
    output_format: str = "jpg",
) -> bool:
    """
    Convert an image file to another format, saving to file or stream.
    
    For DNG files, uses default rendering parameters. For custom DNG rendering,
    use the 'muimg dng convert' CLI command instead.
    
    Args:
        file: Path to image file or file-like object
        output: Output file path (str/Path) or stream (IO[bytes])
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        output_format: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Decode image to numpy array
        image = decode_image(file=file, output_dtype=output_dtype)
        
        # Save using shared helper
        return _save_decoded_image(image, output, output_format)
                
    except Exception as e:
        logger.error(f"Error converting {file}: {e}", exc_info=True)
        raise


def convert_imgformat_to_stream(
    file: Union[str, Path, IO[bytes]],
    output_format: str = "jpg",
    output_dtype: type = np.uint8,
) -> bytes:
    """
    Encode an image file to bytes.
    
    Deprecated: Use convert_imgformat(file, output=BytesIO(), output_format=...) instead.
    
    Args:
        file: Path to image file or file-like object
        output_format: Output format ("jpg", "png", "tiff", etc.)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        
    Returns:
        bytes: Encoded image data
    """
    from io import BytesIO
    stream = BytesIO()
    convert_imgformat(file=file, output=stream, output_dtype=output_dtype, output_format=output_format)
    return stream.getvalue()


def convert_dng(
    file: Union[str, Path, IO[bytes], "DngFile", "DngPage"],
    output: Union[str, Path, IO[bytes]],
    output_dtype: type = np.uint16,
    demosaic_algorithm: str = "OPENCV_EA",
    strict: bool = True,
    use_xmp: bool = True,
    rendering_params: dict = None,
    use_coreimage_if_available: bool = False,
    output_format: str = "jpg",
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
        output_format: Output format for stream output ("jpg", "png", "tiff", etc.)
            Ignored when output is a file path (format determined by extension)
        
    Returns:
        bool: True if successful, False otherwise
        
    Note:
        When passing a DngPage for a preview page (RGB/YCBCR), rendering_params
        are not supported and will cause the function to return False.
        Core Image is not available when passing a DngPage instance.
    """
    try:
        # Decode DNG to numpy array
        image = decode_dng(
            file=file,
            output_dtype=output_dtype,
            demosaic_algorithm=demosaic_algorithm,
            use_coreimage_if_available=use_coreimage_if_available,
            use_xmp=use_xmp,
            rendering_params=rendering_params,
            strict=strict,
        )
        
        # Save using shared helper
        return _save_decoded_image(image, output, output_format)
                
    except ValueError as e:
        # Handle validation errors (e.g., rendering params on preview pages)
        logger.error(f"Validation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error converting DNG: {e}", exc_info=True)
        raise


def convert_dng_to_stream(
    file: Union[str, Path, IO[bytes], "DngFile", "DngPage"],
    output_format: str = "jpg",
    output_dtype: type = np.uint16,
    demosaic_algorithm: str = "OPENCV_EA",
    strict: bool = True,
    use_xmp: bool = True,
    rendering_params: dict = None,
    use_coreimage_if_available: bool = False,
) -> bytes:
    """
    Encode a DNG file to bytes with custom rendering parameters.
    
    Deprecated: Use convert_dng(file, output=BytesIO(), output_format=...) instead.
    
    Args:
        file: DNG file path, file-like object, DngFile instance, or DngPage instance
        output_format: Output format ("jpg", "png", "tiff", etc.)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        demosaic_algorithm: Demosaic algorithm for CFA pages ("RCD", "VNG", etc.)
        strict: If True, raise error on unsupported DNG tags
        use_xmp: Whether to read XMP metadata for rendering defaults
        rendering_params: Optional dict to override rendering parameters
        use_coreimage_if_available: Use Core Image pipeline on macOS if available
        
    Returns:
        bytes: Encoded image data
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
        output_format=output_format,
    )
    return stream.getvalue()