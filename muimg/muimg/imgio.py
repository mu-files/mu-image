import io
import logging
from pathlib import Path
from typing import IO, Union

import numpy as np

from .dngio import DngFile, decode_dng

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
    **processing_params,
) -> np.ndarray:
    """
    Decode an image file to a numpy array.
    
    Supports DNG files (with full raw processing) and standard image formats
    (JPEG, PNG, TIFF, etc.) via OpenCV.
    
    Args:
        file: Path to image file or file-like object
        output_dtype: Output data type (np.uint8 or np.uint16)
        **processing_params: Processing parameters for DNG files including:
            - use_xmp: Whether to read XMP metadata for default values (default: True)
            - use_coreimage_if_available: Use Core Image pipeline on macOS (default: False)
            - temperature, tint, exposure, tone_curve, noise_reduction, orientation, etc.
        
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
        use_xmp = processing_params.pop('use_xmp', True)
        use_coreimage_if_available = processing_params.pop('use_coreimage_if_available', False)
        demosaic_algorithm = processing_params.pop('demosaic_algorithm', 'RCD')
        
        # Remaining params go into rendering_params dict
        rendering_params = processing_params if processing_params else None
        
        return decode_dng(
            file=dng_file,
            output_dtype=output_dtype,
            use_coreimage_if_available=use_coreimage_if_available,
            use_xmp=use_xmp,
            demosaic_algorithm=demosaic_algorithm,
            rendering_params=rendering_params,
        )
    
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

    return _coerce_decoded_image(img, output_dtype=output_dtype)


def convert_imgformat(
    file: Union[str, Path, IO[bytes]],
    output_path: Union[str, Path],
    output_dtype: type = np.uint8,
    **processing_params
) -> bool:
    """
    Convert an image file to another format.
    
    Args:
        file: Path to image file or file-like object
        output_path: Output file path (format determined by extension)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        **processing_params: Processing parameters for DNG files including:
            - use_xmp: Whether to read XMP metadata for default values (default: True)
            - temperature, tint, exposure, tone_curve, noise_reduction, orientation, etc.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Decode image to numpy array
        image = decode_image(
            file=file,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        import cv2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save to file
        output_path = Path(output_path)
        success = cv2.imwrite(str(output_path), bgr_image)
        
        if success:
            logger.info(f"Successfully converted {file} to {output_path}")
            return True
        else:
            logger.error(f"Failed to save file: {output_path}")
            return False
                
    except Exception as e:
        logger.error(f"Error converting {file}: {e}", exc_info=True)
        raise


def convert_imgformat_to_stream(
    file: Union[str, Path, IO[bytes]],
    output_format: str = "jpg",
    output_dtype: type = np.uint8,
    **processing_params
) -> bytes:
    """
    Encode an image file to bytes.
    
    Args:
        file: Path to image file or file-like object
        output_format: Output format ("jpg", "png", "tiff", etc.)
        output_dtype: Output data type (np.uint8 for 8-bit, np.uint16 for 16-bit)
        **processing_params: Processing parameters for DNG files including:
            - use_xmp: Whether to read XMP metadata for default values (default: True)
            - temperature, tint, exposure, tone_curve, noise_reduction, orientation, etc.
        
    Returns:
        bytes: Encoded image data
    """
    try:
        # Decode image to numpy array
        image = decode_image(
            file=file,
            output_dtype=output_dtype,
            **processing_params
        )
        
        # Convert RGB to BGR for OpenCV
        import cv2
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure format has leading dot
        format_ext = f".{output_format.lstrip('.')}"
        
        # Encode image
        success, encoded_buffer = cv2.imencode(format_ext, bgr_image)
        
        if success:
            encoded_bytes = encoded_buffer.tobytes()
            logger.info(f"Encoded {file} to {len(encoded_bytes)} bytes as {output_format}")
            return encoded_bytes
        else:
            logger.error(f"Failed to encode image as {output_format}")
            raise RuntimeError(f"Failed to encode image as {output_format}")
                
    except Exception as e:
        logger.error(f"Error encoding {file} to stream: {e}", exc_info=True)
        raise
