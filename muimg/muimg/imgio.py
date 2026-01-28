import io
from pathlib import Path
from typing import IO, Union

import cv2
import numpy as np

from .dngio import DngFile, decode_raw

def _coerce_decoded_image(img: np.ndarray, output_dtype: type) -> np.ndarray:
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
    use_xmp: bool = True,
    output_dtype: type = np.uint8,
    **processing_params,
) -> np.ndarray:
    if isinstance(file, (str, Path)):
        path = Path(file)
        if path.suffix.lower() == ".dng":
            return decode_raw(
                file=path,
                use_xmp=use_xmp,
                output_dtype=output_dtype,
                **processing_params,
            )
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image")

        return _coerce_decoded_image(img, output_dtype=output_dtype)

    # File-like input
    seekable = hasattr(file, "seek")
    if seekable:
        try:
            file.seek(0)
        except Exception:
            seekable = False

    if not seekable:
        raise ValueError(
            "file-like objects passed to decode_image must be seekable (support seek(0))."
        )

    dng_file = None
    try:
        dng_file = DngFile(file)
        page0 = dng_file.pages[0]
        is_dng = "DNGVersion" in page0.tags
    except Exception:
        is_dng = False
    finally:
        try:
            file.seek(0)
        except Exception:
            pass

    if is_dng and dng_file is not None:
        return decode_raw(
            file=dng_file,
            use_xmp=use_xmp,
            output_dtype=output_dtype,
            **processing_params,
        )
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image")

    return _coerce_decoded_image(img, output_dtype=output_dtype)
