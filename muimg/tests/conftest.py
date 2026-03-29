"""Pytest configuration and fixtures for muimg tests.

Provides shared test utilities for DNG validation and comparison.
"""

import subprocess
from pathlib import Path

import numpy as np
import tifffile


def core_image_available_for_tests() -> bool:
    try:
        from muimg._dngio_coreimage import core_image_available

        return bool(core_image_available)
    except ImportError:
        return False


# =============================================================================
# Shared Test Utilities
# =============================================================================

# Path to the C++ SDK dng_validate tool (reference)
DNG_VALIDATE_PATH = Path.home() / "Projects/C/3dparty/dng_sdk_1_7_1/dng_sdk/targets/mac/release64/dng_validate"


def generate_rgb_ramp(width: int, height: int) -> np.ndarray:
    """Generate synthetic RGB ramp test image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        uint16 RGB image (H, W, 3) with:
        - Red channel: 0-65535 gradient left to right
        - Blue channel: 0-65535 gradient top to bottom
        - Green channel: 0-65535 gradient on diagonal
    """
    img = np.zeros((height, width, 3), dtype=np.uint16)
    
    # Red: left to right gradient
    img[:, :, 0] = np.linspace(0, 65535, width, dtype=np.uint16)[np.newaxis, :]
    
    # Blue: top to bottom gradient
    img[:, :, 2] = np.linspace(0, 65535, height, dtype=np.uint16)[:, np.newaxis]
    
    # Green: diagonal gradient (top-left to bottom-right)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    diagonal = (xx / width + yy / height) / 2.0
    img[:, :, 1] = (diagonal * 65535).astype(np.uint16)
    
    return img


def sample_as_cfa(rgb_img: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Sample RGB image as CFA (Bayer pattern).
    
    Extracts color channels at RGGB positions from full resolution RGB image.
    
    Args:
        rgb_img: RGB image (H, W, 3) in uint16 format
        pattern: CFA pattern, currently only "RGGB" supported
        
    Returns:
        uint16 CFA array (H, W) - single channel with Bayer pattern
        
    Raises:
        ValueError: If pattern is not "RGGB"
    """
    if pattern != "RGGB":
        raise ValueError(f"Only RGGB pattern supported, got {pattern}")
    
    height, width = rgb_img.shape[:2]
    cfa = np.zeros((height, width), dtype=np.uint16)
    
    # RGGB pattern:
    # Row 0 (even): R G R G ...
    # Row 1 (odd):  G B G B ...
    
    # Even rows, even cols: Red
    cfa[0::2, 0::2] = rgb_img[0::2, 0::2, 0]
    
    # Even rows, odd cols: Green
    cfa[0::2, 1::2] = rgb_img[0::2, 1::2, 1]
    
    # Odd rows, even cols: Green
    cfa[1::2, 0::2] = rgb_img[1::2, 0::2, 1]
    
    # Odd rows, odd cols: Blue
    cfa[1::2, 1::2] = rgb_img[1::2, 1::2, 2]
    
    return cfa


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to float [0,1] range."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def compute_diff_stats(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compute difference statistics between two images."""
    diff = np.abs(normalize_image(img1) - normalize_image(img2))
    stats = {
        "mean": np.mean(diff) * 100,
        "p99": np.percentile(diff, 99) * 100,
        "max": np.max(diff) * 100,
    }
    
    # Add per-channel stats if RGB image
    if img1.ndim == 3 and img1.shape[2] == 3:
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            ch_diff = diff[:, :, ch_idx]
            stats[f"mean_{ch_name}"] = np.mean(ch_diff) * 100
            stats[f"p99_{ch_name}"] = np.percentile(ch_diff, 99) * 100
            stats[f"max_{ch_name}"] = np.max(ch_diff) * 100
    
    return stats


def load_tiff(path: Path) -> np.ndarray | None:
    """Load TIFF and convert to interleaved format if needed."""
    if path is None or not path.exists():
        return None
    try:
        with tifffile.TiffFile(str(path)) as tif:
            img = tif.pages[0].asarray()
            # Convert planar (3,H,W) to interleaved (H,W,3)
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[0] < img.shape[1]:
                img = np.moveaxis(img, 0, -1)
            return img
    except Exception:
        return None


def run_dng_validate(dng_path: Path, output_base: Path, timeout: int = 120, ignored_warnings: list[str] | None = None, validate: bool = True) -> np.ndarray | None:
    """Run dng_validate to render a DNG to TIFF.
    
    Args:
        dng_path: Path to DNG file
        output_base: Base path for output (will append .tif)
        timeout: Timeout in seconds
        ignored_warnings: Optional list of warning patterns to ignore (case-insensitive)
        validate: If False, ignore all errors/warnings and just decode (for reference comparison)
        
    Returns:
        Loaded TIFF as numpy array, or None if dng_validate not available
        
    Raises:
        RuntimeError: If dng_validate fails or produces errors (only when validate=True)
        AssertionError: If dng_validate produces warnings (only when validate=True, except ignored ones)
    """
    # Warnings to ignore (add patterns here as needed)
    IGNORED_WARNINGS = ignored_warnings or []
    
    if not DNG_VALIDATE_PATH.exists():
        return None
    
    output_tiff = Path(str(output_base) + ".tif")
    
    try:
        result = subprocess.run(
            [str(DNG_VALIDATE_PATH), "-v", "-16", "-tif", str(output_base), str(dng_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"dng_validate failed with return code {result.returncode}:\n{result.stderr}")
        
        # Only check for errors/warnings if validate=True
        if validate:
            # Check for errors or warnings in output
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            
            # Extract only error/warning lines for cleaner output
            error_lines = [line for line in combined.split('\n') if '*** error:' in line.lower()]
            warning_lines = [line for line in combined.split('\n') if '*** warning:' in line.lower()]
            
            if error_lines:
                errors_text = '\n'.join(error_lines)
                raise RuntimeError(f"dng_validate produced errors:\n{errors_text}")
            
            if warning_lines:
                # Check if this is an ignored warning
                warnings_text = '\n'.join(warning_lines)
                print(f"\ndng_validate warnings:\n{warnings_text}")
                warnings_lower = warnings_text.lower()
                is_ignored = any(ignored in warnings_lower for ignored in IGNORED_WARNINGS)
                if not is_ignored:
                    raise AssertionError(f"dng_validate produced warnings:\n{warnings_text}")
            
    except (subprocess.TimeoutExpired, Exception) as e:
        if isinstance(e, (RuntimeError, AssertionError)):
            raise
        raise RuntimeError(f"dng_validate error: {e}")
    
    return load_tiff(output_tiff)
