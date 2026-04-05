"""Pytest configuration and fixtures for muimg tests.

Provides shared test utilities for DNG validation and comparison.
"""

import subprocess
from pathlib import Path

import numpy as np
import tifffile

from muimg.raw_render import convert_dtype


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


def generate_rgb_ramp(width: int, height: int, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Generate synthetic RGB ramp test image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        dtype: Output data type (np.uint8, np.uint16, np.float32, etc.)
        
    Returns:
        RGB image (H, W, 3) with specified dtype:
        - Red channel: gradient left to right
        - Blue channel: gradient top to bottom
        - Green channel: gradient on diagonal
    """
    # Generate in float32 for highest precision
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    # Red: left to right gradient (0.0 to 1.0)
    img[:, :, 0] = np.linspace(0.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
    
    # Blue: top to bottom gradient (0.0 to 1.0)
    img[:, :, 2] = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    
    # Green: diagonal gradient (top-left to bottom-right, 0.0 to 1.0)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    diagonal = (xx / width + yy / height) / 2.0
    img[:, :, 1] = diagonal
    
    # Convert to target dtype if needed
    if dtype != np.float32:
        img = convert_dtype(img, dtype)
    
    return img


def sample_as_cfa(rgb_img: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Sample RGB image as CFA (Bayer pattern).
    
    Extracts color channels at RGGB positions from full resolution RGB image.
    
    Args:
        rgb_img: RGB image (H, W, 3) - any numeric dtype
        pattern: CFA pattern, currently only "RGGB" supported
        
    Returns:
        CFA array (H, W) with same dtype as input - single channel with Bayer pattern
        
    Raises:
        ValueError: If pattern is not "RGGB"
    """
    if pattern != "RGGB":
        raise ValueError(f"Only RGGB pattern supported, got {pattern}")
    
    height, width = rgb_img.shape[:2]
    cfa = np.zeros((height, width), dtype=rgb_img.dtype)
    
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


def run_dng_validate(dng_path: Path, output_base: Path, timeout: int = 120, ignored_warnings: list[str] | None = None, validate: bool = True, indent: str = "") -> np.ndarray | None:
    """Run dng_validate and muimg metadata validators on a DNG file.
    
    Args:
        dng_path: Path to DNG file
        output_base: Base path for output (will append .tif)
        timeout: Timeout in seconds
        ignored_warnings: Optional list of warning patterns to ignore (case-insensitive)
        validate: If False, ignore all errors/warnings and just decode (for reference comparison)
        indent: String to prepend to validation messages (default: no indent)
        
    Returns:
        Loaded TIFF as numpy array, or None if dng_validate not available
        
    Raises:
        RuntimeError: If dng_validate fails or produces errors (only when validate=True)
        AssertionError: If either validator produces warnings (only when validate=True, except ignored ones)
    """
    # Warnings to ignore (add patterns here as needed)
    IGNORED_WARNINGS = ignored_warnings or []
    
    if not DNG_VALIDATE_PATH.exists():
        return None
    
    output_tiff = Path(str(output_base) + ".tif")
    all_warnings = []
    
    try:
        # Run dng_validate (C++ SDK validator)
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
            # Check for errors or warnings in dng_validate output
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            
            # Extract only error/warning lines for cleaner output
            # Note: dng_validate uses "*** Error:" and "*** Warning:" (capital E/W)
            error_lines = [line for line in combined.split('\n') if line.strip().lower().startswith('*** error:')]
            warning_lines = [line for line in combined.split('\n') if line.strip().lower().startswith('*** warning:')]
            
            # Collect warnings and errors from dng_validate (errors can be ignored too)
            if warning_lines:
                all_warnings.extend(warning_lines)
            if error_lines:
                all_warnings.extend(error_lines)
            
            # Run muimg dng metadata validator
            import sys
            muimg_cmd = [sys.executable, "-m", "muimg.cli", "dng", "metadata", str(dng_path)]
            muimg_result = subprocess.run(
                muimg_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Extract validation issues from muimg output
            if muimg_result.returncode == 0:
                output_lines = muimg_result.stdout.split('\n')
                in_validation_section = False
                for line in output_lines:
                    if '=== DNG Validation Issues ===' in line:
                        in_validation_section = True
                        continue
                    if in_validation_section and line.strip().startswith('*-'):
                        # Convert muimg format to dng_validate format
                        # "*-1: DefaultScale found in ifd0 IFD, should be in raw IFD"
                        # becomes "*** Warning: DefaultScale found in ifd0 IFD, should be in raw IFD ***"
                        issue_text = line.strip().split(':', 1)[1].strip() if ':' in line else line.strip()
                        all_warnings.append(f"*** Warning: {issue_text} ***")
            
            # Filter out ignored warnings/errors, checking each individually
            if all_warnings:
                unignored_warnings = []
                ignored_count = 0
                for warning in all_warnings:
                    warning_lower = warning.lower()
                    is_ignored = any(ignored in warning_lower for ignored in IGNORED_WARNINGS)
                    if not is_ignored:
                        unignored_warnings.append(warning)
                    else:
                        ignored_count += 1
                
                # Only print unignored warnings/errors to make failures clear
                if unignored_warnings:
                    unignored_text = '\n'.join(unignored_warnings)
                    if ignored_count > 0:
                        print(f"{indent}Validation issues ({ignored_count} ignored):")
                    else:
                        print(f"{indent}Validation issues:")
                    print(unignored_text)
                    # Check if any are errors (start with "*** Error:")
                    has_errors = any(line.strip().lower().startswith('*** error:') for line in unignored_warnings)
                    if has_errors:
                        error_summary = '; '.join([line.strip() for line in unignored_warnings if line.strip().lower().startswith('*** error:')])
                        raise RuntimeError(f"dng_validate errors: {error_summary}")
                    else:
                        raise AssertionError(f"Validation produced {len(unignored_warnings)} unignored warning(s)")
                elif ignored_count > 0:
                    print(f"{indent}Validation: {ignored_count} warning(s) ignored")
            
    except (subprocess.TimeoutExpired, Exception) as e:
        if isinstance(e, (RuntimeError, AssertionError)):
            raise
        raise RuntimeError(f"Validation error: {e}")
    
    return load_tiff(output_tiff)
