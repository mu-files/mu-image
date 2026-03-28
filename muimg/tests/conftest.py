"""Pytest configuration and fixtures for muimg tests.

Auto-clones test data from sibling mu-image-testdata repo if not present.
Provides shared test utilities for DNG validation and comparison.
"""

import subprocess
from pathlib import Path

import numpy as np
import tifffile


# Test data lives in sibling repo: mu-image-testdata
# Expected structure:
#   python/mu-image/          (this repo)
#   python/mu-image-testdata/ (sibling repo with test files)
# conftest.py is at: mu-image/muimg/tests/conftest.py
_MUIMG_ROOT = Path(__file__).parent.parent.parent  # .../mu-image
_TESTDATA_REPO = _MUIMG_ROOT.parent / "mu-image-testdata"
_TESTDATA_REPO_URL = "git@github.com:mu-files/mu-image-testdata.git"

# Paths that tests should use
TEST_FILES_DIR = _TESTDATA_REPO / "dngtestfiles"
LOCAL_TEST_FILES_DIR = _TESTDATA_REPO / "dngtestfiles" / "local_testfiles"  # gitignored in testdata repo
OUTPUT_COMPARISON_DIR = Path(__file__).parent / "output_comparison"


def _ensure_testdata():
    """Clone test data repo if not present."""
    if TEST_FILES_DIR.exists():
        return True
    
    print(f"\n[conftest] Test data not found, cloning {_TESTDATA_REPO_URL}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", _TESTDATA_REPO_URL, str(_TESTDATA_REPO)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[conftest] Cloned to {_TESTDATA_REPO}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[conftest] ERROR: Failed to clone: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[conftest] ERROR: git not found")
        return False


# Clone at import time
_testdata_available = _ensure_testdata()


def pytest_configure(config):
    """Configure pytest: warn if test data missing."""
    if not _testdata_available:
        print("\n[conftest] WARNING: Test data unavailable. Some tests will be skipped.\n")


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
