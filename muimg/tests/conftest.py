"""Pytest configuration and fixtures for muimg tests.

Auto-clones test data from sibling mu-image-testdata repo if not present.
"""

import subprocess
from pathlib import Path

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
    """Pytest hook - warn if test data unavailable."""
    if not _testdata_available:
        print("\n[conftest] WARNING: Test data unavailable. Some tests will be skipped.\n")
