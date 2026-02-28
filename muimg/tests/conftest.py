"""Pytest configuration and fixtures for muimg tests.

Handles automatic cloning of test fixtures from mu-test-fixtures repo.
"""

import subprocess
import sys
from pathlib import Path

# Fixtures configuration
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_REPO = "git@github.com:mu-files/mu-image-testdata.git"

# Paths that tests should use
TEST_FILES_DIR = FIXTURES_DIR / "dngtestfiles"
OUTPUT_COMPARISON_DIR = Path(__file__).parent / "output_comparison"


def ensure_fixtures():
    """Clone fixtures repo on first test run if not present."""
    if FIXTURES_DIR.exists() and (FIXTURES_DIR / "dngtestfiles").exists():
        return True
    
    print(f"\n[conftest] Fixtures not found at {FIXTURES_DIR}")
    print(f"[conftest] Cloning from {FIXTURES_REPO}...")
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", FIXTURES_REPO, str(FIXTURES_DIR)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[conftest] Successfully cloned fixtures repository.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[conftest] ERROR: Failed to clone fixtures repo:")
        print(f"[conftest]   {e.stderr}")
        print(f"[conftest] You may need to clone manually:")
        print(f"[conftest]   git clone {FIXTURES_REPO} {FIXTURES_DIR}")
        return False
    except FileNotFoundError:
        print("[conftest] ERROR: git command not found.")
        return False


# Run at import time - ensures fixtures are available before any tests run
_fixtures_available = ensure_fixtures()


def pytest_configure(config):
    """Pytest hook - verify fixtures are available."""
    if not _fixtures_available:
        print("\n[conftest] WARNING: Test fixtures are not available.")
        print("[conftest] Some tests may be skipped.\n")
