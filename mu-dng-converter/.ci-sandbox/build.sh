#!/bin/bash
# Local CI sandbox - clones repo and runs exact commands from build.yml
# Usage: ./build.sh [branch]
#   branch: optional, defaults to "main" (use "dev-branch" for dev)
set -e

BRANCH="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
VENV_DIR="$SCRIPT_DIR/venv"

# Always start fresh: remove old clone, build, and dist
echo "=== Cleaning previous build ==="
rm -rf "$REPO_DIR"

# Clone fresh copy from GitHub (like actions/checkout@v4)
echo "=== Cloning mu-image from GitHub (branch: $BRANCH) ==="
git clone --depth 1 --branch "$BRANCH" https://github.com/mu-files/mu-image.git "$REPO_DIR"

WORKFLOW="$REPO_DIR/.github/workflows/build.yml"
PROJECT_DIR="$REPO_DIR/mu-dng-converter"

# Ensure yq is installed
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    brew install yq
fi

# One-time venv setup (Python 3.12 like CI)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 venv..."
    python3.14 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Extract and run steps from build.yml
cd "$PROJECT_DIR"

echo "=== Install dependencies (from build.yml) ==="
yq '.jobs.build.steps[] | select(.name == "Install dependencies") | .run' "$WORKFLOW" | bash

echo "=== Install macOS dependencies (from build.yml) ==="
yq '.jobs.build.steps[] | select(.name == "Install macOS dependencies") | .run' "$WORKFLOW" | bash

echo "=== Build app (macOS) (from build.yml) ==="
yq '.jobs.build.steps[] | select(.name == "Build app (macOS)") | .run' "$WORKFLOW" | bash

echo ""
echo "Done. Test with:"
echo "  open $PROJECT_DIR/dist/mu-dng-converter.app"
