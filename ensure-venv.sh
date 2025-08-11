#!/bin/bash
# This script ensures the muimg package has a development venv with editable installs

set -e

PACKAGE="muimg"

echo "=== Checking Development Virtual Environment for $PACKAGE ==="

if [ -d "$PACKAGE" ]; then
    if [ ! -d "$PACKAGE/venv" ]; then
        echo "Creating development venv for $PACKAGE..."
        cd "$PACKAGE"
        python3 -m venv venv
        
        # Upgrade pip first
        venv/bin/pip install --upgrade pip
        
        # Install the package itself in editable mode
        venv/bin/pip install -e .
        
        cd ..
        echo "✓ Development venv created for $PACKAGE with editable dependencies"
    else
        echo "✓ Development venv already exists for $PACKAGE"
        echo "  - Skipping existing venv (delete and re-run to recreate with editable dependencies)"
    fi
else
    echo "⚠ Package directory $PACKAGE not found, skipping"
fi

echo "=== Development environment check complete ==="
