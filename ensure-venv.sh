#!/bin/bash
# This script ensures the muimg package has a development venv with editable installs

set -e

PACKAGE="muimg"

echo "=== Checking Development Virtual Environment for $PACKAGE ==="

if [ -d "$PACKAGE" ]; then
    if [ ! -d "$PACKAGE/venv" ]; then
        echo "Creating development venv for $PACKAGE..."
        cd "$PACKAGE"
        /opt/homebrew/bin/python3.11 -m venv venv
        
        # Upgrade pip first
        venv/bin/pip install --upgrade pip
        
        # Install the package itself in editable mode with test extras
        venv/bin/pip install -e .[test]
        
        cd ..
        echo "✓ Development venv created for $PACKAGE with editable dependencies"
    else
        echo "✓ Development venv already exists for $PACKAGE"
        echo "  - Ensuring test extras are installed"
        (
          cd "$PACKAGE"
          venv/bin/pip install -e .[test] >/dev/null 2>&1 || true
        )
    fi
else
    echo "⚠ Package directory $PACKAGE not found, skipping"
fi

echo "=== Development environment check complete ==="
