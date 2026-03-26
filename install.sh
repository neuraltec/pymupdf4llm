#!/bin/bash
# Installation script for pymupdf4llm

set -e

echo "=========================================="
echo "pymupdf4llm installation"
echo "=========================================="
echo ""

if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    echo "   Run this script from the project root directory."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found!"
    echo "   Install Python 3.10 or newer."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python found: $(python3 --version)"

if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]; then
    echo "⚠️  Warning: Python 3.10 or newer is recommended"
fi

VENV_NAME="venv_pymupdf"
if [ -d "$VENV_NAME" ]; then
    echo "✓ Virtual environment '$VENV_NAME' found"
    USE_EXISTING=true
else
    echo "ℹ️  Virtual environment '$VENV_NAME' not found"
    USE_EXISTING=false
fi

if [ "$USE_EXISTING" = true ]; then
    echo ""
    read -p "Use existing virtual environment '$VENV_NAME'? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        USE_EXISTING=true
    else
        USE_EXISTING=false
    fi
fi

if [ "$USE_EXISTING" = false ]; then
    echo ""
    read -p "Virtual environment name (default: venv): " VENV_NAME_INPUT
    VENV_NAME=${VENV_NAME_INPUT:-venv}
    
    if [ -d "$VENV_NAME" ]; then
        echo "⚠️  Virtual environment '$VENV_NAME' already exists!"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            echo "✓ Virtual environment removed"
        else
            echo "Using existing virtual environment..."
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        echo ""
        echo "Creating virtual environment '$VENV_NAME'..."
        python3 -m venv "$VENV_NAME"
        echo "✓ Virtual environment created"
    fi
fi

VENV_PATH="$(cd "$VENV_NAME" && pwd)"
PIP_CMD="$VENV_PATH/bin/pip"
PYTHON_CMD="$VENV_PATH/bin/python"

if [ ! -f "$PIP_CMD" ]; then
    echo "❌ Error: pip not found at $PIP_CMD"
    exit 1
fi

if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Error: python not found at $PYTHON_CMD"
    exit 1
fi

echo ""
echo "✓ Using virtual environment pip: $PIP_CMD"
echo "✓ Using virtual environment python: $PYTHON_CMD"

source "$VENV_PATH/bin/activate" 2>/dev/null || true

echo ""
echo "Updating pip..."
"$PIP_CMD" install --upgrade pip --quiet
echo "✓ pip updated"

echo ""
echo "Installing pymupdf4llm in development mode..."
cd pymupdf4llm
"$PIP_CMD" install -e . --quiet
cd ..
echo "✓ pymupdf4llm installed"

echo ""
echo "Installing development dependencies..."
"$PIP_CMD" install -r requirements-dev.txt --quiet
echo "✓ Dependencies installed"

echo ""
echo "Checking installation..."
if "$PYTHON_CMD" -c "import pymupdf4llm; print('OK')" 2>/dev/null; then
    echo "✓ pymupdf4llm imported successfully"
else
    echo "⚠️  Warning: Could not import pymupdf4llm"
fi

PYTEST_CMD="$VENV_NAME/bin/pytest"
if [ -f "$PYTEST_CMD" ] && "$PYTEST_CMD" --version &> /dev/null; then
    echo "✓ pytest installed: $("$PYTEST_CMD" --version)"
else
    echo "⚠️  Warning: pytest not found"
fi

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To use this virtual environment later:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""

