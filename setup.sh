#!/bin/bash

# Waste Segregation System - Setup Script for Mac Mini M4
# Uses uv for fast Python package management with native arm64 support

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
REQUIREMENTS_FILE="$PROJECT_ROOT/waste_segregation_m4/requirements.txt"
CHECK_MPS_FILE="$PROJECT_ROOT/waste_segregation_m4/utils/check_mps.py"

echo "🚀 Setting up Waste Segregation System on Mac Mini M4..."
echo "📁 Project root: $PROJECT_ROOT"

# Verify requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Error: requirements.txt not found at: $REQUIREMENTS_FILE"
    echo "   Please ensure you're running this script from the project root"
    exit 1
fi

# Determine uv command path
UV_CMD="uv"
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Try to source the env file
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    
    # Use full path if still not in PATH
    if ! command -v uv &> /dev/null; then
        UV_CMD="$HOME/.local/bin/uv"
        if [ ! -f "$UV_CMD" ]; then
            echo "❌ Error: Could not find uv after installation"
            exit 1
        fi
    fi
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Create virtual environment with Python 3.11+ (recommended for M4)
echo "🐍 Creating virtual environment with uv..."
$UV_CMD venv --python 3.11

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install packages with arm64 native support
echo "📚 Installing packages with native arm64 support..."
$UV_CMD pip install --upgrade pip
$UV_CMD pip install -r "$REQUIREMENTS_FILE"

# Verify PyTorch MPS support
echo "🔍 Verifying PyTorch MPS backend..."
if [ -f "$CHECK_MPS_FILE" ]; then
    python "$CHECK_MPS_FILE"
else
    echo "⚠️  Warning: check_mps.py not found at: $CHECK_MPS_FILE"
fi

echo "✨ Setup complete! Activate the environment with: source .venv/bin/activate"