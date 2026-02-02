#!/bin/bash

# Waste Segregation System - Setup Script for Mac Mini M4
# Uses conda for Python package management with native arm64 support

set -e

echo "🚀 Setting up Waste Segregation System on Mac Mini M4..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment with Python 3.11+ and arm64 architecture
echo "🐍 Creating conda environment with Python 3.11 (arm64)..."
conda create -n waste_segregation python=3.11 -y

# Activate conda environment
echo "✅ Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate waste_segregation

# Install packages with conda-forge (better arm64 support)
echo "📚 Installing packages with native arm64 support..."
conda install -c conda-forge pytorch torchvision transformers opencv streamlit -y
pip install mlx
pip install -r waste_segregation_m4/requirements.txt

# Verify PyTorch MPS support
echo "🔍 Verifying PyTorch MPS backend..."
python waste_segregation_m4/utils/check_mps.py

echo "✨ Setup complete! Activate the environment with: conda activate waste_segregation"
