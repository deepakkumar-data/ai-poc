# Waste Segregation System

A deep learning-based waste segregation system with cross-platform support:
- **macOS**: Optimized for Mac Mini M4 with 16-core Neural Engine (MPS acceleration)
- **Windows**: Supports CUDA GPU acceleration or CPU fallback
- **Linux**: Supports CUDA GPU acceleration or CPU fallback

## Project Structure

```
.
├── waste_segregation_m4/     # Main project directory
│   ├── app.py                # Main Streamlit UI
│   ├── classifier.py         # MPS-accelerated model logic
│   ├── conveyor_engine.py    # OpenCV ROI & Video processing
│   ├── utils/
│   │   ├── check_mps.py      # Hardware verification
│   │   └── export_coreml.py  # Model conversion script
│   ├── models/               # Local storage for weights
│   └── requirements.txt      # Python dependencies
├── setup.sh                  # UV-based setup script (macOS/Linux)
├── setup_conda.sh            # Conda-based setup script (macOS/Linux)
├── setup.bat                 # Windows batch setup script
├── setup.ps1                  # Windows PowerShell setup script
└── README.md                 # This file
```

## Prerequisites

### macOS
- macOS 12.3+ (Monterey or later) for MPS support
- Mac Mini M4 with Apple Silicon (recommended) or any Apple Silicon Mac
- Python 3.11 or later

### Windows
- Windows 10 or later
- Python 3.11 or later
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Visual C++ Redistributable (usually installed with Python)

### Linux
- Ubuntu 20.04+ or similar distribution
- Python 3.11 or later
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

## Quick Start

### macOS

#### Option 1: Using UV (Recommended - Faster)

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
cd waste_segregation_m4
streamlit run app.py
```

#### Option 2: Using Conda

```bash
chmod +x setup_conda.sh
./setup_conda.sh
conda activate waste_segregation
cd waste_segregation_m4
streamlit run app.py
```

### Windows

#### Option 1: Using Batch Script (Recommended)

```cmd
setup.bat
.venv\Scripts\activate
cd waste_segregation_m4
streamlit run app.py
```

#### Option 2: Using PowerShell

```powershell
.\setup.ps1
.venv\Scripts\Activate.ps1
cd waste_segregation_m4
streamlit run app.py
```

#### Option 3: Manual Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r waste_segregation_m4\requirements.txt
cd waste_segregation_m4
streamlit run app.py
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r waste_segregation_m4/requirements.txt
cd waste_segregation_m4
streamlit run app.py
```

## Verify Installation

After setup, verify that PyTorch can access your hardware:

```bash
# macOS
python waste_segregation_m4/utils/check_mps.py

# Windows
python waste_segregation_m4\utils\check_mps.py

# Linux
python3 waste_segregation_m4/utils/check_mps.py
```

This will check:
- **macOS**: MPS (Metal Performance Shaders) and Neural Engine
- **Windows/Linux**: CUDA GPU support (if available) or CPU fallback

## Usage

See the [waste_segregation_m4/README.md](waste_segregation_m4/README.md) for detailed usage instructions.

The system uses:
- **PyTorch**: 
  - MPS acceleration on macOS (Apple Silicon)
  - CUDA acceleration on Windows/Linux (NVIDIA GPU)
  - CPU fallback on all platforms
- **MLX**: For Neural Engine acceleration (macOS only, optional)
- **OpenCV**: For image processing (cross-platform)
- **Transformers**: For pre-trained models
- **Streamlit**: For web interface

## Platform-Specific Notes

### macOS
- MPS backend requires macOS 12.3+ and PyTorch 1.12+
- The 16-core Neural Engine is accessible via Core ML and MLX frameworks
- All packages are installed with native arm64 support for optimal performance
- Camera uses AVFoundation backend (native macOS support)

### Windows
- CUDA acceleration requires NVIDIA GPU with CUDA toolkit installed
- Camera uses DirectShow backend (native Windows support)
- If CUDA is not available, the system automatically falls back to CPU
- Visual C++ Redistributable may be required (usually comes with Python)

### Linux
- CUDA acceleration requires NVIDIA GPU with CUDA toolkit installed
- Camera uses Video4Linux2 backend
- If CUDA is not available, the system automatically falls back to CPU

## Cross-Platform Features

- ✅ Automatic device detection (MPS/CUDA/CPU)
- ✅ Platform-appropriate camera backends
- ✅ Cross-platform file paths
- ✅ Same functionality on all platforms

Main application code is in the `waste_segregation_m4/` directory