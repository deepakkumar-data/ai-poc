# Waste Segregation System

A deep learning-based waste segregation system optimized for Mac Mini M4 with 16-core Neural Engine.

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
├── setup.sh                  # UV-based setup script
├── setup_conda.sh            # Conda-based setup script
└── README.md                 # This file
```

## Prerequisites

- macOS 12.3+ (Monterey or later)
- Mac Mini M4 with Apple Silicon
- Python 3.11 or later

## Quick Start

### Option 1: Using UV (Recommended - Faster)

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
cd waste_segregation_m4
streamlit run app.py
```

### Option 2: Using Conda

```bash
chmod +x setup_conda.sh
./setup_conda.sh
conda activate waste_segregation
cd waste_segregation_m4
streamlit run app.py
```

## Verify Installation

After setup, verify that PyTorch can access MPS and the Neural Engine:

```bash
python waste_segregation_m4/utils/check_mps.py
```

## Usage

See the [waste_segregation_m4/README.md](waste_segregation_m4/README.md) for detailed usage instructions.

The system uses:
- **PyTorch with MPS**: For GPU acceleration on Apple Silicon
- **MLX**: For Neural Engine acceleration
- **OpenCV**: For image processing
- **Transformers**: For pre-trained models
- **Streamlit**: For web interface

## Notes

- MPS backend requires macOS 12.3+ and PyTorch 1.12+
- The 16-core Neural Engine is accessible via Core ML and MLX frameworks
- All packages are installed with native arm64 support for optimal performance
- Main application code is in the `waste_segregation_m4/` directory