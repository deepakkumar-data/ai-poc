# Waste Segregation System - Setup Script for Windows (PowerShell)
# Uses Python venv for package management

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Waste Segregation System on Windows..." -ForegroundColor Cyan
Write-Host ""

# Get the directory where this script is located
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = $SCRIPT_DIR
$REQUIREMENTS_FILE = Join-Path $PROJECT_ROOT "waste_segregation_m4\requirements.txt"
$CHECK_MPS_FILE = Join-Path $PROJECT_ROOT "waste_segregation_m4\utils\check_mps.py"

# Verify requirements file exists
if (-not (Test-Path $REQUIREMENTS_FILE)) {
    Write-Host "❌ Error: requirements.txt not found at: $REQUIREMENTS_FILE" -ForegroundColor Red
    Write-Host "   Please ensure you're running this script from the project root" -ForegroundColor Red
    exit 1
}

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "   Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Change to project root directory
Set-Location $PROJECT_ROOT

# Create virtual environment with Python 3.11+
Write-Host ""
Write-Host "🐍 Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host ""
Write-Host "📦 Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install packages
Write-Host ""
Write-Host "📚 Installing packages..." -ForegroundColor Cyan
python -m pip install -r $REQUIREMENTS_FILE
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install packages" -ForegroundColor Red
    exit 1
}

# Verify PyTorch CUDA/CPU support
Write-Host ""
Write-Host "🔍 Verifying PyTorch installation..." -ForegroundColor Cyan
if (Test-Path $CHECK_MPS_FILE) {
    python $CHECK_MPS_FILE
} else {
    Write-Host "⚠️  Warning: check_mps.py not found at: $CHECK_MPS_FILE" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✨ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then run the application:" -ForegroundColor Cyan
Write-Host "  cd waste_segregation_m4" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor Yellow
Write-Host ""
