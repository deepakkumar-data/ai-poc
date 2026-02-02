@echo off
REM Waste Segregation System - Setup Script for Windows
REM Uses Python venv for package management

setlocal enabledelayedexpansion

echo 🚀 Setting up Waste Segregation System on Windows...
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%waste_segregation_m4\requirements.txt"
set "CHECK_MPS_FILE=%PROJECT_ROOT%waste_segregation_m4\utils\check_mps.py"

REM Verify requirements file exists
if not exist "%REQUIREMENTS_FILE%" (
    echo ❌ Error: requirements.txt not found at: %REQUIREMENTS_FILE%
    echo    Please ensure you're running this script from the project root
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo    Please install Python 3.11+ from https://www.python.org/downloads/
    exit /b 1
)

echo ✅ Python found
python --version

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

REM Create virtual environment with Python 3.11+
echo.
echo 🐍 Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo ✅ Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install packages
echo.
echo 📚 Installing packages...
python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo ❌ Failed to install packages
    exit /b 1
)

REM Verify PyTorch CUDA/CPU support
echo.
echo 🔍 Verifying PyTorch installation...
if exist "%CHECK_MPS_FILE%" (
    python "%CHECK_MPS_FILE%"
) else (
    echo ⚠️  Warning: check_mps.py not found at: %CHECK_MPS_FILE%
)

echo.
echo ✨ Setup complete!
echo.
echo To activate the environment, run:
echo   .venv\Scripts\activate
echo.
echo Then run the application:
echo   cd waste_segregation_m4
echo   streamlit run app.py
echo.

pause
