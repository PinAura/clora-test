@echo off
setlocal enabledelayedexpansion

echo ============================================
echo CLoRA Chair + Hotel Room Composition Script
echo ============================================
echo.

:: Check if we're in the correct directory
if not exist "pipeline_clora.py" (
    echo ERROR: pipeline_clora.py not found. Please run this script from the CLoRA repository directory.
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [INFO] Current directory: %CD%
echo [INFO] Checking for required LoRA files...

:: Check for LoRA files
if not exist "models/chair/Tantra__Chair.safetensors" (
    echo ERROR: Tantra__Chair.safetensors not found in current directory
    echo Please ensure the chair LoRA file is in: %CD%
    pause
    exit /b 1
)

if not exist "models/hotelroom/lovehotel_SD15_V7.safetensors" (
    echo ERROR: lovehotel_SD15_V7.safetensors not found in current directory
    echo Please ensure the hotel LoRA file is in: %CD%
    pause
    exit /b 1
)

echo [SUCCESS] Found both LoRA files:
echo   - Tantra__Chair.safetensors
echo   - lovehotel_SD15_V7.safetensors
echo.

:: Check if UV is installed
echo [INFO] Checking for UV package manager...
uv --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] UV not found. Installing UV...
    echo [INFO] Downloading UV installer...
    
    :: Download and install UV
    powershell -Command "& {Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile 'install_uv.ps1'; .\install_uv.ps1; Remove-Item 'install_uv.ps1'}"
    
    if errorlevel 1 (
        echo ERROR: Failed to install UV. Please install it manually from https://github.com/astral-sh/uv
        pause
        exit /b 1
    )
    
    :: Refresh PATH
    call refreshenv >nul 2>&1
    
    :: Try UV again
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: UV installation failed or not in PATH. Please restart your command prompt and try again.
        pause
        exit /b 1
    )
)

echo [SUCCESS] UV is available
uv --version
echo.

:: Install dependencies
echo [INFO] Installing Python dependencies with UV...
uv sync
if errorlevel 1 (
    echo ERROR: Failed to install dependencies with UV
    pause
    exit /b 1
)

echo [SUCCESS] Dependencies installed successfully
echo.

:: Python script already exists, skip creation
echo [INFO] Using existing Python script: clora_chair_hotel.py
echo.

:: Run the Python script
echo [INFO] Running CLoRA composition...
echo [INFO] This may take 5-15 minutes depending on your GPU...
echo.

uv run python clora_chair_hotel.py

if errorlevel 1 (
    echo.
    echo ERROR: CLoRA execution failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS: CLoRA composition completed!
echo ============================================
echo.
echo Check the current directory for the output image:
echo %CD%
echo.
echo The image should be named: chair_in_hotel_room_YYYYMMDD_HHMMSS.png
echo.

:: List generated images
echo Generated images:
dir /b chair_in_hotel_room_*.png 2>nul
if errorlevel 1 (
    echo No output images found. Please check the error messages above.
)

echo.
echo Press any key to exit...
pause >nul
