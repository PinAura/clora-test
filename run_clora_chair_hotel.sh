#!/bin/bash

echo "============================================"
echo "CLoRA Chair + Hotel Room Composition Script"
echo "============================================"
echo

# Check if we're in the correct directory
if [ ! -f "pipeline_clora.py" ]; then
    echo "ERROR: pipeline_clora.py not found. Please run this script from the CLoRA repository directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "[INFO] Current directory: $(pwd)"
echo "[INFO] Checking for required LoRA files..."

# Check for LoRA files
if [ ! -f "Tantra__Chair.safetensors" ]; then
    echo "ERROR: Tantra__Chair.safetensors not found in current directory"
    echo "Please ensure the chair LoRA file is in: $(pwd)"
    exit 1
fi

if [ ! -f "lovehotel_SD15_V7.safetensors" ]; then
    echo "ERROR: lovehotel_SD15_V7.safetensors not found in current directory"
    echo "Please ensure the hotel LoRA file is in: $(pwd)"
    exit 1
fi

echo "[SUCCESS] Found both LoRA files:"
echo "  - Tantra__Chair.safetensors"
echo "  - lovehotel_SD15_V7.safetensors"
echo

# Check if UV is installed
echo "[INFO] Checking for UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "[INFO] UV not found. Installing UV..."
    
    # Install UV using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install UV. Please install it manually from https://github.com/astral-sh/uv"
        exit 1
    fi
    
    # Source the shell configuration to get UV in PATH
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    
    # Try UV again
    if ! command -v uv &> /dev/null; then
        echo "ERROR: UV installation failed or not in PATH. Please restart your shell and try again."
        echo "Or manually add UV to your PATH and re-run this script."
        exit 1
    fi
fi

echo "[SUCCESS] UV is available"
uv --version
echo

# Install dependencies
echo "[INFO] Installing Python dependencies with UV..."
uv sync

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies with UV"
    exit 1
fi

echo "[SUCCESS] Dependencies installed successfully"
echo

# Python script already exists, skip creation
echo "[INFO] Using existing Python script: clora_chair_hotel.py"
echo

# Run the Python script
echo "[INFO] Running CLoRA composition..."
echo "[INFO] This may take 5-15 minutes depending on your GPU..."
echo

uv run python clora_chair_hotel.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: CLoRA execution failed. Check the error messages above."
    exit 1
fi

echo
echo "============================================"
echo "SUCCESS: CLoRA composition completed!"
echo "============================================"
echo
echo "Check the current directory for the output image:"
echo "$(pwd)"
echo
echo "The image should be named: chair_in_hotel_room_YYYYMMDD_HHMMSS.png"
echo

# List generated images
echo "Generated images:"
ls -la chair_in_hotel_room_*.png 2>/dev/null || echo "No output images found. Please check the error messages above."

echo
echo "Script completed."
