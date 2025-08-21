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
if [ ! -f "models/chair/Tantra__Chair.safetensors" ]; then
    echo "ERROR: Tantra__Chair.safetensors not found in current directory"
    echo "Please ensure the chair LoRA file is in: $(pwd)"
    exit 1
fi

if [ ! -f "models/hotelroom/lovehotel_SD15_V7.safetensors" ]; then
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

    # Detect where UV was actually installed
    UV_INSTALL_PATH=""
    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_INSTALL_PATH="$HOME/.local/bin"
        echo "[INFO] UV installed to $HOME/.local/bin"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        UV_INSTALL_PATH="$HOME/.cargo/bin"
        echo "[INFO] UV installed to $HOME/.cargo/bin"
    elif [ -f "/root/.local/bin/uv" ]; then
        UV_INSTALL_PATH="/root/.local/bin"
        echo "[INFO] UV installed to /root/.local/bin"
    elif [ -f "/root/.cargo/bin/uv" ]; then
        UV_INSTALL_PATH="/root/.cargo/bin"
        echo "[INFO] UV installed to /root/.cargo/bin"
    else
        echo "ERROR: Could not find UV installation. Checked common locations:"
        echo "  - $HOME/.local/bin/uv"
        echo "  - $HOME/.cargo/bin/uv"
        echo "  - /root/.local/bin/uv"
        echo "  - /root/.cargo/bin/uv"
        exit 1
    fi

    # Add UV to PATH for this session
    export PATH="$UV_INSTALL_PATH:$PATH"

    # Also try to add to common shell config files for future sessions
    UV_PATH_LINE="export PATH=\"$UV_INSTALL_PATH:\$PATH\""

    # Add to .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "$UV_INSTALL_PATH" "$HOME/.bashrc"; then
            echo "# Added by CLoRA setup script" >> "$HOME/.bashrc"
            echo "$UV_PATH_LINE" >> "$HOME/.bashrc"
            echo "[INFO] Added UV to PATH in ~/.bashrc"
        fi
    fi

    # Add to .zshrc if it exists
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "$UV_INSTALL_PATH" "$HOME/.zshrc"; then
            echo "# Added by CLoRA setup script" >> "$HOME/.zshrc"
            echo "$UV_PATH_LINE" >> "$HOME/.zshrc"
            echo "[INFO] Added UV to PATH in ~/.zshrc"
        fi
    fi

    # Add to .profile as fallback
    if [ -f "$HOME/.profile" ]; then
        if ! grep -q "$UV_INSTALL_PATH" "$HOME/.profile"; then
            echo "# Added by CLoRA setup script" >> "$HOME/.profile"
            echo "$UV_PATH_LINE" >> "$HOME/.profile"
            echo "[INFO] Added UV to PATH in ~/.profile"
        fi
    fi

    # Verify UV is now available
    if ! command -v uv &> /dev/null; then
        echo "ERROR: UV installation failed or not in PATH."
        echo "UV should be installed at: $UV_INSTALL_PATH/uv"
        echo "Please manually add $UV_INSTALL_PATH to your PATH and re-run this script."
        echo "You can do this by running: export PATH=\"$UV_INSTALL_PATH:\$PATH\""
        exit 1
    fi
fi

echo "[SUCCESS] UV is available"
uv --version
echo

# Check and install system dependencies for matplotlib
echo "[INFO] Checking system dependencies for matplotlib..."
if ! command -v make &> /dev/null; then
    echo "[INFO] GNU make not found. Installing build dependencies..."

    # Detect package manager and install dependencies
    if command -v apt-get &> /dev/null; then
        echo "[INFO] Using apt-get to install dependencies..."
        apt-get update -qq
        apt-get install -y build-essential pkg-config libfreetype6-dev libpng-dev
    elif command -v yum &> /dev/null; then
        echo "[INFO] Using yum to install dependencies..."
        yum install -y gcc gcc-c++ make pkgconfig freetype-devel libpng-devel
    elif command -v dnf &> /dev/null; then
        echo "[INFO] Using dnf to install dependencies..."
        dnf install -y gcc gcc-c++ make pkgconfig freetype-devel libpng-devel
    elif command -v pacman &> /dev/null; then
        echo "[INFO] Using pacman to install dependencies..."
        pacman -S --noconfirm base-devel freetype2 libpng
    else
        echo "WARNING: Could not detect package manager. You may need to manually install:"
        echo "  - build-essential (gcc, g++, make)"
        echo "  - pkg-config"
        echo "  - freetype development headers"
        echo "  - libpng development headers"
        echo ""
        echo "Continuing anyway - UV might be able to build without system packages..."
    fi
else
    echo "[SUCCESS] Build tools are available"
fi

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
