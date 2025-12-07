#!/bin/bash
# InfiniteTalk Environment Setup Script

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"

# Step 1: Create venv
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv "$VENV_PATH"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Step 2: Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install xformers
echo "Installing xformers..."
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install Flash Attention dependencies
echo "Installing Flash Attention dependencies..."
pip install misaki[en] ninja psutil packaging wheel

# Step 5: Install Flash Attention (this compiles and takes time)
echo "Installing Flash Attention (may take 5-10 minutes)..."
pip install flash_attn==2.7.4.post1

# Step 6: Install requirements.txt
echo "Installing project requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Step 6.5: Pin xfuser to 0.4.1 for PyTorch 2.4.1 compatibility
echo "Pinning xfuser to 0.4.1..."
pip install xfuser==0.4.1

# Step 6.6: Reinstall xformers after xfuser pinning (in case it was removed)
echo "Ensuring xformers is installed..."
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Step 7: Install librosa
echo "Installing librosa..."
pip install librosa

# Step 7.5: Install hf_transfer for faster model downloads
echo "Installing hf_transfer..."
pip install hf_transfer

# Step 8: Check and install FFmpeg if needed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found, attempting to install..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg ffmpeg-devel
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo "ERROR: Cannot install FFmpeg automatically. Please install manually:"
        echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "  CentOS/RHEL: sudo yum install ffmpeg ffmpeg-devel"
        echo "  Or download from: https://ffmpeg.org/download.html"
        exit 1
    fi
else
    echo "FFmpeg is already installed"
fi

# More steps will be added here as we test them
