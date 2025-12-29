#!/bin/bash
# InfiniteTalk Model Weights Download Script
# For use in Docker container with conda environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$PROJECT_DIR/weights"

echo "=========================================="
echo "InfiniteTalk Model Weights Download"
echo "=========================================="
echo ""
echo "This will download ~100GB+ of model weights"
echo "Make sure you have enough disk space and a stable internet connection"
echo ""

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "multitalk" ]; then
    echo "Activating conda environment 'multitalk'..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate multitalk
fi

# Enable fast downloads with hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create weights directory
mkdir -p "$WEIGHTS_DIR"

# HuggingFace CLI (hf) and hf_transfer are preinstalled in the Docker image.
# If you run outside the image and see errors, install them manually:
#   pip install "huggingface_hub[cli]" hf_transfer

# Download Wan2.1-I2V-14B-480P base model
echo "Downloading Wan2.1-I2V-14B-480P base model..."
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WEIGHTS_DIR/Wan2.1-I2V-14B-480P"

# Download chinese-wav2vec2-base audio encoder
echo "Downloading chinese-wav2vec2-base audio encoder..."
hf download TencentGameMate/chinese-wav2vec2-base --local-dir "$WEIGHTS_DIR/chinese-wav2vec2-base"

# Download specific model.safetensors for wav2vec2
echo "Downloading specific wav2vec2 model file..."
hf download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir "$WEIGHTS_DIR/chinese-wav2vec2-base"

# Download InfiniteTalk weights
echo "Downloading InfiniteTalk weights..."
hf download MeiGen-AI/InfiniteTalk --local-dir "$WEIGHTS_DIR/InfiniteTalk"

# Download FusionX LoRA for faster inference (optional but recommended)
echo "Downloading FusionX LoRA accelerator..."
hf download vrgamedevgirl84/Wan14BT2VFusioniX FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors --local-dir "$WEIGHTS_DIR/FusionX"

echo ""
echo "=========================================="
echo "All model weights downloaded successfully!"
echo "=========================================="
echo "Weights location: $WEIGHTS_DIR"
