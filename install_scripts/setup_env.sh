#!/bin/bash
# InfiniteTalk Master Setup Script
# Runs environment setup and model weights download

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "InfiniteTalk Complete Setup"
echo "=========================================="
echo ""

# Step 1: Run environment setup
echo "Step 1/2: Setting up environment..."
echo ""
bash "$SCRIPT_DIR/install_dependencies.sh"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""

# Step 2: Run model weights download
echo "Step 2/2: Downloading model weights..."
echo ""
bash "$SCRIPT_DIR/download_weights.sh"

echo ""
echo "=========================================="
echo "Complete setup finished successfully!"
echo "=========================================="
echo ""
echo "To use InfiniteTalk:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run inference: python generate_infinitetalk.py --help"
