#!/bin/bash

# Activate the multitalk conda environment
# Usage: source install_scripts/activate_env.sh

source /opt/conda/etc/profile.d/conda.sh
conda activate multitalk

echo "✓ Conda environment 'multitalk' activated"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
