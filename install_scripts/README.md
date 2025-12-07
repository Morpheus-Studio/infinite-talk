# InfiniteTalk Installation Guide

## System Requirements

### Storage Space
⚠️ **IMPORTANT: You need approximately 270GB of free disk space** for model weights downloads:
- **Wan2.1-I2V-14B-480P**: ~80GB (image-to-video model)
- **chinese-wav2vec2-base**: ~0.5GB (audio encoder)
- **InfiniteTalk**: ~60GB (video generation + quantized models)
- **Cache & overhead**: ~130GB (HuggingFace hub cache, extraction buffers)

**Total: ~270GB minimum required**

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA 12.1 support (RTX 3090, A100, H100, etc.)
- **RAM**: 32GB+ VRAM recommended (supports up to 80GB for 14B models)
- **CPU RAM**: 64GB+ system RAM recommended
- **Compute**: CUDA 12.1 capable device

### Software Requirements
- Python 3.10+
- FFmpeg (for video processing)
- CUDA Toolkit 12.1
- cuDNN compatible with CUDA 12.1

### Important Version Notes
- **PyTorch**: 2.4.1 (pinned for stability)
- **xfuser**: 0.4.1 (pinned - newer versions require PyTorch 2.5+)
- **Flash Attention**: 2.7.4.post1 (requires compilation)

## Installation Steps

### Step 1: Install Dependencies
```bash
bash install_scripts/install_dependencies.sh
```

This script:
- Creates a Python 3.10 virtual environment at `.venv`
- Installs PyTorch 2.4.1 with CUDA 12.1 support
- Installs xformers and Flash Attention (requires compilation, takes 5-10 min)
- Installs all project requirements from `requirements.txt`
- Installs FFmpeg for video processing
- Sets up HuggingFace CLI tools for model downloads

### Step 2: Download Model Weights
```bash
bash install_scripts/download_weights.sh
```

⚠️ **This will download ~100GB+ of model files. Make sure you have:**
- Stable internet connection (high bandwidth recommended)
- 270GB+ free disk space
- Time for download to complete (can take a while)

Downloaded models will be stored in `./weights/` directory:
```
weights/
├── Wan2.1-I2V-14B-480P/          (image-to-video model)
├── chinese-wav2vec2-base/         (audio encoder)
└── InfiniteTalk/                  (video generation models)
    ├── quant_models/              (quantized versions for lower VRAM)
    └── single/                    (main single-person model)
```

### One-Command Complete Installation
```bash
bash install_scripts/setup_env.sh
```

This runs both scripts sequentially: dependencies → model downloads

## Disk Space Optimization Tips

If you're running low on disk space:

1. **Use quantized models** - The `quant_models/` folder contains INT8 quantized versions that use ~60% less space
2. **Remove cache after download** - After installation, you can clear HuggingFace cache:
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```
3. **Use external storage** - Download to an external SSD/HDD and mount it to your project directory

## Using the Installation

After installation completes, activate the environment:
```bash
source .venv/bin/activate
```

Then run the inference script:
```bash
python generate_infinitetalk.py --help
```

## Troubleshooting

### "Not enough disk space" error
- Free up at least 270GB
- Check with: `df -h`
- Use quantized models instead (in `weights/InfiniteTalk/quant_models/`)

### FFmpeg not found
The script will attempt to install FFmpeg automatically. If it fails:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **CentOS/RHEL**: `sudo yum install ffmpeg ffmpeg-devel`
- **macOS**: `brew install ffmpeg`

### CUDA-related errors
Ensure CUDA 12.1 is installed and set in PATH:
```bash
nvcc --version
```

### Flash Attention compilation fails
This can happen on some systems. Try:
```bash
pip install flash-attn --no-build-isolation
```

## Internet Speed Considerations

Model download speeds:
- **1 Mbps**: ~24 hours for 270GB
- **10 Mbps**: ~6 hours
- **100 Mbps**: ~36 minutes
- **1 Gbps**: ~3-4 minutes

The script uses `hf_transfer` to maximize download speeds with HuggingFace Hub.

## Post-Installation

Once setup is complete:
1. ✅ Virtual environment ready at `.venv/`
2. ✅ All dependencies installed
3. ✅ All model weights downloaded to `./weights/`
4. ✅ Ready to run inference

Next step: `python generate_infinitetalk.py --help` to see available options.
