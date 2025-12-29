# InfiniteTalk Installation Guide

## Docker Installation (Recommended for RunPod/Cloud)

The project includes a Dockerfile with all dependencies pre-installed. This is the recommended method for RunPod or any cloud GPU provider.

### Quick Start with Docker

1. **Build the image** (from project root):
   ```bash
   ./build.sh
   ```
   This builds and pushes `jonathan28alkalay/infinite-talk:0.01` to Docker Hub.

2. **Use on RunPod**:
   - Create a template with `jonathan28alkalay/infinite-talk:0.01`
   - Expose port 22 for SSH (optional)
   - Mount persistent storage to `/workspace` (not `/app` - see note below)
   - Set environment variable: `PUBLIC_KEY=<your-ssh-public-key>`

3. **Download model weights** (in RunPod pod terminal):
   ```bash
   cd /app
   ./install_scripts/download_weights.sh
   ```

### Important Docker Notes
- Project code is baked into `/app` inside the container
- If you mount a volume to `/app`, it will overlay the built-in code (mount to `/workspace` instead)
- The conda environment `multitalk` is pre-activated
- All dependencies (PyTorch, xformers, flash-attn) are pre-installed

---

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
- **PyTorch**: 2.2.2+cu121 (pinned for CUDA 12.1)
- **xformers**: 0.0.25.post1 (pinned)
- **Flash Attention**: 2.7.4.post1 (compiled for SM90/Hopper)
- **xfuser**: 0.4.1 (pinned)

---

## Using the Docker Container

After the pod starts and weights are downloaded:

```bash
# Activate conda environment (if not already active)
source /opt/conda/etc/profile.d/conda.sh
conda activate multitalk

# Run inference
cd /app
python generate_infinitetalk.py --help
```

---

## Disk Space Optimization Tips

If you're running low on disk space:

1. **Use quantized models** - The `quant_models/` folder contains INT8 quantized versions that use ~60% less space
2. **Remove cache after download** - After installation, you can clear HuggingFace cache:
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```
3. **Use persistent volumes** - On RunPod, mount a large persistent volume to `/workspace` for weights

---

## Using the Installation

The conda environment is pre-activated in the Docker container. Just run:
```bash
cd /app
python generate_infinitetalk.py --help
```

---

## Troubleshooting

### Docker: `/app` directory is empty
This happens if you mount a volume to `/app`. Solution:
- Mount persistent storage to `/workspace` instead of `/app`
- The project code is baked into `/app` in the Docker image

### "Not enough disk space" error
- Free up at least 270GB
- Check with: `df -h`
- On RunPod, use a larger persistent volume
- Use quantized models instead (in `weights/InfiniteTalk/quant_models/`)

### `huggingface-cli` not found
The download script will auto-install it. If that fails:
```bash
pip install "huggingface_hub[cli,hf_transfer]"
```

### CUDA-related errors
Ensure CUDA 12.1 is available (should be in Docker image):
```bash
nvcc --version
nvidia-smi
```

### Flash Attention issues
Flash Attention is pre-compiled in the Docker image for Hopper (SM90). If you need to rebuild:
```bash
TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation --force-reinstall
```

---

## Post-Installation

Once setup is complete:
1. ✅ Docker container running with all dependencies
2. ✅ Conda environment `multitalk` activated
3. ✅ Model weights downloaded to `/app/weights/`
4. ✅ Ready to run inference

Next step: `python generate_infinitetalk.py --help` to see available options.
