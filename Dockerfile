# InfiniteTalk Dockerfile (CUDA 12.1, Python 3.10)
# Mirrors install_scripts/install_dependencies.sh without downloading weights.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Enable faster HF downloads inside the container (user triggers manually later)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System deps: build tools and utilities; Python via Conda
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

# Install Miniforge (conda-forge, avoids Anaconda TOS; arch-aware)
RUN arch=$(uname -m) \
  && if [ "$arch" = "x86_64" ]; then MF_ARCH="x86_64"; \
     elif [ "$arch" = "aarch64" ]; then MF_ARCH="aarch64"; \
     else echo "Unsupported architecture: $arch" && exit 1; fi \
  && wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MF_ARCH}.sh" -O /tmp/miniforge.sh \
  && bash /tmp/miniforge.sh -b -p /opt/conda \
  && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:${PATH}
SHELL ["/bin/bash", "-lc"]

# Copy only requirements first for better build caching
COPY requirements.txt ./

# Create conda environment and install conda-forge packages (ffmpeg, librosa)
RUN conda create -y -n multitalk python=3.10 \
  && conda install -y -n multitalk -c conda-forge ffmpeg librosa \
  && conda clean -afy

# Pip installs inside the conda env per README and install script
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate multitalk \
  && pip install --upgrade pip \
  # PyTorch stack pinned for reproducibility (CUDA 12.1 wheels)
  && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
       torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  # xformers pinned to matching CUDA
  && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 xformers==0.0.28 \
  # Flash Attention dependencies
  && pip install --no-cache-dir "misaki[en]" ninja psutil packaging wheel \
  # Flash Attention (disable build isolation to see installed torch)
  && TORCH_CUDA_ARCH_LIST="9.0" pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1 \
    # HuggingFace CLI and fast transfer support baked into image
    && pip install --no-cache-dir "huggingface_hub[cli]" hf_transfer \
  # Project requirements
  && pip install --no-cache-dir -r requirements.txt \
  # xfuser pin
  && pip install --no-cache-dir xfuser==0.4.1

# Copy the rest of the project
COPY . .

# Create temp directory for temporary files (weights directory created by download_weights.sh)
RUN mkdir -p /app/temp

# Default to using the conda env executables
ENV CONDA_DEFAULT_ENV=multitalk
ENV PATH=/opt/conda/envs/multitalk/bin:/opt/conda/bin:${PATH}
ENV PYTHONUNBUFFERED=1

# No weights are downloaded at build time; user will handle inside container

# Default to running as interactive shell, but can be overridden for RunPod
CMD ["/bin/bash"]
