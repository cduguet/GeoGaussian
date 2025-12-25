FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for older Python versions
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install system dependencies
RUN apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install pip for python 3.8
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && python3.8 get-pip.py && rm get-pip.py

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other dependencies
RUN pip install --ignore-installed \
    tqdm \
    plyfile \
    scipy \
    matplotlib \
    open3d \
    pillow \
    ninja

WORKDIR /app

# Set architecture list for A100 (8.0) and force CUDA build
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV FORCE_CUDA="1"

# Copy submodules and install them
COPY submodules /app/submodules
RUN pip install /app/submodules/diff-gaussian-rasterization
RUN pip install /app/submodules/simple-knn