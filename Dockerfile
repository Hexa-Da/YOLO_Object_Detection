# FROM --platform=linux/arm64/v8 python:3.11-slim
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# libs utiles pour OpenCV/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
# Laisse pip choisir les roues aarch64 natives
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installer Ultralytics depuis PyPI
RUN pip3 install ultralytics==8.3.205
