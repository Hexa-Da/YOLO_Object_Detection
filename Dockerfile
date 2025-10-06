FROM --platform=linux/arm64/v8 python:3.11-slim
# FROM --platform=linux/amd64 python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# libs utiles pour OpenCV/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# Laisse pip choisir les roues aarch64 natives
RUN pip install ultralytics==8.3.205 torch torchvision

# yolo est dispo via ultralytics (console script)