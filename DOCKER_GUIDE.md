# Guide Complet : Entraînement YOLOv8/YOLOv11 avec Docker Multi-Architecture

Ce guide détaille le processus complet d'entraînement d'un modèle YOLOv8 ou YOLOv11 personnalisé pour la détection d'objets, en utilisant Docker pour assurer la reproductibilité et la portabilité sur différentes architectures (ARM64, AMD64 avec/sans GPU).

## 🎯 Vue d'ensemble

### Architectures Supportées
- **ARM64** : Mac Apple Silicon (M1/M2/M3)
- **AMD64 CPU** : PC Intel/AMD sans GPU
- **AMD64 GPU** : PC Intel/AMD avec GPU NVIDIA

### Problématique Initiale
- Besoin d'entraîner un modèle YOLO sur un dataset personnalisé
- Incompatibilité des images Docker selon l'architecture
- Nécessité d'optimiser les paramètres selon le matériel disponible
- Solution : 3 Dockerfiles personnalisés pour chaque architecture

## Installation et Configuration

### Prérequis Généraux
- Docker Desktop installé
- Dataset au format YOLO (images + labels)
- Projet YOLOv8 existant

### Prérequis Spécifiques GPU (AMD64 uniquement)
- Drivers NVIDIA installés et à jour
- NVIDIA Container Toolkit : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- Docker avec support GPU activé

## Préparation du Dataset

### Format YOLO Requis
- **Images** : JPG/PNG dans `images/{train,valid,test}/`
- **Labels** : Fichiers `.txt` correspondants dans `labels/{train,valid,test}/`
- **Format label** : `class_id cx cy w h` (coordonnées normalisées 0-1)

### Fichier data.yaml
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4
names: ['car', 'truck', 'bus', 'van']
```
 Sources de Datasets : Roboflow (Export YOLOv8 directement)

## Création des Images Docker

Le projet inclut 3 Dockerfiles optimisés pour chaque architecture :

### 1️⃣ Dockerfile.arm64 (Mac Apple Silicon)

**Fichier : `Dockerfile.arm64`**
```dockerfile
FROM --platform=linux/arm64 python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Installation de PyTorch et Ultralytics
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install ultralytics>=8.3.205

CMD ["/bin/bash"]
```

**Construction :**
```bash
docker build --platform=linux/arm64 -f Dockerfile.arm64 -t username/my-yolo:arm64 .
```

### 2️⃣ Dockerfile.amd64 (PC Intel/AMD - CPU uniquement)

**Fichier : `Dockerfile.amd64`**
```dockerfile
FROM --platform=linux/amd64 python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Installation de PyTorch et Ultralytics
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install ultralytics>=8.3.205

CMD ["/bin/bash"]
```

**Construction :**
```bash
docker build --platform=linux/amd64 -f Dockerfile.amd64 -t username/my-yolo:amd64 .
```

### 3️⃣ Dockerfile.gpu (PC avec GPU NVIDIA)

**Fichier : `Dockerfile.gpu`**
```dockerfile
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# Installation de Python et dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Installation de PyTorch avec support CUDA et Ultralytics
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install ultralytics>=8.3.205

CMD ["/bin/bash"]
```

**Construction :**
```bash
docker build --platform=linux/amd64 -f Dockerfile.gpu -t username/my-yolo:gpu .
```

## Déploiement sur Autre Machine

### Via Docker Hub 

**1. Pousser l'image :**
```bash
# Pour ARM64
docker push username/my-yolo:arm64

# Pour AMD64 CPU
docker push username/my-yolo:amd64

# Pour AMD64 GPU
docker push username/my-yolo:gpu
```

**2. Récupérer sur une autre machine :**
```bash
# Sur Mac Apple Silicon
docker pull username/my-yolo:arm64

# Sur PC Intel/AMD
docker pull username/my-yolo:amd64

# Sur PC avec GPU NVIDIA
docker pull username/my-yolo:gpu
```

## Entraînement du Modèle

### Sur Mac Apple Silicon (ARM64)

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  username/my-yolo:arm64 \
  yolo train model=yolov8n.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=50 imgsz=640 batch=8 workers=0 project=/workspace/runs
```

### Sur PC Intel/AMD - CPU uniquement (AMD64)

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  username/my-yolo:amd64 \
  yolo train model=yolo11m.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=100 imgsz=640 batch=16 workers=4 project=/workspace/runs
```

### Sur PC avec GPU NVIDIA (AMD64)

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  username/my-yolo:gpu \
  yolo train model=yolov8x.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=500 imgsz=1024 batch=32 device=0 project=/workspace/runs
```


## Interprétation des Résultats

### Fichiers Générés
- `runs/train/weights/best.pt` : Meilleur modèle (mAP le plus élevé)
- `runs/train/weights/last.pt` : Dernier modèle (dernier epoch)
- `runs/train/results.png` : Courbes d'entraînement
- `runs/train/results.csv` : Métriques détaillées

### Métriques Clés
- **mAP50** : Précision moyenne à IoU=0.5 (objectif principal)
- **mAP50-95** : Précision moyenne sur IoU 0.5-0.95 (plus strict)
- **Precision** : % de détections correctes
- **Recall** : % d'objets réels détectés

### Signes de Bon Entraînement
- ✅ Losses qui diminuent régulièrement
- ✅ mAP qui augmente et se stabilise
- ✅ Precision et Recall équilibrés
- ❌ Pas de sur-apprentissage (courbes train/val qui divergent)

### Exemple de Résultats (5 epochs)
- **mAP50** : 0 → 0.24 (bon début)
- **mAP50-95** : 0 → 0.11 (en progression)
- **Precision** : 0 → 0.95 (excellent)
- **Recall** : 0 → 0.06-0.16 (à améliorer)

## Ressources Supplémentaires

- [Documentation Ultralytics](https://docs.ultralytics.com/)
- [YOLO Training Tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- [Docker Multi-platform](https://docs.docker.com/build/building/multi-platform/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)