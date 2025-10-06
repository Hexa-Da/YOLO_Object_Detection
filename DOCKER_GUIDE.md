# Guide Complet : Entraînement YOLOv8 avec Docker

Ce guide détaille le processus complet d'entraînement d'un modèle YOLOv8 personnalisé pour la détection de véhicules sur images aériennes, en utilisant Docker pour assurer la reproductibilité et la portabilité.

### Problématique Initiale
- Besoin d'entraîner un modèle YOLOv8 sur un dataset personnalisé
- Incompatibilité de l'image officielle Ultralytics avec Mac M1/M2 (architecture ARM64)
- Solution : Création d'une image Docker ARM64 personnalisée

## Installation et Configuration

### Prérequis
- Docker Desktop installé
- Dataset au format YOLO (images + labels)
- Projet YOLOv8 existant


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

### Sources de Datasets
1. **Roboflow** (recommandé) : Export YOLOv8 directement
2. **COCO128** : Dataset d'exemple pour tests
3. **Datasets personnalisés** : Conversion manuelle nécessaire

## Création de l'Image Docker

### Dockerfile ARM64
```dockerfile
FROM --platform=linux/arm64/v8 python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
WORKDIR /workspace

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Installation Python
RUN pip install --upgrade pip
RUN pip install ultralytics==8.3.205 torch torchvision
```

### Construction de l'Image
```bash
docker build --platform=linux/arm64/v8 -t my-yolo-arm64 .
```

### Pour Architecture x86_64
```dockerfile
FROM --platform=linux/amd64 python:3.11-slim
# ... reste identique
```

## Entraînement du Modèle

### Commande d'Entraînement
```bash
docker run --rm -it \
  -v "/Users/paul-antoine/Documents/PYTHON/yolov8_object_detection":/workspace \
  my-yolo-arm64 \
  yolo train model=yolov8n.pt \
       data="/workspace/datasets/my_dataset/Aerial Cars.v2-aerial_cars.yolov8/data.yaml" \
       epochs=50 imgsz=640 batch=8 workers=0 project=/workspace/runs
```

### Paramètres Recommandés

#### Mac M1/M2 (CPU)
```bash
epochs=50
imgsz=640
batch=8-16
workers=4-8
```

#### PC avec GPU NVIDIA
```bash
epochs=100
imgsz=800
batch=32-64
workers=8-16
--gpus all
```

#### Serveur Cloud
```bash
epochs=200+
imgsz=1024
batch=64+
workers=16+
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

## Utilisation du Modèle Entraîné

### Test avec analyze_image.py
```python
# Modifier dans main()
model_path = "runs/train/weights/best.pt"
```

### Test avec analyze_video.py
```python
# Modifier dans main()
model_path = "runs/train/weights/best.pt"
```

### Prédiction Directe
```bash
docker run --rm -it \
  -v "/Users/paul-antoine/Documents/PYTHON/yolov8_object_detection":/workspace \
  my-yolo-arm64 \
  yolo predict model=/workspace/runs/train/weights/best.pt \
       source=/workspace/test.jpg
```

### Évaluation du Modèle
```bash
docker run --rm -it \
  -v "/Users/paul-antoine/Documents/PYTHON/yolov8_object_detection":/workspace \
  my-yolo-arm64 \
  yolo val model=/workspace/runs/train/weights/best.pt \
       data="/workspace/datasets/my_dataset/Aerial Cars.v2-aerial_cars.yolov8/data.yaml"
```

## Déploiement sur Autre Machine

### Avantages Docker
- **Portabilité** : Même environnement partout
- **Reproductibilité** : Résultats identiques
- **Isolation** : Pas de conflits de dépendances

### Étapes de Déploiement


```bash
# Option A : Docker Hub
docker tag my-yolo-arm64 username/my-yolo-arm64:latest
docker push username/my-yolo-arm64:latest

# Sur l'autre machine
docker pull username/my-yolo-arm64:latest
```

```bash
# Option B : Fichier
docker save my-yolo-arm64 -o my-yolo-arm64.tar
# Transférer le fichier
docker load -i my-yolo-arm64.tar