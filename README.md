# Analyseur d'images et vidéos YOLOv8 + PyTorch + Multi-Datasets + Entraînement Personnalisé

Ce projet analyse des images et vidéos en utilisant YOLOv8 pré-entraîné sur différents datasets avec PyTorch pour détecter, classifier et segmenter les objets selon les classes disponibles dans chaque dataset. Il inclut également la possibilité d'entraîner des modèles personnalisés avec Docker.

## 🚀 Installation

### Méthode 1 : Installation automatique (Recommandée)

```bash
# 1. Aller dans le dossier du projet
cd YOLO_Object_Detection

# 2. Exécuter le script d'installation
chmod +x install.sh
./install.sh
```

Le script d'installation :
- ✅ Vérifie que Python 3 et pip sont installés
- 📦 Crée un environnement virtuel (recommandé)
- 📥 Installe automatiquement toutes les dépendances
- 🔧 Rend les scripts exécutables

### Méthode 2 : Installation manuelle

#### Étape 1 : Créer un environnement virtuel

```bash
# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate  
```

#### Étape 2 : Installer les dépendances

```bash
# Avec l'environnement virtuel activé
pip install -r requirements.txt
```

**Dépendances installées :**
- `ultralytics>=8.0.0` - Framework YOLOv8 (8.3.209 pour YOLOv11)
- `torch>=1.9.0` - PyTorch pour le deep learning
- `torchvision>=0.10.0` - Vision utilities
- `opencv-python>=4.5.0` - Traitement d'images
- `Pillow>=8.0.0` - Manipulation d'images
- `matplotlib>=3.3.0` - Visualisation
- `numpy>=1.21.0` - Calculs numériques

## 🤔 Pourquoi un environnement virtuel ?

### Le problème sur macOS avec Homebrew

Système géré par **Homebrew**, qui empêche l'installation directe de packages pour protéger votre installation système :

```
error: externally-managed-environment
```

### Avantages de l'environnement virtuel

1. **🔒 Sécurité** : N'affecte pas votre Python système
2. **🎯 Isolation** : Chaque projet a ses propres versions de packages
3. **🔄 Reproductibilité** : Même environnement sur différentes machines
4. **🧹 Propreté** : Facile à supprimer si besoin
5. **⚡ Performance** : Évite les conflits de versions

### Activation de l'environnement virtuel

**Important :** Vous devez activer l'environnement virtuel à chaque nouvelle session terminal :

```bash
cd YOLO_Object_Detection
source venv/bin/activate
```

**Comment savoir si l'environnement est activé :**
- Le prompt affiche `(venv)` au début
- `which python` pointe vers le dossier `venv/bin/python`


## 📸 Analyse d'images

### Analyser une image avec différents datasets

```bash
# Lancer l'analyse d'image
python analyze_image.py
```

### Ce que fait le script d'analyse d'images

1. **📊 Affiche les informations** sur le dataset sélectionné
2. **📥 Charge l'image** spécifiée
3. **🤖 Charge le modèle** YOLOv8 pré-entraîné (avec cache pour éviter les rechargements)
4. **🔍 Analyse l'image** avec PyTorch et YOLOv8 selon les classes du dataset
5. **📊 Affiche les résultats** : objets détectés + ID + confiance + position
6. **🖼️ Crée une visualisation** avec des boîtes de détection et contours de segmentation

## 🎬 Analyse de vidéos

### Analyser une vidéo en temps réel

```bash
# Lancer l'analyse de vidéo
python analyze_video.py
```

### Ce que fait le script d'analyse de vidéos

1. **📊 Affiche les informations** sur le dataset sélectionné
2. **📥 Charge la vidéo** spécifiée
3. **🤖 Charge le modèle** YOLOv8 pré-entraîné (avec cache pour éviter les rechargements)
4. **🔍 Analyse chaque frame** avec PyTorch et YOLOv8 selon les classes du dataset
5. **📊 Affiche les statistiques** : frames traitées, détections totales, FPS de traitement
6. **🖼️ Affiche la vidéo en temps réel** avec des boîtes de détection et contours de segmentation
7. **⌨️ Contrôles** : Appuyez sur 'q' pour quitter l'analyse

### Fonctionnalités spéciales pour les vidéos

- **🎥 Affichage en temps réel** : Visualisation frame par frame avec annotations
- **📊 Statistiques en direct** : Compteur de frames, détections par frame
- **🖥️ Redimensionnement automatique** : Adaptation à la taille d'écran (max 2560x1440)

## ⚙️ Configuration des datasets

### Utilisation des modèles pré-entraînés

Dans `analyze_image.py` et `analyze_video.py`, vous pouvez choisir le dataset à utiliser :

```python
# Exemples de modèles pour différents datasets :
model_paths = {
    # DÉTECTION COCO (80 classes)
    'n-coco': "yolov8n.pt",           
    's-coco': "yolov8s.pt",           
    'm-coco': "yolov8m.pt",           
    'l-coco': "yolov8l.pt",           
    'x-coco': "yolov8x.pt",           
    
    # SEGMENTATION COCO (80 classes + masques)
    'n-seg': "yolov8n-seg.pt",
    's-seg': "yolov8s-seg.pt", 
    'm-seg': "yolov8m-seg.pt",
    'l-seg': "yolov8l-seg.pt",
    'x-seg': "yolov8x-seg.pt",
    
    # DÉTECTION Open Images V7 (600 classes)
    'n-oiv7': "yolov8n-oiv7.pt",      
    's-oiv7': "yolov8s-oiv7.pt",      
    'm-oiv7': "yolov8m-oiv7.pt",      
    'l-oiv7': "yolov8l-oiv7.pt",      
    'x-oiv7': "yolov8x-oiv7.pt",  

    # Modèles personnalisés
    'custom-trained': "runs/train/weights/best.pt",   

    # Modèles YOLOv11
    'nv11-coco': "yolo11n.pt",
    'sv11-coco': "yolo11s.pt",
    'mv11-coco': "yolo11m.pt",
    'lv11-coco': "yolo11l.pt",
    'xv11-coco': "yolo11x.pt",
}

# Choisir le dataset à utiliser
dataset_choice = 'm-seg'  # Segmentation COCO Medium
```

### 🎯 Dataset COCO (Common Objects in Context)

**Le plus populaire et performant pour la détection d'objets généraux**

#### Caractéristiques
- **🔢 80 classes d'objets** 
- **📈 118,287 images d'entraînement**
- **📈 5,000 images de validation**
- **📈 40,670 images de test**
- **🎯 1.5 million d'objets annotés**
- **🌐 Site officiel** : https://cocodataset.org/

#### Modèles disponibles
- **Détection** : `yolov8n.pt` à `yolov8x.pt` (boîtes de détection)
- **Segmentation** : `yolov8n-seg.pt` à `yolov8x-seg.pt` (masques de pixels)

### 🌐 Dataset Open Images V7

**Dataset très large avec de nombreuses classes spécialisées**

#### Caractéristiques
- **🔢 600 classes d'objets** très variées
- **📈 Plus de 9 millions d'images**
- **🎯 Plus de 36 millions de boîtes de détection**
- **🌐 Site officiel** : https://storage.googleapis.com/openimages/web/index.html

#### Modèles disponibles
- **Détection uniquement** : `yolov8n-oiv7.pt` à `yolov8x-oiv7.pt`
- **❌ Pas de segmentation** : Open Images V7 ne propose pas de modèles de segmentation

#### Avantages
- **🎯 Classes très spécifiques** : Différents types de chiens, voitures, etc.
- **🌍 Diversité** : Couvre de nombreux domaines spécialisés
- **📊 Échelle** : Dataset très large

#### Inconvénients
- **⚠️ Précision moindre** : Moins performant que COCO sur les objets généraux
- **📊 Classes déséquilibrées** : Certaines classes très rares
- **🔍 Complexité** : Plus difficile à utiliser efficacement

### 🔄 Système de cache intelligent

Le script utilise un **système de cache global** pour éviter de recharger les modèles :

```python
# Cache global pour les modèles chargés
_model_cache = {}

def get_model(model_path):
    """
    Récupère un modèle depuis le cache ou le charge s'il n'existe pas
    """
    if model_path not in _model_cache:
        print(f"🔄 Chargement du modèle {model_path}...")
        _model_cache[model_path] = YOLO(model_path)
        print(f"✅ Modèle {model_path} chargé avec succès")
    else:
        print(f"♻️  Utilisation du modèle {model_path} depuis le cache")
    
    return _model_cache[model_path]
```

## 🧠 Comment fonctionne YOLO ?

### Qu'est-ce que YOLO ?

**YOLO** (You Only Look Once) est un réseau de neurones convolutifs spécialisé dans la **détection d'objets en temps réel**.

### Principe de fonctionnement

1. **🔍 Analyse globale** : YOLO examine l'image entière en une seule fois
2. **📦 Détection simultanée** : Trouve tous les objets en parallèle
3. **🎯 Prédiction directe** : Prédit directement les boîtes de détection
4. **⚡ Vitesse** : Beaucoup plus rapide que les méthodes traditionnelles

### Processus de detection et segmentation

1. **📥 Entrée** : Image redimensionnée à 640x640 pixels
2. **🧠 Traitement** : Le réseau extrait des caractéristiques
3. **🎯 Prédiction** : Pour chaque zone, prédit :
   - **Position** : Coordonnées de la boîte (x1, y1, x2, y2)
   - **Confiance** : Probabilité que ce soit un objet (0-1)
   - **Classe** : Type d'objet selon le dataset utilisé
   - **Masque** : Masque de pixels précis de l'objet
4. **🔍 Filtrage** : Garde seulement les détections avec confiance > seuil
5. **📊 Résultat** : Liste des objets segmentés avec boîtes + masques

### Modèles YOLOv8 disponibles

| Taille | Détection COCO | Segmentation COCO | Détection OIV7 | Vitesse | Précision | Usage |
|--------|----------------|-------------------|----------------|---------|-----------|-------|
| Nano   | yolov8n.pt | yolov8n-seg.pt | yolov8n-oiv7.pt | +++ | ++ | Mobile, IoT |
| Small  | yolov8s.pt | yolov8s-seg.pt | yolov8s-oiv7.pt | ++ | +++ | Équilibre |
| **Medium** | **yolov8m.pt** | **yolov8m-seg.pt** | **yolov8m-oiv7.pt** | + | ++++ | **Recommandé** |
| Large  | yolov8l.pt | yolov8l-seg.pt | yolov8l-oiv7.pt | - | +++++ | Haute précision |
| Huge   | yolov8x.pt | yolov8x-seg.pt | yolov8x-oiv7.pt | -- | +++++ | Recherche |

### YOLOv8 vs YOLOv11

| Caractéristique | YOLOv8 | YOLOv11 |
|----------------|---------|----------|
| **Année** | 2023 | 2024 |
| **mAP COCO** | 53.9% (m) | 54.7% (m) |
| **Vitesse** | Référence | Légèrement plus lent |
| **Précision** | Excellente | Améliorée (+0.8%) |
| **Architecture** | C2f backbone | C3k2 + C2PSA |
| **Paramètres** | Plus léger | Légèrement plus lourd |
| **Usage recommandé** | Production | Haute précision |


## 🎓 Entraînement de Modèles Personnalisés

### Prérequis
- Docker Desktop installé
- Dataset au format YOLO (images + labels)

### Structure pour l'Entraînement
```
YOLO_Object_Detection/
├── datasets/
│   └── my_dataset/
│       └── Nom_du_dataset.yolov8/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── valid/
│           │   ├── images/
│           │   └── labels/
│           └── data.yaml
├── runs/
│   └── train/
│       ├── weights/
│       │   ├── best.pt      # Meilleur modèle
│       │   └── last.pt      # Dernier modèle
│       └── results.png      # Courbes d'entraînement
├── Dockerfile               # Image Docker pour entraînement
├── DOCKER_GUIDE.md          # Guide complet d'entraînement
└── ...
```

### Entraînement avec Docker

#### Sur Mac (ARM64)

1. **Créer l'image Docker :**
```bash
docker build -f Dockerfile.arm64 -t my-yolo-arm64 .
```

2. **Lancer l'entraînement :**
```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  my-yolo-arm64 \
  yolo train model=yolov8n.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=50 imgsz=640 batch=8 workers=0 project=/workspace/runs
```

#### Sur PC Linux/Windows (AMD64)

1. **Créer l'image Docker :**
```bash
docker build -f Dockerfile.amd64 -t my-yolo-amd64 .
```

2. **Lancer l'entraînement :**
```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  my-yolo-amd64 \
  yolo train model=yolov8x.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=100 imgsz=640 batch=16 workers=4 project=/workspace/runs
```

#### Sur PC avec GPU (AMD64)

1. **Créer l'image Docker :**
```bash
docker build -f Dockerfile.gpu -t my-yolo-gpu . 
```

2. **Lancer l'entraînement :**
```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  my-yolo-gpu \
  yolo train model=yolov8x-seg.pt \
       data="/workspace/datasets/my_dataset/data.yaml" \
       epochs=500 imgsz=640 batch=32 device=0 project=/workspace/runs
```

3. **Utiliser le modèle entraîné :**
```python
# Dans analyze_image.py ou analyze_video.py
model_path = "custom-trained"
```

### Guide Complet
Consultez `DOCKER_GUIDE.md` pour un guide détaillé sur l'entraînement de modèles personnalisés.

## 📁 Structure du projet

```
YOLO_Object_Detection/
├── venv/                    # Environnement virtuel Python
├── requirements.txt         # Dépendances Python
├── install.sh               # Script d'installation automatique
├── analyze_image.py         # Script d'analyse d'images multi-datasets
├── analyze_video.py         # Script d'analyse de vidéos multi-datasets
├── Dockerfile.arm64         # Image Docker pour Mac Apple Silicon
├── Dockerfile.amd64         # Image Docker pour PC (CPU)
├── Dockerfile.gpu           # Image Docker pour PC avec GPU NVIDIA
├── DOCKER_GUIDE.md          # Guide d'entraînement avec Docker
├── datasets/                # Datasets pour entraînement
├── runs/                    # Résultats d'entraînement
├── Images/                  # Images de test
├── Videos/                  # Video de test
└── README.md                # Ce fichier

```

**Fichiers générés automatiquement :**
- `venv/` - Environnement virtuel créé lors de l'installation
- `yolov8*.pt` - Modèles YOLOv8 téléchargés automatiquement (~50-200MB chacun)
- `output.png` - sauvegarde de l'image analysée 
- `runs/train/weights/` - Modèles entraînés personnalisés

## 📖 Ressources supplémentaires

- [Documentation Ultralytics](https://docs.ultralytics.com/)
- [Dataset COCO](https://cocodataset.org/)
- [Dataset Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
- [Roboflow Universe](https://universe.roboflow.com/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
- [Docker Documentation](https://docs.docker.com/)

## 📄 Licence

Ce projet utilise la bibliothèque Ultralytics (licence AGPL-3.0) et PyTorch.