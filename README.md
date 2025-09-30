# 🔍 Analyseur d'images YOLOv8 + PyTorch + Dataset COCO

Ce projet analyse l'image `objet-pub.png` en utilisant YOLOv8 pré-entraîné sur le dataset COCO avec PyTorch pour détecter et identifier les objets selon les 80 classes du dataset COCO.

## 📋 Table des matières

1. [Installation](#installation)
2. [Pourquoi un environnement virtuel ?](#pourquoi-un-environnement-virtuel)
3. [Utilisation](#utilisation)
4. [Dataset COCO](#dataset-coco)
5. [Comment fonctionne YOLO ?](#comment-fonctionne-yolo)
6. [Structure du projet](#structure-du-projet)
7. [Résultats d'analyse](#résultats-danalyse)

## 🚀 Installation

### Méthode 1 : Installation automatique (Recommandée)

```bash
# 1. Aller dans le dossier du projet
cd /Users/$(whoami)/Documents/PYTHON/yolov8_object_detection

# 2. Exécuter le script d'installation
chmod +x install.sh
./install.sh
```

Le script d'installation :
- ✅ Vérifie que Python 3 et pip sont installés
- 📦 Crée un environnement virtuel (recommandé)
- 📥 Installe automatiquement toutes les dépendances
- 🗂️ Crée le dossier `examples/`
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
- `ultralytics>=8.0.0` - Framework YOLOv8
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
cd /Users/$(whoami)/Documents/PYTHON/yolov8_object_detection
source venv/bin/activate
```

**Comment savoir si l'environnement est activé :**
- Le prompt affiche `(venv)` au début
- `which python` pointe vers le dossier `venv/bin/python`


## 🎯 Utilisation

### Analyser l'image example.jpg

```bash
# 1. Activer l'environnement virtuel
source venv/bin/activate

# 2. Lancer l'analyse
python analyze_image.py
```

### Ce que fait le script

1. **📊 Affiche les informations** sur le dataset COCO (80 classes, 1.5M objets)
2. **📥 Charge l'image** `example.jpg`
3. **🤖 Télécharge le modèle** YOLOv8 pré-entraîné sur COCO (yolov8m.pt - ~50MB)
4. **🔍 Analyse l'image** avec PyTorch et YOLOv8 selon les classes COCO
5. **📊 Affiche les résultats** : objets détectés + ID COCO + confiance + position
6. **🖼️ Crée une visualisation** avec des boîtes de détection colorées
7. **💾 Sauvegarde** l'image annotée (`detection.png`)


## 📊 Dataset COCO

### Qu'est-ce que le dataset COCO ?

**COCO** (Common Objects in Context) est l'un des datasets de référence pour la détection d'objets, la segmentation et la génération de légendes d'images.

### Caractéristiques du dataset COCO

- **🔢 80 classes d'objets** différentes
- **📈 118,287 images d'entraînement**
- **📈 5,000 images de validation**
- **📈 40,670 images de test**
- **🎯 1.5 million d'objets annotés**
- **🌐 Site officiel** : https://cocodataset.org/

### Classes détectables par le modèle

Le modèle YOLOv8 pré-entraîné peut détecter ces **80 classes** :

#### 👥 Personnes
- `person` (ID: 0)

#### 🚗 Véhicules
- `bicycle` (ID: 1), `car` (ID: 2), `motorcycle` (ID: 3), `airplane` (ID: 4)
- `bus` (ID: 5), `train` (ID: 6), `truck` (ID: 7), `boat` (ID: 8)

#### 🐕 Animaux
- `bird` (ID: 14), `cat` (ID: 15), `dog` (ID: 16), `horse` (ID: 17)
- `sheep` (ID: 18), `cow` (ID: 19), `elephant` (ID: 20), `bear` (ID: 21)
- `zebra` (ID: 22), `giraffe` (ID: 23)

#### 🏠 Objets du quotidien
- `chair` (ID: 56), `couch` (ID: 57), `potted plant` (ID: 58), `bed` (ID: 59)
- `dining table` (ID: 60), `toilet` (ID: 61), `tv` (ID: 62), `laptop` (ID: 63)
- `mouse` (ID: 64), `remote` (ID: 65), `keyboard` (ID: 66), `cell phone` (ID: 67)

#### 🍎 Nourriture
- `banana` (ID: 46), `apple` (ID: 47), `sandwich` (ID: 48), `orange` (ID: 49)
- `broccoli` (ID: 50), `carrot` (ID: 51), `hot dog` (ID: 52), `pizza` (ID: 53)
- `donut` (ID: 54), `cake` (ID: 55)

#### ⚽ Sports et loisirs
- `sports ball` (ID: 32), `kite` (ID: 33), `baseball bat` (ID: 34)
- `baseball glove` (ID: 35), `skateboard` (ID: 36), `surfboard` (ID: 37)
- `tennis racket` (ID: 38)

### Pourquoi utiliser COCO ?

1. **🎯 Standard de référence** : Utilisé par la communauté scientifique
2. **📊 Diversité** : Couvre de nombreux domaines d'application
3. **🔍 Qualité** : Annotations précises et vérifiées
4. **🌍 Échelle** : Plus de 1.5 million d'objets annotés
5. **🔄 Mise à jour** : Dataset régulièrement mis à jour


## 🧠 Comment fonctionne YOLO ?

### Qu'est-ce que YOLO ?

**YOLO** (You Only Look Once) est un réseau de neurones convolutifs spécialisé dans la **détection d'objets en temps réel**.

### Principe de fonctionnement

1. **🔍 Analyse globale** : YOLO examine l'image entière en une seule fois
2. **📦 Détection simultanée** : Trouve tous les objets en parallèle
3. **🎯 Prédiction directe** : Prédit directement les boîtes de détection
4. **⚡ Vitesse** : Beaucoup plus rapide que les méthodes traditionnelles

### Architecture YOLOv8

```
Image (640x640) → Backbone → Neck → Head → Détections
     ↓              ↓        ↓      ↓         ↓
   Pixels      Caractéristiques  Fusion  Prédictions  [x,y,w,h,conf,class]
```

### Processus de détection

1. **📥 Entrée** : Image redimensionnée à 640x640 pixels
2. **🧠 Traitement** : Le réseau extrait des caractéristiques
3. **🎯 Prédiction** : Pour chaque zone, prédit :
   - **Position** : Coordonnées de la boîte (x1, y1, x2, y2)
   - **Confiance** : Probabilité que ce soit un objet (0-1)
   - **Classe** : Type d'objet (person, car, dog, etc.)
4. **🔍 Filtrage** : Garde seulement les détections avec confiance > seuil
5. **📊 Résultat** : Liste des objets détectés avec leurs propriétés

### Modèles YOLOv8 disponibles

| Taille | Fichier | Vitesse | Précision | Usage |
|--------|---------|---------|-----------|-------|
| Nano   | yolov8n.pt | ⚡⚡⚡ | ⭐⭐ | Mobile, IoT |
| Small  | yolov8s.pt | ⚡⚡ | ⭐⭐⭐ | Équilibre |
| **Medium** | **yolov8m.pt** | ⚡ | ⭐⭐⭐⭐ | **Recommandé** |
| Large  | yolov8l.pt | 🐌 | ⭐⭐⭐⭐⭐ | Haute précision |
| Huge   | yolov8x.pt | 🐌🐌 | ⭐⭐⭐⭐⭐ | Recherche |

### Classes détectables

Le modèle pré-entraîné peut détecter **80 classes** du dataset COCO (voir section [Dataset COCO](#dataset-coco) pour la liste complète) :

- **👥 Personnes** : person (ID: 0)
- **🐕 Animaux** : cat (ID: 15), dog (ID: 16), horse (ID: 17), cow (ID: 19), sheep (ID: 18), etc.
- **🚗 Véhicules** : car (ID: 2), bus (ID: 5), truck (ID: 7), motorcycle (ID: 3), bicycle (ID: 1), etc.
- **🏠 Objets** : chair (ID: 56), dining table (ID: 60), laptop (ID: 63), cell phone (ID: 67), book, etc.
- **🍎 Nourriture** : apple (ID: 47), banana (ID: 46), pizza (ID: 53), cake (ID: 55), etc.
- **⚽ Sports** : sports ball (ID: 32), tennis racket (ID: 38), baseball bat (ID: 34), etc.

## 📁 Structure du projet

```
yolov8_object_detection/
├── venv/                    # Environnement virtuel Python
├── requirements.txt         # Dépendances Python
├── install.sh               # Script d'installation automatique
├── analyze_image.py         # Script principal d'analyse
├── example.jpg              # Image à analyser
├── README.md                # Ce fichier
├── examples/                # Dossier pour les images d'exemple
├── detection.png            # Image annotée (générée automatiquement)
└── yolov8m.pt               # Modèle YOLOv8 (téléchargé automatiquement)
```

**Fichiers générés automatiquement :**
- `venv/` - Environnement virtuel créé lors de l'installation
- `detection.png` - Image annotée générée par l'analyse
- `yolov8m.pt` - Modèle YOLOv8 téléchargé automatiquement (~50MB)





### Visualisation

Le script génère :
1. **Image originale** : L'image `objet-pub.png` telle quelle
2. **Image annotée** : Avec des boîtes colorées autour des objets détectés
3. **Image sauvegardée** : `detection_objet-pub.png` pour référence

## 🔧 Personnalisation

### Changer le modèle

Modifiez la ligne dans `analyze_image.py` :

```python
# Modèle plus rapide (moins précis)
model = YOLO("yolov8s.pt")

# Modèle plus précis (plus lent)
model = YOLO("yolov8l.pt")
```

### Analyser une autre image

Modifiez la ligne dans `analyze_image.py` :

```python
image_path = "votre_image.jpg"  # Changez ici
```

## 🚨 Dépannage

### Erreur "externally-managed-environment"

**Problème :** Votre système Python est géré par Homebrew.

**Solution :** Utilisez l'environnement virtuel (voir section Installation)

### Erreur "ModuleNotFoundError: No module named 'ultralytics'"

**Problème :** L'environnement virtuel n'est pas activé.

**Solution :**
```bash
source venv/bin/activate
python analyze_image.py
```

### Erreur "command not found: python"

**Problème :** Python n'est pas dans le PATH.

**Solution :**
```bash
python3 analyze_image.py
```

## 📖 Ressources supplémentaires

- [Documentation Ultralytics](https://docs.ultralytics.com/)
- [Dataset COCO](https://cocodataset.org/)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)

## 📄 Licence

Ce projet utilise la bibliothèque Ultralytics (licence AGPL-3.0) et PyTorch.