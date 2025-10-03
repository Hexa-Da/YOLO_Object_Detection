# 🔍 Analyseur d'images YOLOv8 + PyTorch + Multi-Datasets

Ce projet analyse des images en utilisant YOLOv8 pré-entraîné sur différents datasets avec PyTorch pour détecter et identifier les objets selon les classes disponibles dans chaque dataset.

## 📋 Table des matières

1. [Installation](#installation)
2. [Pourquoi un environnement virtuel ?](#pourquoi-un-environnement-virtuel)
3. [Utilisation](#utilisation)
4. [Datasets supportés](#datasets-supportés)
5. [Comment fonctionne YOLO ?](#comment-fonctionne-yolo)
6. [Structure du projet](#structure-du-projet)
7. [Résultats d'analyse](#résultats-danalyse)
8. [Comparaison des datasets](#comparaison-des-datasets)

## 🚀 Installation

### Méthode 1 : Installation automatique (Recommandée)

```bash
# 1. Aller dans le dossier du projet
cd yolov8_object_detection

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
cd yolov8_object_detection
source venv/bin/activate
```

**Comment savoir si l'environnement est activé :**
- Le prompt affiche `(venv)` au début
- `which python` pointe vers le dossier `venv/bin/python`

## 🎯 Utilisation

### Analyser une image avec différents datasets

```bash
# 1. Activer l'environnement virtuel
source venv/bin/activate

# 2. Lancer l'analyse
python analyze_image.py
```

### Ce que fait le script

1. **📊 Affiche les informations** sur le dataset sélectionné
2. **📥 Charge l'image** spécifiée
3. **🤖 Charge le modèle** YOLOv8 pré-entraîné (avec cache pour éviter les rechargements)
4. **🔍 Analyse l'image** avec PyTorch et YOLOv8 selon les classes du dataset
5. **📊 Affiche les résultats** : objets détectés + ID + confiance + position
6. **🖼️ Crée une visualisation** avec des boîtes de détection colorées

### Configuration des datasets

Dans `analyze_image.py`, vous pouvez choisir le dataset à utiliser :

```python
# Exemples de modèles pour différents datasets :
model_paths = {
    'n-coco': "yolov8n.pt",           # COCO Nano (80 classes)
    's-coco': "yolov8s.pt",           # COCO Small (80 classes)
    'm-coco': "yolov8m.pt",           # COCO Medium (80 classes)
    'l-coco': "yolov8l.pt",           # COCO Large (80 classes)
    'x-coco': "yolov8x.pt",           # COCO XLarge (80 classes)
    'n-oiv7': "yolov8n-oiv7.pt",      # Open Images V7 Nano (600 classes)
    's-oiv7': "yolov8s-oiv7.pt",      # Open Images V7 Small (600 classes)
    'm-oiv7': "yolov8m-oiv7.pt",      # Open Images V7 Medium (600 classes)
    'l-oiv7': "yolov8l-oiv7.pt",      # Open Images V7 Large (600 classes)
    'x-oiv7': "yolov8x-oiv7.pt",      # Open Images V7 XLarge (600 classes)
}

# Choisir le dataset à utiliser
dataset_choice = 'm-coco'  # Changez ici pour tester différents datasets
```

## 📊 Datasets supportés

### 🎯 Dataset COCO (Common Objects in Context)

**Le plus populaire et performant pour la détection d'objets généraux**

#### Caractéristiques
- **🔢 80 classes d'objets** 
- **📈 118,287 images d'entraînement**
- **📈 5,000 images de validation**
- **📈 40,670 images de test**
- **🎯 1.5 million d'objets annotés**
- **🌐 Site officiel** : https://cocodataset.org/

### 🌐 Dataset Open Images V7

**Dataset très large avec de nombreuses classes spécialisées**

#### Caractéristiques
- **🔢 600 classes d'objets** très variées
- **📈 Plus de 9 millions d'images**
- **🎯 Plus de 36 millions de boîtes de détection**
- **🌐 Site officiel** : https://storage.googleapis.com/openimages/web/index.html

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
   - **Classe** : Type d'objet selon le dataset utilisé
4. **🔍 Filtrage** : Garde seulement les détections avec confiance > seuil
5. **📊 Résultat** : Liste des objets détectés avec leurs propriétés

### Modèles YOLOv8 disponibles

| Taille | Fichier COCO | Fichier OIV7 | Vitesse | Précision | Usage |
|--------|--------------|--------------|---------|-----------|-------|
| Nano   | yolov8n.pt | yolov8n-oiv7.pt | +++ | ++ | Mobile, IoT |
| Small  | yolov8s.pt | yolov8s-oiv7.pt | ++ | +++ | Équilibre |
| **Medium** | **yolov8m.pt** | **yolov8m-oiv7.pt** | + | ++++ | **Recommandé** |
| Large  | yolov8l.pt | yolov8l-oiv7.pt | - | +++++ | Haute précision |
| Huge   | yolov8x.pt | yolov8x-oiv7.pt | -- | +++++ | Recherche |

## 📁 Structure du projet

```
yolov8_object_detection/
├── venv/                    # Environnement virtuel Python
├── requirements.txt         # Dépendances Python
├── install.sh               # Script d'installation automatique
├── analyze_image.py         # Script principal d'analyse multi-datasets
├── votre_image.jpg          # Image de test
└──README.md                 # Ce fichier

```

**Fichiers générés automatiquement :**
- `venv/` - Environnement virtuel créé lors de l'installation
- `yolov8*.pt` - Modèles YOLOv8 téléchargés automatiquement (~50-200MB chacun)

### Visualisation

Le script génère une **image annotée** avec des boîtes colorées autour des objets détectés, adaptée au dataset utilisé.

## 🔧 Personnalisation

### Changer le dataset et la taille du modèle

Modifiez les lignes dans `analyze_image.py` :

```python
# Dataset COCO - Modèles plus rapides
dataset_choice = 'n-coco'  # Nano (plus rapide)
dataset_choice = 's-coco'  # Small
dataset_choice = 'm-coco'  # Medium (recommandé)
dataset_choice = 'l-coco'  # Large (plus précis)
dataset_choice = 'x-coco'  # XLarge (très précis)

# Dataset Open Images V7 - Classes spécialisées
dataset_choice = 'n-oiv7'  # Nano
dataset_choice = 'm-oiv7'  # Medium (recommandé)
dataset_choice = 'x-oiv7'  # XLarge (très précis)
```

### Analyser une autre image

Modifiez la ligne dans `analyze_image.py` :

```python
image_path = "votre_image.jpg"  # Changez ici
```

## 📊 Comparaison des datasets

### Quand utiliser COCO ?

✅ **Utilisez COCO si :**
- Vous voulez détecter des objets généraux (personnes, voitures, animaux courants)
- Vous avez besoin de la meilleure précision possible
- Vous travaillez sur des applications grand public
- Vous voulez des résultats rapides et fiables

### Quand utiliser Open Images V7 ?

✅ **Utilisez Open Images V7 si :**
- Vous avez besoin de classes très spécifiques (types de chiens, modèles de voitures)
- Vous travaillez sur des domaines spécialisés
- Vous avez besoin de détecter des objets rares
- Vous pouvez accepter une précision légèrement moindre

### Comparaison des performances

| Critère | COCO (80 classes) | Open Images V7 (600 classes) |
|---------|-------------------|------------------------------|
| **Précision générale** | +++++ | +++ |
| **Vitesse** | +++++ | ++++ |
| **Classes spécialisées** | ++ | +++++ |
| **Facilité d'utilisation** | +++++ | +++ |
| **Taille du modèle** | +++++ | +++ |

## 📖 Ressources supplémentaires

- [Documentation Ultralytics](https://docs.ultralytics.com/)
- [Dataset COCO](https://cocodataset.org/)
- [Dataset Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)

## 📄 Licence

Ce projet utilise la bibliothèque Ultralytics (licence AGPL-3.0) et PyTorch.