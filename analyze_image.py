#!/usr/bin/env python3
"""
Script d'analyse d'images utilisant YOLOv8 pré-entraîné sur le dataset COCO
Détecte et identifie les objets selon les 80 classes du dataset COCO
"""

import cv2 # pour la lecture des images
import torch # pour le deep learning
from ultralytics import YOLO # pour le modèle YOLOv8
import matplotlib.pyplot as plt # pour la visualisation des images
import matplotlib.patches as patches # pour les boîtes de détection
from PIL import Image # pour la manipulation des images
import numpy as np # pour les calculs numériques

_model_cache = {}

# Classes du dataset COCO (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_model(model_path):
    """
    Récupère un modèle depuis le cache ou le charge s'il n'existe pas
    
    Args:
        model_path (str): Chemin vers le modèle YOLOv8
        
    Returns:
        YOLO: Modèle YOLOv8 chargé
    """
    if model_path not in _model_cache:
        print(f"🔄 Chargement du modèle {model_path}...")
        try:
            _model_cache[model_path] = YOLO(model_path)
            print(f"✅ Modèle {model_path} chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    else:
        print(f"♻️  Utilisation du modèle {model_path} depuis le cache")
    
    return _model_cache[model_path]

def analyze_image(image_path, model_path):
    """
    Analyse une image avec YOLOv8 pré-entraîné sur le dataset COCO
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        model_path (str): Chemin vers le modèle YOLOv8 pré-entraîné COCO
    """
    
    # Vérifier si l'image existe
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erreur: Impossible de charger l'image {image_path}")
            return
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return
    
    # Récupérer le modèle (depuis le cache ou le charger)
    try:
        model = get_model(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # Effectuer la détection
    try:
        results = model(image_path)
        result = results[0]
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        return
    
    # Afficher uniquement les résultats de détection PyTorch + COCO
    print()
    print("🎯 DÉTECTION PYTORCH + DATASET COCO")
    print("=" * 50)

    
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"📊 Nombre d'objets détectés: {len(result.boxes)}")
        print()
        
        # Créer une figure pour afficher l'image avec les détections
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Image avec détections
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title("Objets détectés", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Couleurs pour les boîtes de détection
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, box in enumerate(result.boxes):
            # Coordonnées de la boîte
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Informations de l'objet
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            # Afficher les informations avec l'ID de classe COCO
            print(f"🔸 Objet {i+1}:")
            print(f"   📝 Nom: {class_name}")
            print(f"   🆔 ID COCO: {class_id}")
            print(f"   🎯 Confiance: {confidence:.2%}")
            print(f"   📍 Position: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
            print(f"   📏 Taille: {int(x2-x1)}x{int(y2-y1)} pixels")
            print()
            
            # Dessiner la boîte de détection
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Ajouter le label
            ax.text(
                x1, y1-10, f"{class_name} {confidence:.2%}",
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                color='white'
            )
        
        # Afficher la figure
        plt.tight_layout()
        plt.show()   
        
    else:
        print("❌ Aucun objet détecté dans l'image")
    
    print(f"\n🏁 Analyse terminée!")

def show_coco_info():
    """
    Affiche les informations sur le dataset COCO utilisé
    """
    print()
    print("📊 INFORMATIONS DATASET COCO")
    print("=" * 50)
    print(f"🔢 Nombre total de classes: {len(COCO_CLASSES)}")
    print(f"📚 Dataset: Common Objects in Context (COCO)")
    print(f"🌐 Site officiel: https://cocodataset.org/")
    print(f"📈 Images d'entraînement: 118,287")
    print(f"📈 Images de validation: 5,000")
    print(f"📈 Images de test: 40,670")
    print(f"🎯 Objets annotés: 1.5 million")
    print()
    print("🏷️ Catégories principales:")
    print("   👥 Personnes: person")
    print("   🚗 Véhicules: car, bus, truck, motorcycle, bicycle, etc.")
    print("   🐕 Animaux: cat, dog, horse, cow, sheep, bird, etc.")
    print("   🏠 Objets: chair, table, laptop, cell phone, book, etc.")
    print("   🍎 Nourriture: apple, banana, pizza, cake, etc.")
    print("   ⚽ Sports: sports ball, tennis racket, baseball bat, etc.")
    print()

def main():
    """Fonction principale"""
    image_path = "example.jpg"
    model_path = "yolov8x.pt"
    
    # Afficher les informations sur le dataset COCO
    show_coco_info()
    
    # Lancer l'analyse
    analyze_image(image_path, model_path)

if __name__ == "__main__":
    main()
