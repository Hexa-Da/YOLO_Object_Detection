#!/usr/bin/env python3
"""
Script d'analyse d'images utilisant YOLOv8 pré-entraîné sur différents datasets
Supporte COCO, Open Images V7, et modèles personnalisés
"""

import cv2 # pour la lecture des images
import torch # pour le deep learning
from ultralytics import YOLO # pour le modèle YOLOv8 # pour la segmentation des images
import matplotlib.pyplot as plt # pour la visualisation des images
import matplotlib.patches as patches # pour les boîtes de détection
from PIL import Image # pour la manipulation des images
import numpy as np # pour les calculs numériques

_model_cache = {}

# Informations sur les datasets supportés
DATASET_INFO = {
    'coco': {
        'name': 'COCO',
        'full_name': 'Common Objects in Context',
        'classes': None, # 80 classes
        'num_classes': 80,
        'website': 'https://cocodataset.org/',
        'description': 'Dataset généraliste avec 80 classes d\'objets communs'
    },
    'oiv7': {
        'name': 'Open Images V7',
        'full_name': 'Open Images Dataset V7',
        'classes': None,  # 600 classes
        'num_classes': 600,
        'website': 'https://storage.googleapis.com/openimages/web/index.html',
        'description': 'Dataset large avec 600 classes d\'objets variés'
    },
    'seg': {
        'name': 'Segmentation',
        'full_name': 'YOLOv8 Segmentation (COCO)',
        'classes': None,  # 80 classes COCO
        'num_classes': 80,
        'website': 'https://cocodataset.org/',
        'description': 'Segmentation d\'objets avec masques de pixels précis'
    }
}

def detect_dataset_type(model_path):
    """
    Détecte le type de dataset basé sur le nom du modèle
    
    Args:
        model_path (str): Chemin vers le modèle
        
    Returns:
        str: Type de dataset ('coco', 'oiv7', 'seg')
    """
    if 'seg' in model_path.lower():
        if model_path[6] == 'n':
            return 'n-seg'
        elif model_path[6] == 's':
            return 's-seg'
        elif model_path[6] == 'm':
            return 'm-seg'
        elif model_path[6] == 'l':
            return 'l-seg'
        elif model_path[6] == 'x':
            return 'x-seg'
    elif 'oiv7' in model_path.lower():
        if model_path[6] == 'n':
            return 'n-oiv7'
        elif model_path[6] == 's':
            return 's-oiv7'
        elif model_path[6] == 'm':
            return 'm-oiv7'
        elif model_path[6] == 'l':
            return 'l-oiv7'
        elif model_path[6] == 'x':
            return 'x-oiv7'
    else:
        if model_path[6] == 'n':
            return 'n-coco'
        elif model_path[6] == 's':
            return 's-coco'
        elif model_path[6] == 'm':
            return 'm-coco'
        elif model_path[6] == 'l':
            return 'l-coco'
        elif model_path[6] == 'x':
            return 'x-coco'

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

def analyze_image(image_path, model_path, seuil_conf):
    """
    Analyse une image avec YOLOv8 pré-entraîné sur différents datasets
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        model_path (str): Chemin vers le modèle YOLOv8
    """
    
    # Détecter le type de dataset
    dataset_type = detect_dataset_type(model_path)
    dataset_info = DATASET_INFO[dataset_type[2:]]
    
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
        results = model(image_path,conf=seuil_conf)
        result = results[0]
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        return
    
    # Afficher les résultats de détection
    print()
    print(f"🎯 DÉTECTION PYTORCH + DATASET {dataset_info['name'].upper()}")
    print("=" * 50)
    print()
    
    if (result.boxes is not None and len(result.boxes) > 0) or (result.masks is not None and len(result.masks) > 0):
        print(f"📊 Nombre d'objets détectés: {len(result.boxes) if result.boxes is not None else 0} (Seuil de confiance: {seuil_conf})")
        print()
        
        # Créer une figure pour afficher l'image avec les détections
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Image avec détections
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(f"Objets détectés ({dataset_type})", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Couleurs pour les boîtes de détection et masques
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Afficher d'abord les masques s'ils existent
        if result.masks is not None:
            
            for i, mask in enumerate(result.masks):
                # Convertir le masque en numpy array
                mask_array = mask.data[0].cpu().numpy()
                
                # Redimensionner le masque pour qu'il corresponde exactement à l'image
                mask_resized = cv2.resize(mask_array.astype(np.uint8), 
                                        (image_rgb.shape[1], image_rgb.shape[0]))
                
                # Dessiner uniquement les contours du masque
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convertir les contours pour matplotlib
                for contour in contours:
                    # Convertir les coordonnées OpenCV vers matplotlib
                    contour_points = contour.reshape(-1, 2)
                    
                    # Dessiner le contour
                    color = colors[i % len(colors)]
                    ax.plot(contour_points[:, 0], contour_points[:, 1], 
                           color=color, linewidth=3, alpha=0.8)
                
                # Calculer le centre du masque pour placer le label
                if len(contours) > 0:
                    # Utiliser le plus grand contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Obtenir les informations de la boîte correspondante
                        if result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names[class_id]
                            
                            # Ajouter le label au centre
                            color = colors[i % len(colors)]
                            ax.text(
                                cx, cy, f"{class_name} {confidence:.2%}",
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                                color='white', ha='center', va='center'
                            )
        
        # Ensuite afficher les boîtes et labels (seulement pour les modèles sans segmentation)
        if result.boxes is not None and result.masks is None:
            for i, box in enumerate(result.boxes):
                # Coordonnées de la boîte
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Informations de l'objet
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Afficher les informations
                print(f"🔸 Objet {i+1}:")
                print(f"   📝 Nom: {class_name}")
                print(f"   🆔 ID: {class_id}")
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
        
        # Afficher l'image originale même si aucun objet n'est détecté
        plt.figure(figsize=(10, 8))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(f"Image originale - Aucun objet détecté (conf ≥ {seuil_conf})", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
    
    print(f"\n🏁 Analyse terminée!")
    print()

def show_dataset_info(dataset_type):
    """
    Affiche les informations sur le dataset utilisé
    
    Args:
        dataset_type (str): Type de dataset ('coco', 'oiv7', 'seg')
    """
    dataset_info = DATASET_INFO[dataset_type[2:]]
    
    print()
    print(f"📊 INFORMATIONS DATASET {dataset_info['name'].upper()}")
    print("=" * 50)
    print(f"📚 Dataset: {dataset_info['full_name']}")
    print(f"🔢 Nombre de classes: {dataset_info['num_classes']}")
    print(f"🌐 Site officiel: {dataset_info['website']}")
    print(f"📝 Description: {dataset_info['description']}")
    print()
    
    if dataset_type == 'coco':
        print("🏷️ Catégories principales:")
        print("   👥 Personnes: person")
        print("   🚗 Véhicules: car, bus, truck, motorcycle, bicycle, etc.")
        print("   🐕 Animaux: cat, dog, horse, cow, sheep, bird, etc.")
        print("   🏠 Objets: chair, table, laptop, cell phone, book, etc.")
        print("   🍎 Nourriture: apple, banana, pizza, cake, etc.")
        print("   ⚽ Sports: sports ball, tennis racket, baseball bat, etc.")
    elif dataset_type == 'oiv7':
        print("🏷️ Catégories principales:")
        print("   👥 Personnes et parties du corps")
        print("   🚗 Véhicules de tous types")
        print("   🐕 Animaux domestiques et sauvages")
        print("   🏠 Objets du quotidien")
        print("   🍎 Nourriture et boissons")
        print("   ⚽ Sports et loisirs")
        print("   🎨 Art et culture")
        print("   🌍 Nature et environnement")
    elif dataset_type == 'seg':
        print("🎨 Fonctionnalités de segmentation:")
        print("   🔍 Détection d'objets + masques de pixels")
        print("   📐 Contours précis des objets")
        print("   🎯 80 classes COCO avec segmentation")
        print("   💡 Idéal pour l'édition d'images et la robotique")
    print()

def main():
    """Fonction principale"""
    # Configuration - Changez ces valeurs pour tester différents datasets
    image_path = "test.jpg"
    
    # Exemples de modèles pour différents datasets :
    model_paths = {
        # Modèles de DÉTECTION
        'n-coco': "yolov8n.pt",           
        's-coco': "yolov8s.pt",           
        'm-coco': "yolov8m.pt",           
        'l-coco': "yolov8l.pt",           
        'x-coco': "yolov8x.pt",           
        
        # Modèles de SEGMENTATION
        'n-seg': "yolov8n-seg.pt",
        's-seg': "yolov8s-seg.pt", 
        'm-seg': "yolov8m-seg.pt",
        'l-seg': "yolov8l-seg.pt",
        'x-seg': "yolov8x-seg.pt",
        
        # Modèles Open Images V7
        'n-oiv7': "yolov8n-oiv7.pt",      
        's-oiv7': "yolov8s-oiv7.pt",      
        'm-oiv7': "yolov8m-oiv7.pt",      
        'l-oiv7': "yolov8l-oiv7.pt",      
        'x-oiv7': "yolov8x-oiv7.pt",      
    }
    
    # Choisir le dataset à utiliser
    dataset_choice = 'x-seg' 
    model_path = model_paths[dataset_choice]

    # Choisir le seuil de confiance
    seuil_conf = 0.25
    
    # Afficher les informations sur le dataset
    show_dataset_info(dataset_choice)
    
    # Lancer l'analyse
    analyze_image(image_path, model_path, seuil_conf)

if __name__ == "__main__":
    main()
