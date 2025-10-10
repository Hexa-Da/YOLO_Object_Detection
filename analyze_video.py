#!/usr/bin/env python3
"""
Script d'analyse de vidéos utilisant YOLOv8 pré-entraîné sur différents datasets
Supporte COCO, Open Images V7, et modèles personnalisés
"""

import cv2 # pour la lecture des vidéos
import torch # pour le deep learning
from ultralytics import YOLO # pour le modèle YOLOv8
import matplotlib.pyplot as plt # pour la visualisation
import matplotlib.patches as patches # pour les boîtes de détection
from PIL import Image # pour la manipulation d'images
import numpy as np # pour les calculs numériques
import os # pour le système de fichiers
import time # pour mesurer les performances

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
    },
    'custom-trained': {
        'name': 'Modèle personnalisé',
        'full_name': 'Modèle personnalisé',
        'classes': None,  # 4 classes
        'num_classes': 4,
        'website': 'https://github.com/ultralytics/ultralytics',
        'description': 'Modèle YOLOv8 fine-tuné sur dataset Aerial Cars'
    }
}

def detect_dataset_type(model_path):
    """
    Détecte le type de dataset basé sur le nom du modèle
    
    Args:
        model_path (str): Chemin vers le modèle
        
    Returns:
        str: Type de dataset ('coco', 'oiv7', 'seg', 'custom-trained')
    """
    if 'best.pt' in model_path or 'last.pt' in model_path:
        return 'custom-trained'
    elif 'seg' in model_path.lower():
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

def analyze_video(video_path, model_path, seuil_conf):
    """
    Analyse une vidéo avec YOLOv8 pré-entraîné sur différents datasets
    
    Args:
        video_path (str): Chemin vers la vidéo à analyser
        model_path (str): Chemin vers le modèle YOLOv8
        seuil_conf (float): Seuil de confiance minimum
    """
    
    # Détecter le type de dataset
    dataset_type = detect_dataset_type(model_path)
    if dataset_type == 'custom-trained':
        dataset_info = DATASET_INFO[dataset_type]
    else:
        dataset_info = DATASET_INFO[dataset_type[2:]]
    
    # Vérifier si la vidéo existe
    try:
        cap = cv2.VideoCapture(video_path)
        if cap is None:
            print(f"❌ Erreur: Impossible de charger la vidéo {video_path}")
            return
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la vidéo: {e}")
        cap.release()
        return
    
    # Récupérer le modèle (depuis le cache ou le charger)
    try:
        model = get_model(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        cap.release()
        return
    
    print(f"\n🎬 ANALYSE VIDÉO YOLOV8 + DATASET {dataset_info['name'].upper()}")
    print("=" * 50)

    # Obtenir les propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"🎥 Propriétés de la vidéo:")
    print(f"   📏 Résolution: {width}x{height}")
    print(f"   ⏱️  FPS: {fps}")
    print(f"   📊 Frames totales: {total_frames}")
    print(f"   ⏰ Durée: {duration:.1f} secondes")
    print(f"   🎯 Seuil de confiance: {seuil_conf}")
    print()
    
    # Statistiques
    frame_count = 0
    total_detections = 0
    detection_history = []
    start_time = time.time()
    
    # Couleurs pour les boîtes de détection et masques
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print("🚀 Début de l'analyse...")
    
    try:
        # Créer une fenêtre pour afficher la vidéo
        cv2.namedWindow("Analyse Video YOLOv8", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Analyse Video YOLOv8", 2560, 1440)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Afficher le progrès
            if frame_count % 30 == 0:  # Toutes les 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progrès: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Effectuer la détection sur la frame
            try:
                results = model(frame, conf=seuil_conf)
                result = results[0]
                
                # Compter les détections
                frame_detections = 0
                if result.boxes is not None and result.masks is None:
                    frame_detections = len(result.boxes)
                    total_detections += frame_detections
                
                detection_history.append(frame_detections)
                
                # Dessiner les détections sur la frame
                annotated_frame = frame.copy()
                
                # Dessiner les masques s'ils existent
                if result.masks is not None:
                    for i, mask in enumerate(result.masks):
                        try:
                            mask_array = mask.data[0].cpu().numpy()
                            
                            # Vérifications de sécurité
                            if mask_array.size == 0 or mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
                                continue
                            
                            # Redimensionner le masque seulement si nécessaire
                            if mask_array.shape != (height, width):
                                if mask_array.shape[0] > 0 and mask_array.shape[1] > 0:
                                    mask_resized = cv2.resize(mask_array.astype(np.uint8), 
                                                            (width, height))
                                else:
                                    continue
                            else:
                                mask_resized = mask_array.astype(np.uint8)
                            
                            # Créer un masque coloré
                            color = colors[i % len(colors)]
                            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                            colored_mask[mask_resized == 1] = [int(c * 255) for c in color[:3]]
                            
                            # Dessiner uniquement les contours du masque
                            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(annotated_frame, contours, -1, [int(c * 255) for c in color[:3]], 2)

                            # Calculer le centre du contour pour placer le label
                            for contour in contours:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # CORRECTION : Utiliser result.boxes pour obtenir la classe et la confiance
                                    if result.boxes is not None and i < len(result.boxes):
                                        box = result.boxes[i]
                                        class_id = int(box.cls[0].cpu().numpy())
                                        class_name = result.names[class_id]
                                        confidence = box.conf[0].cpu().numpy()
                                        
                                        # Ajouter le label avec fond coloré
                                        label = f"{class_name} {confidence:.2f}"
                                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        
                                        # Dessiner le fond du label
                                        cv2.rectangle(annotated_frame, 
                                                    (cx - label_w//2, cy - label_h - 10),
                                                    (cx + label_w//2, cy), 
                                                    [int(c * 255) for c in color[:3]], 
                                                    -1)
                                        
                                        # Dessiner le texte
                                        cv2.putText(annotated_frame, 
                                                  label, 
                                                  (cx - label_w//2, cy - 5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.6, 
                                                  (255, 255, 255), 
                                                  2)
                            
                        except Exception as e:
                            print(f"⚠️  Erreur masque {i}: {e}")
                            continue
                
                # Dessiner les boîtes de détection
                if result.boxes is not None and result.masks is None:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # Dessiner la boîte
                        color = colors[i % len(colors)]
                        color_bgr = [int(c * 255) for c in color[:3]]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
                        
                        # Ajouter le label avec fond coloré
                        label = f"{class_name} {confidence:.2f}"
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1-label_h-10), (x1+label_w, y1), color_bgr, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Ajouter des informations sur la frame avec fond coloré
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections}"
                (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (5, 5), (15+text_w, 35+text_h), (0, 100, 0), -1)
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Redimensionner pour un affichage plus grand
                max_width = 2560
                max_height = 1440
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
                else:
                    # Agrandir la vidéo si elle est plus petite que la taille maximale
                    scale = min(max_width/width, max_height/height)
                    if scale > 1:
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
                
                # Afficher la frame
                cv2.imshow('Analyse Video YOLOv8', annotated_frame)
                
                # Attendre 1ms et vérifier si l'utilisateur veut quitter (touche 'q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Analyse interrompue par l'utilisateur")
                    break
                
            except Exception as e:
                print(f"❌ Erreur lors de l'analyse de la frame {frame_count}: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n⏹️  Analyse interrompue par l'utilisateur")
    
    finally:
        # Nettoyer
        cap.release()       
        
        # Calculer les statistiques finales
        end_time = time.time()
        processing_time = end_time - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"\n📊 RÉSULTATS DE L'ANALYSE")
        print("=" * 50)
        print(f"🎬 Frames traitées: {frame_count}")
        print(f"🔍 Total détections: {total_detections}")
        print(f"📈 Moyenne détections/frame: {total_detections/frame_count:.2f}")
        print(f"⏱️  Temps de traitement: {processing_time:.1f} secondes")
        print(f"⚡ FPS de traitement: {avg_fps:.1f}")
        print(f"🎯 FPS vidéo original: {fps}")
        
        print(f"\n🏁 Analyse terminée!")
        print()

def show_dataset_info(dataset_type):
    """
    Affiche les informations sur le dataset utilisé
    
    Args:
        dataset_type (str): Type de dataset ('coco', 'oiv7', 'seg')
    """

    # Gestion du modèle personnalisé
    if dataset_type == 'custom-trained':
        print()
        print(f"📊 INFORMATIONS MODÈLE PERSONNALISÉ")
        print("=" * 50)
        print(f"📚 Dataset: Modèle entraîné personnalisé")
        print(f"🔢 Nombre de classes: 4")
        print(f"🌐 Type: Détection de véhicules aériens")
        print(f"📝 Description: Modèle YOLOv8 fine-tuné sur dataset Aerial Cars")
        print()
        print("🏷️ Classes détectées:")
        print("   🚗 car")
        print("   🚛 truck") 
        print("   🚌 bus")
        print("   🚐 van")
        print()
        return

    # Gestion des datasets pré-entraînés
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
        print("   💡 Idéal pour l'édition de vidéos et la robotique")
    print()

def main():
    """Fonction principale"""    
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

        # Modèles personnalisés
        'custom-trained': "runs/train6/weights/best.pt",
    }
    
    # Choisir le dataset à utiliser
    dataset_choice = 'custom-trained' 
    model_path = model_paths[dataset_choice]

    # Configuration - Changez ces valeurs pour tester différents datasets
    video_path = "Videos/video2.mp4"

    # Choisir le seuil de confiance
    seuil_conf = 0.8
    
    # Afficher les informations sur le dataset
    show_dataset_info(dataset_choice)
    
    # Lancer l'analyse
    analyze_video(video_path, model_path, seuil_conf)

if __name__ == "__main__":
    main()
