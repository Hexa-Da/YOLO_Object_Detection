#!/usr/bin/env python3
"""
Script d'analyse en temps réel (LIVE) utilisant YOLOv8 sur flux caméra
Supporte COCO, Open Images V7, et modèles personnalisés
Utilise la webcam ou toute autre caméra connectée
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
    'custom-trained-aerial-cars': {
        'name': 'Modèle personnalisé',
        'full_name': 'Modèle personnalisé',
        'classes': None,  # 4 classes
        'num_classes': 4,
        'website': 'https://github.com/ultralytics/ultralytics',
        'description': 'Modèle YOLOv8 fine-tuné sur dataset Aerial Cars'
    },
    'custom-trained-traffic-watch': {
        'name': 'Modèle personnalisé',
        'full_name': 'Modèle personnalisé',
        'classes': None,  # 14 classes
        'num_classes': 14,
        'website': 'https://github.com/ultralytics/ultralytics',
        'description': 'Modèle YOLOv8 fine-tuné sur dataset Traffic Watch'
    }
}

def detect_dataset_type(model_path):
    """
    Détecte le type de dataset basé sur le nom du modèle
    
    Args:
        model_path (str): Chemin vers le modèle
        
    Returns:
        str: Type de dataset ('coco', 'oiv7', 'seg', 'custom-trained-aerial-cars', 'custom-trained-traffic-watch')
    """
    if 'train6' in model_path:
        return 'custom-trained-aerial-cars'
    elif 'train1' in model_path:
        return 'custom-trained-traffic-watch'
    elif '11' in model_path:
        if model_path[6] == 'n':
            return 'nv11-coco'
        elif model_path[6] == 's':
            return 'sv11-coco'
        elif model_path[6] == 'm':
            return 'mv11-coco'
        elif model_path[6] == 'l':
            return 'lv11-coco'
        elif model_path[6] == 'x':
            return 'xv11-coco'
    elif 'seg' in model_path:
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
    elif 'oiv7' in model_path:
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

def analyze_live(camera_index, model_path, seuil_conf):
    """
    Analyse un flux vidéo en direct (webcam) avec YOLOv8 pré-entraîné sur différents datasets
    
    Args:
        camera_index (int): Index de la caméra (0 = webcam par défaut, 1, 2... pour autres caméras)
        model_path (str): Chemin vers le modèle YOLOv8
        seuil_conf (float): Seuil de confiance minimum
    """
    
    # Détecter le type de dataset
    dataset_type = detect_dataset_type(model_path)
    if 'custom-trained' in dataset_type:
        dataset_info = DATASET_INFO[dataset_type]
    else:
        dataset_info = DATASET_INFO[dataset_type.split('-')[1]]
    
    # Ouvrir la caméra
    try:
        cap = cv2.VideoCapture(camera_index)
        
        # Vérifier que la caméra est bien ouverte
        if not cap.isOpened():
            print(f"❌ Erreur: Impossible d'ouvrir la caméra {camera_index}")
            return
    except Exception as e:
        print(f"❌ Erreur lors de l'ouverture de la caméra: {e}")
        return
    
    # Récupérer le modèle (depuis le cache ou le charger)
    try:
        model = get_model(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        cap.release()
        return
    
    # Configuration optionnelle de la caméra (résolution)
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print(f"\n📹 ANALYSE EN DIRECT (LIVE) - YOLOV8 + DATASET {dataset_info['name'].upper()}")
    print("=" * 50)

    # Obtenir les propriétés de la caméra
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 Propriétés de la caméra:")
    print(f"   📷 Caméra: {camera_index}")
    print(f"   📏 Résolution: {width}x{height}")
    print(f"   ⏱️  FPS caméra: {fps if fps > 0 else 'Auto'}")
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
        # Créer une fenêtre pour afficher le flux live
        cv2.namedWindow("Analyse LIVE YOLOv8", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Analyse LIVE YOLOv8", width, height)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur de lecture de la caméra")
                break
            
            frame_count += 1
            
            # Afficher les statistiques en temps réel
            if frame_count % 30 == 0:  # Toutes les 30 frames
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"⏱️  Temps: {elapsed_time:.1f}s | FPS: {current_fps:.1f} | Frames: {frame_count} | Détections: {total_detections}")
            
            # Effectuer la détection sur la frame
            try:
                results = model(frame, conf=seuil_conf)
                result = results[0]
                
                # Compter les détections
                frame_detections = 0
                if result.boxes is not None:
                    frame_detections = len(result.boxes)
                    total_detections += frame_detections
                elif result.masks is not None:
                    frame_detections = len(result.maskss)
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
                elapsed = time.time() - start_time
                fps_live = frame_count / elapsed if elapsed > 0 else 0
                info_text = f"LIVE | FPS: {fps_live:.1f} | Detections: {frame_detections}"
                (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (5, 5), (15+text_w, 35+text_h), (0, 0, 200), -1)
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
                cv2.imshow('Analyse LIVE YOLOv8', annotated_frame)
                
                # Attendre 1ms et vérifier si l'utilisateur veut quitter (touche 'q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Analyse arrêtée par l'utilisateur")
                    break
                
            except Exception as e:
                print(f"❌ Erreur lors de l'analyse de la frame {frame_count}: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n⏹️  Analyse interrompue par l'utilisateur")
    
    finally:
        # Nettoyer
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculer les statistiques finales
        end_time = time.time()
        processing_time = end_time - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"\n📊 RÉSULTATS DE LA SESSION LIVE")
        print("=" * 50)
        print(f"🎬 Frames traitées: {frame_count}")
        print(f"🔍 Total détections: {total_detections}")
        if frame_count > 0:
            print(f"📈 Moyenne détections/frame: {total_detections/frame_count:.2f}")
        print(f"⏱️  Durée de la session: {processing_time:.1f} secondes")
        print(f"⚡ FPS moyen de traitement: {avg_fps:.1f}")
        if processing_time > 0:
            print(f"📊 Détections/seconde: {total_detections/processing_time:.2f}")
        
        print(f"\n🏁 Session terminée!")
        print()

def show_dataset_info(dataset_type):
    """
    Affiche les informations sur le dataset utilisé
    
    Args:
        dataset_type (str): Type de dataset ('coco', 'oiv7', 'seg', 'custom-trained-aerial-cars', 'custom-trained-traffic-watch')
    """

    # Gestion du modèle personnalisé
    if 'custom-trained-aerial-cars' in dataset_type:
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

    if 'custom-trained-traffic-watch' in dataset_type:
        print()
        print(f"📊 INFORMATIONS MODÈLE PERSONNALISÉ")
        print("=" * 50)
        print(f"📚 Dataset: Modèle entraîné personnalisé")
        print(f"🔢 Nombre de classes: 14")
        print(f"🌐 Type: Détection de véhicules terrestres")
        print(f"📝 Description: Modèle YOLOv8 fine-tuné sur dataset Traffic Watch")
        print()
        print("🏷️ Classes détectées:")
        print("    car")
        print("    pedestrian")
        print("    motorcyclist")
        print("    traffic-lights")
        print("    bus")
        print("    minivan")
        print("    lorry")
        print("    person")
        print("    motorcycle")
        print("    cyclist")
        print("    road-signs")
        print("    bicycle")
        print("    minibus")
        print("    road-vendor")
        print()
        return

    # Gestion des datasets pré-entraînés
    dataset_info = DATASET_INFO[dataset_type.split('-')[1]]
    
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
        'custom-trained-aerial-cars': "runs/train6/weights/best.pt",
        'custom-trained-traffic-watch': "runs/train12/weights/best.pt",

        # Modèles YOLOv11
        'nv11-coco': "yolo11n.pt",
        'sv11-coco': "yolo11s.pt",
        'mv11-coco': "yolo11m.pt",
        'lv11-coco': "yolo11l.pt",
        'xv11-coco': "yolo11x.pt",
    }
    
    # Configuration
    # Index de la caméra: 0 = webcam par défaut, 1, 2... pour d'autres caméras
    camera_index = 0
    
    # Choisir le modèle à utiliser 
    dataset_choice = 'n-seg'  # Modèle léger pour temps réel
    model_path = model_paths[dataset_choice]

    # Choisir le seuil de confiance 
    seuil_conf = 0.5
    
    # Afficher les informations sur le dataset
    show_dataset_info(dataset_choice)
    
    # Lancer l'analyse en direct
    analyze_live(camera_index, model_path, seuil_conf)

if __name__ == "__main__":
    main()
