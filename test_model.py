from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import argparse

def test_single_image(model_path, image_path, output_dir="test_results"):
    """Tester le modèle sur une seule image"""
    
    # Charger le modèle entraîné
    model = YOLO(model_path)
    
    # Créer le dossier de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prédire sur l'image
    results = model.predict(
        source=image_path,
        save=True,
        project=output_dir,
        name='single_prediction',
        exist_ok=True,
        conf=0.5,  # seuil de confiance
        show_labels=True,
        show_conf=True
    )
    
    # Afficher les informations de détection
    for r in results:
        print(f"\n🔍 Résultats pour {image_path}:")
        print(f"Nombre de plaques détectées: {len(r.boxes)}")
        
        for i, box in enumerate(r.boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  Plaque {i+1}: Confiance={conf:.2f}, Position=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    
    print(f"✅ Résultat sauvegardé dans: {output_dir}/single_prediction/")

def test_folder(model_path, folder_path, output_dir="test_results"):
    """Tester le modèle sur un dossier d'images"""
    
    model = YOLO(model_path)
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extensions d'images supportées
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Trouver toutes les images
    image_files = []
    for ext in extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans {folder_path}")
        return
    
    print(f"🔍 Test sur {len(image_files)} images...")
    
    # Prédire sur toutes les images
    results = model.predict(
        source=folder_path,
        save=True,
        project=output_dir,
        name='folder_predictions',
        exist_ok=True,
        conf=0.5
    )
    
    # Statistiques
    total_detections = sum(len(r.boxes) for r in results)
    images_with_plates = sum(1 for r in results if len(r.boxes) > 0)
    
    print(f"\n📊 Statistiques:")
    print(f"Images testées: {len(image_files)}")
    print(f"Images avec plaques détectées: {images_with_plates}")
    print(f"Total détections: {total_detections}")
    print(f"Moyenne détections/image: {total_detections/len(image_files):.2f}")
    print(f"✅ Résultats sauvegardés dans: {output_dir}/folder_predictions/")

def test_webcam(model_path):
    """Tester le modèle en temps réel avec la webcam"""
    
    model = YOLO(model_path)
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return
    
    print("🎥 Test en temps réel - Appuyez sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prédire sur le frame
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Dessiner les résultats
        annotated_frame = results[0].plot()
        
        # Afficher
        cv2.imshow('Détection de Plaques - Appuyez sur Q pour quitter', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test webcam terminé")

def show_model_info(model_path):
    """Afficher les informations du modèle"""
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    model = YOLO(model_path)
    
    print("🤖 INFORMATIONS DU MODÈLE")
    print("=" * 40)
    print(f"Chemin: {model_path}")
    print(f"Classes: {model.names}")
    print(f"Nombre de classes: {len(model.names)}")
    
    # Taille du fichier
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Taille: {size_mb:.1f} MB")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="Tester le modèle de détection de plaques")
    parser.add_argument("--model", default="license_plate_detection/weights/best.pt", 
                       help="Chemin vers le modèle")
    parser.add_argument("--image", help="Tester sur une seule image")
    parser.add_argument("--folder", help="Tester sur un dossier d'images")
    parser.add_argument("--webcam", action="store_true", help="Test en temps réel avec webcam")
    parser.add_argument("--info", action="store_true", help="Afficher les infos du modèle")
    
    args = parser.parse_args()
    
    model_path = args.model
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("Assurez-vous d'avoir entraîné le modèle avec train_yolo.py")
        return
    
    if args.info:
        show_model_info(model_path)
    elif args.image:
        test_single_image(model_path, args.image)
    elif args.folder:
        test_folder(model_path, args.folder)
    elif args.webcam:
        test_webcam(model_path)
    else:
        # Mode interactif
        print("🧪 TESTEUR DE MODÈLE DE DÉTECTION DE PLAQUES")
        print("=" * 50)
        show_model_info(model_path)
        print("\nOptions:")
        print("1. Tester une image")
        print("2. Tester un dossier")
        print("3. Test webcam temps réel")
        print("4. Voir les résultats existants")
        
        choice = input("\nChoisissez une option (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Chemin vers l'image: ").strip()
            if image_path:
                test_single_image(model_path, image_path)
        elif choice == "2":
            folder_path = input("Chemin vers le dossier: ").strip()
            if folder_path:
                test_folder(model_path, folder_path)
        elif choice == "3":
            test_webcam(model_path)
        elif choice == "4":
            print("\n📁 Résultats existants:")
            if os.path.exists("predictions"):
                print("- predictions/ (images de validation testées)")
            if os.path.exists("test_results"):
                print("- test_results/ (vos tests personnalisés)")
            print("\nOuvrez ces dossiers pour voir les images avec détections!")

if __name__ == "__main__":
    main()
