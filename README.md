# Détection de Plaques d'Immatriculation avec YOLO

Ce projet permet d'entraîner un modèle d'IA pour détecter et délimiter les zones de plaques d'immatriculation sur les voitures.

## Dataset
- **Total**: 172 images (140 train, 32 validation)
- **Structure**: 
  - `dataset/images/train/` - Images d'entraînement
  - `dataset/images/val/` - Images de validation
  - `dataset/labels/train/` - Annotations d'entraînement (format YOLO)
  - `dataset/labels/val/` - Annotations de validation (format YOLO)

## Installation

```bash
pip install -r requirements.txt
```

## Étapes

### 1. Annotation des Images
**OBLIGATOIRE** - Vous devez d'abord annoter vos images :

```bash
python annotation_tool.py
```

L'outil d'annotation offre :
- Interface graphique intuitive
- Détection automatique basique
- Annotation manuelle précise
- Sauvegarde au format YOLO

**Instructions d'utilisation :**
1. Cliquez et glissez pour dessiner une boîte autour de la plaque
2. Utilisez "Auto-Détection" pour une aide automatique
3. Cliquez "Sauvegarder" pour enregistrer l'annotation
4. "Image Suivante" pour passer à la suivante

### 2. Entraînement du Modèle

```bash
python train_yolo.py
```

## Estimation du Temps d'Entraînement

**Avec GPU (RTX 3060/4060 ou équivalent) :**
- ~2-3 heures pour 100 époques

**Sans GPU (CPU seulement) :**
- ~15-20 heures pour 100 époques

**Facteurs influençant le temps :**
- Puissance du GPU/CPU
- Taille des images
- Nombre d'époques
- Taille du batch

## Architecture du Modèle

- **YOLOv8n** (version nano) - Rapide et efficace
- **1 classe** : license_plate
- **Format d'annotation** : YOLO (x_center, y_center, width, height normalisés)

## Résultats

Après l'entraînement, vous trouverez :
- `license_plate_detection/weights/best.pt` - Meilleur modèle
- `license_plate_detection/` - Métriques et graphiques
- `predictions/` - Tests sur images de validation

## Utilisation du Modèle Entraîné

```python
from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO('license_plate_detection/weights/best.pt')

# Prédire sur une nouvelle image
results = model.predict('nouvelle_image.jpg')

# Afficher les résultats
results[0].show()
```

## Conseils pour de Meilleurs Résultats

1. **Annotations précises** - La qualité des annotations est cruciale
2. **Variété des données** - Images avec différents angles, éclairages, types de plaques
3. **Augmentation des données** - YOLO applique automatiquement des augmentations
4. **Ajustement des hyperparamètres** - Modifier epochs, batch_size selon vos ressources

## Dépannage

- **Erreur de mémoire** : Réduire `batch_size` dans `train_yolo.py`
- **Entraînement lent** : Vérifier que CUDA est disponible pour GPU
- **Mauvaises détections** : Améliorer la qualité/quantité des annotations
