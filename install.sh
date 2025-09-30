#!/bin/bash

echo "🚀 Installation du projet YOLOv8 Object Detection"
echo "=================================================="

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

echo "✅ Python3 trouvé: $(python3 --version)"

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

echo "✅ pip3 trouvé"

# Créer un environnement virtuel (optionnel)
read -p "Voulez-vous créer un environnement virtuel? (y/N): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
fi

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dépendances installées avec succès"
else
    echo "❌ Erreur lors de l'installation des dépendances"
    exit 1
fi

# Créer le dossier examples
mkdir -p examples
echo "✅ Dossier examples créé"

# Rendre les scripts exécutables
chmod +x *.py
echo "✅ Scripts rendus exécutables"

echo ""
echo "🎉 Installation terminée avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Testez l'installation: python3 test_detection.py"
echo "2. Lancez l'application web: python3 object_detector.py"
echo "3. Consultez le README.md pour plus d'informations"
echo ""
echo "🌐 L'application web sera disponible à: http://localhost:8080"
