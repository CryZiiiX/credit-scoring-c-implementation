#!/bin/bash

# Script de téléchargement du dataset Credit Risk depuis Kaggle
# Nécessite kaggle CLI installé : pip install kaggle

echo "========================================"
echo "  TÉLÉCHARGEMENT DU DATASET"
echo "  Credit Risk Dataset (Kaggle)"
echo "========================================"
echo ""

# Vérifier que kaggle CLI est installé
if ! command -v kaggle &> /dev/null; then
    echo "❌ Erreur: kaggle CLI n'est pas installé"
    echo ""
    echo "Installation requise:"
    echo "  pip install kaggle"
    echo ""
    echo "Configuration requise:"
    echo "  1. Créer un compte Kaggle"
    echo "  2. Aller dans Account > Create New API Token"
    echo "  3. Placer le fichier kaggle.json dans ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Vérifier les credentials Kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Erreur: Credentials Kaggle non trouvés"
    echo ""
    echo "Configuration requise:"
    echo "  1. Aller sur kaggle.com > Account > Create New API Token"
    echo "  2. Placer le fichier kaggle.json dans ~/.kaggle/"
    echo "  3. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Créer le répertoire data/raw si nécessaire
mkdir -p data/raw

echo "📂 Répertoire de destination: data/raw/"
echo ""

# Télécharger le dataset
# Note: Remplacez 'laotse/credit-risk-dataset' par le bon identifiant Kaggle
echo "📥 Téléchargement du dataset depuis Kaggle..."
echo ""

# Option 1: Si vous connaissez l'identifiant exact du dataset
# kaggle datasets download -d laotse/credit-risk-dataset -p data/raw/ --unzip

# Option 2: Téléchargement manuel alternatif
echo "ℹ️  Pour télécharger le dataset manuellement:"
echo "   1. Allez sur https://www.kaggle.com/datasets (recherchez 'credit risk')"
echo "   2. Téléchargez le fichier CSV"
echo "   3. Placez-le dans data/raw/credit_risk_dataset.csv"
echo ""

# Option 3: Utiliser wget si le dataset est disponible publiquement
# URL_EXEMPLE="https://example.com/credit_risk_dataset.csv"
# wget -O data/raw/credit_risk_dataset.csv $URL_EXEMPLE

# Vérifier si le fichier existe déjà
if [ -f "data/raw/credit_risk_dataset.csv" ]; then
    echo "✓ Dataset déjà présent: data/raw/credit_risk_dataset.csv"
    
    # Afficher quelques statistiques
    echo ""
    echo "📊 Statistiques du fichier:"
    echo "   Taille: $(du -h data/raw/credit_risk_dataset.csv | cut -f1)"
    echo "   Lignes: $(wc -l < data/raw/credit_risk_dataset.csv)"
    
    # Afficher les premières lignes
    echo ""
    echo "📄 Aperçu (premières lignes):"
    head -n 3 data/raw/credit_risk_dataset.csv
    
    echo ""
    echo "✓ Dataset prêt à l'emploi!"
else
    echo "❌ Dataset non trouvé"
    echo ""
    echo "Téléchargez manuellement le dataset et placez-le dans:"
    echo "   data/raw/credit_risk_dataset.csv"
fi

echo ""
echo "========================================"
echo "✓ SCRIPT TERMINÉ"
echo "========================================"

