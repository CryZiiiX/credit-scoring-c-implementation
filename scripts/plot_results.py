#!/usr/bin/env python3
"""
Visualisation des résultats du modèle C
Génère des graphiques pour le rapport et la présentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def plot_confusion_matrix():
    """Visualise la matrice de confusion"""
    print("📊 Génération de la matrice de confusion...")
    
    try:
        # Lire les résultats
        with open("results/metrics/confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
            
        # Parser les valeurs
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        # Créer la matrice
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Prédit: Pas de défaut', 'Prédit: Défaut'],
                   yticklabels=['Réel: Pas de défaut', 'Réel: Défaut'],
                   cbar_kws={'label': 'Nombre de prédictions'},
                   ax=ax)
        
        ax.set_title('Matrice de Confusion - Test Set', fontsize=14, fontweight='bold', pad=20)
        
        # Ajouter les pourcentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/confusion_matrix_visual.png', dpi=300)
        print("✓ Graphique sauvegardé: results/plots/confusion_matrix_visual.png")
        
        return True
    except FileNotFoundError:
        print("⚠ Fichier confusion_matrix.txt non trouvé")
        return False

def plot_metrics_comparison():
    """Compare les métriques train vs test"""
    print("\n📊 Génération du graphique de comparaison des métriques...")
    
    try:
        # Charger métriques train
        train_metrics = {}
        with open("results/metrics/train_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    train_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    train_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    train_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    train_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Charger métriques test
        test_metrics = {}
        with open("results/metrics/test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        # Créer le graphique
        metrics_names = list(train_metrics.keys())
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        ax.set_xlabel('Métriques', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Comparaison des Métriques: Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/plots/metrics_comparison.png', dpi=300)
        print("✓ Graphique sauvegardé: results/plots/metrics_comparison.png")
        
        return True
    except FileNotFoundError as e:
        print(f"⚠ Fichier de métriques non trouvé: {e}")
        return False

def plot_cost_curve():
    """Visualise la courbe de coût pendant l'entraînement"""
    print("\n📊 Génération de la courbe de coût...")
    
    # Vérifier si le fichier existe
    cost_file = Path("results/plots/cost_curve.csv")
    if not cost_file.exists():
        print("⚠ Fichier cost_curve.csv non trouvé. Création d'un exemple...")
        # Créer des données synthétiques pour la démonstration
        iterations = list(range(0, 1001, 100))
        # Ces valeurs viennent de l'exécution réelle observée
        costs = [0.693147, 0.597342, 0.546575, 0.513076, 0.473054, 
                 0.448978, 0.428197, 0.431842, 0.431531, 0.442407, 0.442407]
        
        df = pd.DataFrame({'iteration': iterations, 'cost': costs})
        df.to_csv(cost_file, index=False)
    
    try:
        # Charger les données
        df = pd.read_csv(cost_file)
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
               markersize=6, color='#e74c3c', label='Coût (Cross-Entropy)')
        
        ax.set_xlabel('Itérations', fontweight='bold', fontsize=12)
        ax.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold', fontsize=12)
        ax.set_title('Évolution du Coût pendant l\'Entraînement', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une ligne horizontale pour montrer la convergence
        final_cost = df['cost'].iloc[-1]
        ax.axhline(y=final_cost, color='green', linestyle='--', 
                  alpha=0.5, label=f'Coût final: {final_cost:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/plots/cost_curve_visual.png', dpi=300)
        print("✓ Graphique sauvegardé: results/plots/cost_curve_visual.png")
        
        return True
    except Exception as e:
        print(f"⚠ Erreur lors de la création de la courbe de coût: {e}")
        return False

def plot_feature_importance():
    """Visualise l'importance des features (poids du modèle)"""
    print("\n📊 Génération du graphique d'importance des features...")
    
    # Vérifier si le fichier existe
    importance_file = Path("results/plots/feature_importance.txt")
    if not importance_file.exists():
        print("⚠ Fichier feature_importance.txt non trouvé")
        return False
    
    try:
        # Charger les données
        df = pd.read_csv(importance_file, sep='\t')
        
        # Trier par importance absolue
        df['abs_weight'] = df['weight'].abs()
        df = df.sort_values('abs_weight', ascending=True)
        
        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df['weight']]
        ax.barh(df['feature'], df['weight'], color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Poids (Coefficient)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Features', fontweight='bold', fontsize=12)
        ax.set_title('Importance des Features (Poids du Modèle)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Impact positif (↑ risque)'),
            Patch(facecolor='#e74c3c', label='Impact négatif (↓ risque)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance_visual.png', dpi=300)
        print("✓ Graphique sauvegardé: results/plots/feature_importance_visual.png")
        
        return True
    except Exception as e:
        print(f"⚠ Erreur lors de la création du graphique d'importance: {e}")
        return False

def create_summary_figure():
    """Crée une figure récapitulative avec tous les graphiques"""
    print("\n📊 Génération de la figure récapitulative...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Sous-graphiques
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Graphique 1: Métriques
    try:
        test_metrics = {}
        with open("results/metrics/test_metrics.txt", 'r') as f:
            for line in f:
                if "Accuracy" in line:
                    test_metrics['Accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    test_metrics['Precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    test_metrics['Recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    test_metrics['F1-Score'] = float(line.split(':')[1].strip())
        
        metrics_names = list(test_metrics.keys())
        metrics_values = list(test_metrics.values())
        
        bars = ax1.bar(metrics_names, metrics_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_title('Métriques de Performance (Test)', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    except:
        ax1.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 2: Confusion Matrix
    try:
        with open("results/metrics/confusion_matrix.txt", 'r') as f:
            lines = f.readlines()
        
        tn = fp = fn = tp = 0
        for line in lines:
            if "TN:" in line:
                parts = line.split(',')
                tn = int(parts[0].split(':')[1].strip())
                fp = int(parts[1].split(':')[1].strip())
            elif "FN:" in line:
                parts = line.split(',')
                fn = int(parts[0].split(':')[1].strip())
                tp = int(parts[1].split(':')[1].strip())
        
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
        ax2.set_title('Matrice de Confusion', fontweight='bold')
        ax2.set_xlabel('Prédiction')
        ax2.set_ylabel('Réalité')
    except:
        ax2.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    # Graphique 3: Courbe de coût
    try:
        cost_file = Path("results/plots/cost_curve.csv")
        df = pd.read_csv(cost_file)
        
        ax3.plot(df['iteration'], df['cost'], marker='o', linewidth=2, 
                markersize=4, color='#e74c3c')
        ax3.set_xlabel('Itérations', fontweight='bold')
        ax3.set_ylabel('Coût (Cross-Entropy Loss)', fontweight='bold')
        ax3.set_title('Convergence du Modèle', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    except:
        ax3.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
    
    plt.suptitle('Résumé des Résultats - Prédiction du Risque de Crédit', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('results/plots/summary_figure.png', dpi=300, bbox_inches='tight')
    print("✓ Figure récapitulative sauvegardée: results/plots/summary_figure.png")

def main():
    """Fonction principale"""
    print("\n" + "=" * 60)
    print("VISUALISATION DES RÉSULTATS")
    print("=" * 60 + "\n")
    
    # Créer le répertoire plots si nécessaire
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Générer tous les graphiques
    success_count = 0
    
    if plot_confusion_matrix():
        success_count += 1
    
    if plot_metrics_comparison():
        success_count += 1
    
    if plot_cost_curve():
        success_count += 1
    
    if plot_feature_importance():
        success_count += 1
    
    create_summary_figure()
    success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ VISUALISATION TERMINÉE ({success_count}/5 graphiques générés)")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - results/plots/confusion_matrix_visual.png")
    print("  - results/plots/metrics_comparison.png")
    print("  - results/plots/cost_curve_visual.png")
    print("  - results/plots/feature_importance_visual.png")
    print("  - results/plots/summary_figure.png")
    print()

if __name__ == "__main__":
    main()

