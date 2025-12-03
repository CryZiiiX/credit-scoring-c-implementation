#!/usr/bin/env python3
"""
Comparaison de l'implémentation C avec scikit-learn
Valide que les résultats sont similaires
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

def load_and_preprocess_data():
    """Charge et prétraite les données comme dans l'implémentation C"""
    print("📂 Chargement du dataset...")
    df = pd.read_csv("data/raw/credit_risk_dataset.csv")
    
    print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Encodage des variables catégorielles (comme dans encoder.c)
    print("\n🔄 Encodage des variables catégorielles...")
    
    # person_home_ownership: RENT=0, OWN=1, MORTGAGE=2, OTHER=3
    home_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    df['person_home_ownership'] = df['person_home_ownership'].map(home_mapping)
    
    # loan_intent: PERSONAL=0, EDUCATION=1, MEDICAL=2, VENTURE=3, HOMEIMPROVEMENT=4, DEBTCONSOLIDATION=5
    intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 
                     'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
    df['loan_intent'] = df['loan_intent'].map(intent_mapping)
    
    # loan_grade: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)
    
    # cb_person_default_on_file: N=0, Y=1
    default_mapping = {'N': 0, 'Y': 1}
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_mapping)
    
    print("✓ Variables catégorielles encodées")
    
    # Gestion des valeurs manquantes (remplacement par la moyenne)
    print("\n🔧 Gestion des valeurs manquantes...")
    df = df.fillna(df.mean())
    print("✓ Valeurs manquantes traitées")
    
    # Séparer X et y
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    return X, y

def train_sklearn_model(X_train, y_train, X_test, y_test):
    """Entraîne un modèle sklearn pour comparaison"""
    print("\n🤖 Entraînement du modèle scikit-learn...")
    
    # Normalisation (StandardScaler comme dans scaler.c)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Régression logistique
    # Paramètres similaires à l'implémentation C:
    # - solver='lbfgs' (optimizer)
    # - max_iter=1000 (comme dans main.c)
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("✓ Modèle entraîné")
    
    return model, scaler, X_train_scaled, X_test_scaled

def load_c_results():
    """Charge les résultats de l'implémentation C"""
    print("\n📊 Chargement des résultats de l'implémentation C...")
    
    results = {}
    
    # Charger les métriques de test
    try:
        with open("results/metrics/test_metrics.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy" in line:
                    results['c_accuracy'] = float(line.split(':')[1].strip())
                elif "Precision" in line:
                    results['c_precision'] = float(line.split(':')[1].strip())
                elif "Recall" in line:
                    results['c_recall'] = float(line.split(':')[1].strip())
                elif "F1-Score" in line:
                    results['c_f1'] = float(line.split(':')[1].strip())
        
        print("✓ Métriques C chargées")
    except FileNotFoundError:
        print("⚠ Fichiers de résultats C non trouvés. Exécutez d'abord le programme C.")
        return None
    
    return results

def compare_results(sklearn_metrics, c_results):
    """Compare les résultats sklearn vs C"""
    print("\n" + "=" * 60)
    print("COMPARAISON DES RÉSULTATS")
    print("=" * 60)
    
    if c_results is None:
        print("\n⚠ Impossible de comparer : résultats C manquants")
        return
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Métrique", "Scikit-learn", "C (custom)", "Différence"))
    print("-" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        sklearn_val = sklearn_metrics[metric]
        c_val = c_results[f'c_{metric}']
        diff = abs(sklearn_val - c_val)
        
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(name, sklearn_val, c_val, diff))
    
    print("\n" + "=" * 60)
    
    # Analyse des différences
    print("\n📊 ANALYSE:")
    
    max_diff = max([abs(sklearn_metrics[m] - c_results[f'c_{m}']) for m in metrics])
    
    if max_diff < 0.01:
        print("✓ Excellent ! Les résultats sont quasiment identiques (différence < 1%)")
    elif max_diff < 0.05:
        print("✓ Bon ! Les résultats sont très similaires (différence < 5%)")
    elif max_diff < 0.10:
        print("⚠ Les résultats sont similaires mais avec quelques différences (< 10%)")
    else:
        print("⚠ Les résultats diffèrent significativement (> 10%)")
    
    print("\nCauses possibles de différences:")
    print("  - Différences d'optimisation (L-BFGS vs Gradient Descent)")
    print("  - Précision numérique (float vs double)")
    print("  - Initialisation aléatoire du split train/test")
    print("  - Nombre d'itérations différent avant convergence")

def save_comparison_report(sklearn_metrics, c_results):
    """Sauvegarde un rapport de comparaison"""
    output_path = Path("results/sklearn_comparison.txt")
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COMPARAISON SCIKIT-LEARN VS IMPLÉMENTATION C\n")
        f.write("=" * 60 + "\n\n")
        
        if c_results:
            f.write("{:<20} {:<15} {:<15} {:<15}\n".format("Métrique", "Scikit-learn", "C (custom)", "Différence"))
            f.write("-" * 70 + "\n")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for metric, name in zip(metrics, metric_names):
                sklearn_val = sklearn_metrics[metric]
                c_val = c_results[f'c_{metric}']
                diff = abs(sklearn_val - c_val)
                
                f.write("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}\n".format(name, sklearn_val, c_val, diff))
    
    print(f"\n✓ Rapport sauvegardé: {output_path}")

def main():
    """Fonction principale"""
    print("\n" + "=" * 60)
    print("VALIDATION AVEC SCIKIT-LEARN")
    print("=" * 60 + "\n")
    
    # Charger et prétraiter
    X, y = load_and_preprocess_data()
    
    # Split train/test (80/20 comme dans le code C)
    print("\n✂️ Split train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Entraîner le modèle sklearn
    model, scaler, X_train_scaled, X_test_scaled = train_sklearn_model(
        X_train, y_train, X_test, y_test
    )
    
    # Prédictions
    print("\n🔮 Prédictions sur le test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculer les métriques
    sklearn_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print("\n📊 Métriques scikit-learn:")
    print(f"  Accuracy:  {sklearn_metrics['accuracy']:.4f}")
    print(f"  Precision: {sklearn_metrics['precision']:.4f}")
    print(f"  Recall:    {sklearn_metrics['recall']:.4f}")
    print(f"  F1-Score:  {sklearn_metrics['f1']:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Charger et comparer avec les résultats C
    c_results = load_c_results()
    compare_results(sklearn_metrics, c_results)
    save_comparison_report(sklearn_metrics, c_results)
    
    print("\n" + "=" * 60)
    print("✓ COMPARAISON TERMINÉE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

