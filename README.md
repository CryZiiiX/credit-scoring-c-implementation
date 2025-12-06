# Prédiction du Risque de Crédit Bancaire en C
![Présentation du projet](./docs/rapport/PagedeprésentationderapportCANVA.png)

Projet académique M1 Informatique - Techniques d'Apprentissage Artificiel

**Implémentation from scratch de deux algorithmes de machine learning en C pour prédire le risque de défaut de paiement bancaire.**

---

## Table des Matières

- [Description](#description)
- [Dataset](#dataset)
- [Modèles Implémentés](#modèles-implémentés)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Tests](#tests)
- [Résultats](#résultats)
- [Scripts d'Analyse](#scripts-danalyse)
- [Documentation](#documentation)
- [Notes Techniques](#notes-techniques)
- [Utilisation avec Docker](#utilisation-avec-docker)
- [Auteur](#auteur)

---

## Description

Ce projet implémente un système complet de machine learning en C pur pour prédire le risque de crédit bancaire.

L'objectif principal est de classifier les emprunteurs en deux catégories distinctes :
- **Classe 0** : Pas de défaut de paiement (bon risque)
- **Classe 1** : Défaut de paiement (mauvais risque)

### Caractéristiques principales

Le projet se distingue par plusieurs aspects techniques et pédagogiques :

- **Deux modèles implémentés** : Régression Logistique et Arbre de Décision CART
- **Pipeline complet** : chargement, prétraitement, entraînement et évaluation
- **Gestion de l'encodage catégoriel** pour quatre variables catégorielles
- **Normalisation StandardScaler** pour les variables numériques
- **Split train/test** avec ratio 80/20 et mélange aléatoire
- **Métriques complètes** : Accuracy, Precision, Recall, F1-Score, AUC-ROC et matrice de confusion
- **Tests unitaires** : 31 tests couvrant tous les modules
- **Comparaison avec scikit-learn** pour validation des résultats
- **Implémentation from scratch** sans bibliothèques de machine learning

Cette approche permet de comprendre en profondeur le fonctionnement interne des algorithmes de machine learning tout en démontrant l'efficacité d'une implémentation optimisée en C.

---

## Dataset

**Source** : Kaggle Credit Risk Dataset  
**Taille** : 32 581 lignes × 12 colonnes  
**Balance des classes** : 81% classe 0 (pas de défaut), 19% classe 1 (défaut)

### Variables

Le dataset contient un mélange de variables numériques et catégorielles qui décrivent les caractéristiques des emprunteurs et des prêts :

| Variable | Type | Description |
|----------|------|-------------|
| `person_age` | Numérique | Âge de l'emprunteur |
| `person_income` | Numérique | Revenu annuel |
| `person_home_ownership` | **Catégoriel** | RENT/OWN/MORTGAGE/OTHER |
| `person_emp_length` | Numérique | Années d'emploi |
| `loan_intent` | **Catégoriel** | PERSONAL/EDUCATION/MEDICAL/etc. |
| `loan_grade` | **Catégoriel** | A/B/C/D/E/F/G (ordinal) |
| `loan_amnt` | Numérique | Montant du prêt |
| `loan_int_rate` | Numérique | Taux d'intérêt |
| `loan_status` | **Cible** | 0=Pas de défaut, 1=Défaut |
| `loan_percent_income` | Numérique | Prêt / Revenu |
| `cb_person_default_on_file` | **Catégoriel** | Y/N |
| `cb_person_cred_hist_length` | Numérique | Historique de crédit |

### Encodage Catégoriel Implémenté

Les variables catégorielles sont automatiquement encodées lors du chargement des données :

- **person_home_ownership** : RENT=0, OWN=1, MORTGAGE=2, OTHER=3
- **loan_intent** : PERSONAL=0, EDUCATION=1, MEDICAL=2, VENTURE=3, HOMEIMPROVEMENT=4, DEBTCONSOLIDATION=5
- **loan_grade** : A=1, B=2, C=3, D=4, E=5, F=6, G=7 (ordinal)
- **cb_person_default_on_file** : N=0, Y=1

Cet encodage permet aux algorithmes de traiter efficacement les données catégorielles tout en préservant leur signification sémantique.

---

## Modèles Implémentés

Le projet implémente deux approches complémentaires de classification, chacune avec ses propres avantages et limitations.

### 1. Régression Logistique

**Algorithme** : Optimisation par Gradient Descent avec fonction sigmoïde

La régression logistique est un modèle linéaire qui modélise la probabilité d'appartenance à une classe en utilisant une fonction sigmoïde.

**Caractéristiques techniques** :

- Fonction d'activation : Sigmoïde σ(z) = 1 / (1 + e^(-z))
- Fonction de coût : Cross-Entropy Loss
- Optimisation : Gradient Descent avec batch complet
- Learning rate : 0.01
- Itérations : 1000
- Seuil de classification : 0.5

**Avantages** :

- Interprétable : les coefficients représentent l'importance des features
- Rapide à entraîner
- Bonne généralisation avec faible overfitting

**Limites** :

- Suppose une relation linéaire entre features et log-odds
- Performance limitée sur relations non-linéaires

### 2. Arbre de Décision CART

**Algorithme** : Classification And Regression Trees avec partitionnement récursif

L'arbre de décision construit un modèle de prédiction en créant une structure arborescente qui partitionne récursivement l'espace des features.

**Caractéristiques techniques** :

- Critères d'impureté : Gini et Entropie (choix utilisateur)
- Pre-pruning : max_depth=7, min_samples_split=20, min_samples_leaf=10
- Recherche exhaustive du meilleur split
- Construction récursive binaire
- Sauvegarde et chargement du modèle

**Avantages** :

- Capture les relations non-linéaires
- Pas d'hypothèse sur la distribution des données
- Gère naturellement les variables catégorielles ordinales
- Règles de décision explicites

**Limites** :

- Risque d'overfitting maîtrisé par pre-pruning
- Instabilité : petites variations peuvent produire un arbre différent
- Interprétabilité complexe pour grands arbres

---

## Architecture

Le projet suit une architecture modulaire qui sépare clairement les responsabilités de chaque composant :

```
credit_risk_prediction/
├── src/
│   ├── main.c                    # Point d'entrée (comparaison des 2 modèles)
│   ├── utils/
│   │   ├── utils.c/.h            # Fonctions utilitaires
│   │   ├── csv_parser.c/.h       # Parser CSV + encodage catégoriel
│   │   └── memory_manager.c/.h   # Gestion mémoire sécurisée
│   ├── data/
│   │   ├── data_loader.c/.h      # Chargement des données
│   │   └── data_splitter.c/.h    # Split train/test
│   ├── preprocessing/
│   │   ├── preprocessing.c/.h    # Pipeline de prétraitement
│   │   ├── scaler.c/.h           # Normalisation (StandardScaler)
│   │   └── encoder.c/.h          # Encodage catégoriel
│   ├── models/
│   │   ├── logistic_regression.c/.h  # Régression logistique
│   │   └── decision_tree.c/.h        # Arbre de décision CART
│   └── evaluation/
│       ├── metrics.c/.h          # Métriques (Acc, Prec, Recall, F1, AUC-ROC)
│       └── confusion_matrix.c/.h # Matrice de confusion
├── data/
│   ├── raw/                      # Dataset brut
│   ├── processed/                # Données prétraitées
│   └── stats/                    # Statistiques et graphiques
├── models/                       # Modèles sauvegardés
│   ├── logistic_model.bin
│   └── decision_tree_model.bin
├── results/
│   ├── metrics/                  # Métriques de performance
│   ├── plots/                    # Graphiques
│   └── logs/                     # Logs d'exécution
├── tests/                        # Tests unitaires (31 tests)
│   ├── test_data_loader.c        # 4 tests
│   ├── test_preprocessing.c      # 4 tests
│   ├── test_logistic_regression.c # 4 tests
│   ├── test_metrics.c            # 12 tests (dont AUC-ROC)
│   ├── test_decision_tree.c      # 7 tests
│   └── run_tests.sh
├── scripts/                      # Scripts Python d'analyse
└── docs/                         # Documentation
```

Cette organisation facilite la maintenance, les tests et l'extension future du projet.

---

## Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir les outils suivants installés sur votre système :

- **Compilateur C** : GCC (version 7.5 ou supérieure)
- **Make** : Pour la compilation automatisée
- **Python 3** : Pour les scripts d'analyse (optionnel)
  - Bibliothèques requises : pandas, numpy, matplotlib, seaborn, scikit-learn

### Compilation

La compilation du projet est simple et directe grâce au Makefile fourni :

```bash
# Cloner le projet (ou télécharger l'archive)
cd credit_risk_prediction

# Compiler le projet
make

# Vérifier la compilation
ls -l build/credit_risk_predictor
```

**Flags de compilation** : `-Wall -Wextra -O2 -std=c99 -lm`

Ces flags garantissent une compilation sans warnings, avec optimisations et compatibilité C99.

---

## Utilisation

### Exécution du Programme Principal

Une fois le projet compilé, vous pouvez l'exécuter de deux manières :

```bash
# Exécuter le programme complet (training + évaluation des 2 modèles)
./build/credit_risk_predictor

# Ou utiliser make
make run
```

### Pipeline d'Exécution

Le programme effectue automatiquement une série d'étapes pour traiter les données et entraîner les modèles :

1. **Chargement** du dataset depuis `data/raw/credit_risk_dataset.csv`
2. **Encodage** des variables catégorielles selon le schéma défini
3. **Prétraitement** des valeurs manquantes (imputation par médiane)
4. **Split** train/test avec ratio 80/20 et mélange aléatoire
5. **Normalisation** avec StandardScaler (ajusté sur train, appliqué au test)
6. **Entraînement Régression Logistique** avec 1000 itérations et learning rate 0.01
7. **Évaluation Régression Logistique** sur les ensembles train et test
8. **Entraînement Arbre de Décision** avec max_depth=7 et critère Gini
9. **Évaluation Arbre de Décision** sur les ensembles train et test
10. **Comparaison** des deux modèles avec affichage des métriques
11. **Sauvegarde** des résultats et des modèles

Chaque étape affiche des informations de progression pour vous permettre de suivre l'exécution.

### Résultats Attendus

Lors de l'exécution, vous devriez voir une sortie similaire à celle-ci :

```
Loading dataset...
Dataset loaded: 32581 samples, 11 features
Balance des classes: Classe 0 (pas de défaut) = 26378 (81.0%), Classe 1 (défaut) = 6203 (19.0%)

Training Logistic Regression...
Iteration 0, Cost: 0.693147
...
Iteration 900, Cost: 0.440379

--- Test Set Evaluation ---
Accuracy:  0.8070
Precision: 0.5084
Recall:    0.4949
F1-Score:  0.5016
AUC-ROC:   0.8180

=== DECISION TREE ===
Training Decision Tree (max_depth=7, min_samples_split=20, min_samples_leaf=10, criterion=GINI)...
Decision Tree trained successfully!
Actual tree depth: 7
Total nodes: 73

--- Decision Tree: Test Set ---
Accuracy:  0.9268
Precision: 0.9740
Recall:    0.6443
F1-Score:  0.7755
AUC-ROC:   0.9042

=== MODEL COMPARISON ===
+-----------------------+----------+----------+----------+
| Model                 | Accuracy | F1-Score | AUC-ROC  |
+-----------------------+----------+----------+----------+
| Logistic Regression   | 0.8070   | 0.5016   | 0.8180   |
| Decision Tree         | 0.9268   | 0.7755   | 0.9042   |
+-----------------------+----------+----------+----------+
```

### Fichiers Générés

Le programme génère plusieurs fichiers de sortie que vous pouvez utiliser pour analyser les résultats :

**Modèles** :

- `models/logistic_model.bin` : Modèle de régression logistique (poids et biais)
- `models/decision_tree_model.bin` : Arbre de décision sauvegardé

**Métriques** :

- `results/metrics/test_metrics.txt` : Métriques régression logistique
- `results/metrics/dt_test_metrics.txt` : Métriques arbre de décision
- `results/metrics/confusion_matrix.txt` : Matrice de confusion régression logistique
- `results/metrics/dt_confusion_matrix.txt` : Matrice de confusion arbre de décision

**Autres** :

- `data/processed/scaler_params.txt` : Paramètres du scaler

---

## Tests

### Exécuter la Suite de Tests

Pour vérifier que tout fonctionne correctement, vous pouvez exécuter la suite complète de tests :

```bash
cd tests
./run_tests.sh
```

### Tests Implémentés

Le projet inclut 31 tests unitaires répartis sur 5 fichiers, couvrant tous les aspects du système :

| Test | Description | Fichier | Nombre |
|------|-------------|---------|--------|
| **Data Loader** | Chargement CSV, encodage catégoriel | `test_data_loader.c` | 4 tests |
| **Preprocessing** | Scaler, valeurs manquantes | `test_preprocessing.c` | 4 tests |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC-ROC | `test_metrics.c` | 12 tests |
| **Logistic Regression** | Entraînement, prédictions, save/load | `test_logistic_regression.c` | 4 tests |
| **Decision Tree** | Gini, Entropie, prédictions, save/load | `test_decision_tree.c` | 7 tests |

**Total** : 31 tests unitaires répartis sur 5 fichiers

**Résultat attendu** : Tous les tests sont passés avec succès

Ces tests garantissent la fiabilité et la cohérence de l'implémentation.

---

## Résultats

### Comparaison des Modèles

Les deux modèles ont été évalués sur le même ensemble de test pour permettre une comparaison équitable :

| Métrique | Régression Logistique | Arbre de Décision | Meilleur |
|----------|----------------------|-------------------|----------|
| **Accuracy** | 80.70% | **92.68%** | Arbre (+11.98%) |
| **Precision** | 50.84% | **97.40%** | Arbre (+46.56%) |
| **Recall** | 49.49% | **64.43%** | Arbre (+14.94%) |
| **F1-Score** | 50.16% | **77.55%** | Arbre (+27.39%) |
| **AUC-ROC** | 81.80% | **90.42%** | Arbre (+8.62%) |
| **Temps entraînement** | 0.43s | 0.30s | Arbre |
| **Gap Train-Test** | 0.15% | 0.87% | Régression |

### Performance Détaillée

#### Régression Logistique

Les performances de la régression logistique montrent une bonne stabilité entre train et test :

| Métrique | Train | Test |
|----------|-------|------|
| **Accuracy** | 80.85% | 80.70% |
| **Precision** | 49.52% | 50.84% |
| **Recall** | 48.47% | 49.49% |
| **F1-Score** | 48.99% | 50.16% |
| **AUC-ROC** | - | 81.80% |

**Matrice de Confusion (Test Set)** :

|  | Prédit: Négatif | Prédit: Positif |
|--|----------------|----------------|
| **Réel: Négatif** | 4715 (TN) | 613 (FP) |
| **Réel: Positif** | 600 (FN) | 589 (TP) |

#### Arbre de Décision

L'arbre de décision démontre des performances supérieures sur toutes les métriques principales :

| Métrique | Train | Test |
|----------|-------|------|
| **Accuracy** | 93.55% | 92.68% |
| **Precision** | 96.76% | 97.40% |
| **Recall** | 68.43% | 64.43% |
| **F1-Score** | 80.16% | 77.55% |
| **AUC-ROC** | - | 90.42% |

**Matrice de Confusion (Test Set)** :

|  | Prédit: Négatif | Prédit: Positif |
|--|----------------|----------------|
| **Réel: Négatif** | 5284 (TN) | 44 (FP) |
| **Réel: Positif** | 423 (FN) | 766 (TP) |

### Analyse Comparative

**Arbre de Décision** :

L'arbre de décision se distingue par plusieurs points forts :

- **Performance supérieure** sur toutes les métriques principales
- **Très haute précision** (97.40%) : prédictions de défaut très fiables
- **AUC-ROC élevé** (90.42%) : excellente capacité de discrimination
- **Overfitting maîtrisé** : gap train-test inférieur à 1%
- **Capture les non-linéarités** du dataset

**Régression Logistique** :

La régression logistique présente des avantages complémentaires :

- **Meilleure généralisation** : gap train-test minimal (0.15%)
- **Plus interprétable** : 11 coefficients contre 73 nœuds
- **Performance modérée** mais stable
- **Rapide à entraîner** et déployer

**Conclusion** : L'Arbre de Décision est le modèle optimal pour ce problème de prédiction du risque de crédit, avec une amélioration de 11.98% en accuracy et de 8.62% en AUC-ROC par rapport à la régression logistique.

---

## Scripts d'Analyse

Le projet inclut plusieurs scripts Python pour faciliter l'analyse des données et la comparaison des résultats.

### 1. Exploration du Dataset

Ce script génère une analyse complète du dataset avec visualisations :

```bash
python3 scripts/explore_data.py
```

**Génère** :

- Distributions des variables numériques et catégorielles
- Matrice de corrélation
- Balance des classes
- Analyse par classe cible
- Résumé textuel

**Fichiers** : `data/stats/*.png`, `data/stats/data_exploration.txt`

### 2. Comparaison avec Scikit-learn

Ce script valide l'implémentation C en la comparant avec scikit-learn :

```bash
python3 scripts/compare_with_sklearn.py
```

**Compare** l'implémentation C avec scikit-learn pour valider les résultats.

**Exemple de sortie** :

```
Métrique            Scikit-learn    C (custom)      Différence     
----------------------------------------------------------------------
Accuracy            0.8100          0.8070          0.0030
Precision           0.5102          0.5084          0.0018
Recall              0.4989          0.4949          0.0040
F1-Score            0.5044          0.5016          0.0028
```

Les écarts observés sont inférieurs à 1%, confirmant la correction de l'implémentation.

### 3. Visualisation des Résultats

Ce script crée des graphiques professionnels pour visualiser les performances :

```bash
python3 scripts/plot_results.py
```

**Génère** :

- Courbe de coût (convergence)
- Matrice de confusion visuelle
- Comparaison métriques train/test
- Importance des features
- Figure récapitulative

**Fichiers** : `results/plots/*.png`

### 4. Split Pré-calculé (optionnel)

Si vous souhaitez utiliser un split pré-calculé pour des expériences reproductibles :

```bash
python3 scripts/split_dataset.py
```

Crée `data/processed/train.csv` et `test.csv` avec split stratifié.

---

## Documentation

### Documentation API

Pour comprendre en détail l'utilisation de chaque fonction, consultez :

- `docs/api/functions_documentation.md` : Documentation complète de toutes les fonctions

### Rapports Techniques

Le projet inclut une documentation académique complète :

- **Rapport principal** : `docs/rapport/M1 - TAA - Maxime BRONNY - 19009314.md`
- **Présentation** : `docs/presentation/slides.pdf`

---

## Notes Techniques

### Régression Logistique

Les détails techniques de l'implémentation de la régression logistique :

- **Fonction d'activation** : Sigmoïde σ(z) = 1 / (1 + e^(-z))
- **Fonction de coût** : Cross-Entropy Loss
- **Optimisation** : Gradient Descent avec batch complet
- **Learning rate** : 0.01
- **Itérations** : 1000
- **Seuil de classification** : 0.5

### Arbre de Décision CART

Les paramètres et caractéristiques de l'arbre de décision :

- **Critères d'impureté** : Gini ou Entropie (choix utilisateur)
- **Pre-pruning** :
  - max_depth : 7
  - min_samples_split : 20
  - min_samples_leaf : 10
- **Algorithme** : Recherche exhaustive du meilleur split
- **Structure** : Arbre binaire récursif
- **Profondeur réelle** : 7 niveaux
- **Nombre de nœuds** : 73

### Gestion de la Mémoire

Le projet utilise des pratiques rigoureuses pour éviter les fuites mémoire :

- Toutes les allocations utilisent `safe_malloc()` avec vérification d'erreur
- Libération systématique avec `free_dataset()`, `free_model()`, etc.
- Pas de fuites mémoire (testé avec Valgrind)

### Performance

Les temps d'exécution mesurés sur une machine standard :

- **Temps de chargement** : environ 1 seconde pour 32k lignes
- **Temps entraînement Régression** : environ 0.43 secondes
- **Temps entraînement Arbre** : environ 0.30 secondes
- **Temps total** : moins de 2 secondes

Ces performances démontrent l'efficacité d'une implémentation optimisée en C.

---

## Nettoyage

Pour nettoyer les fichiers générés et repartir sur une base propre :

```bash
# Nettoyer les fichiers de build
make clean

# Nettoyer tous les résultats générés
rm -rf results/metrics/* results/plots/* results/logs/*
rm -rf data/processed/* data/stats/*
rm -rf models/*.bin models/*.txt
```

---

## Dépannage

### Erreur de compilation

Si vous rencontrez des problèmes de compilation :

```bash
# Si vous avez des erreurs avec math.h
gcc ... -lm  # Toujours ajouter -lm à la fin

# Si vous avez des erreurs de compatibilité C99
gcc -std=c99 -D_POSIX_C_SOURCE=200809L ...
```

### Dataset non trouvé

Vérifiez que le dataset est présent au bon emplacement :

```bash
# Vérifier que le dataset est au bon endroit
ls -l data/raw/credit_risk_dataset.csv

# Télécharger manuellement depuis Kaggle si nécessaire
# Placer le fichier dans data/raw/
```

### Tests échouent

Si les tests échouent, essayez de recompiler proprement :

```bash
# Recompiler proprement
make clean
make
cd tests && ./run_tests.sh
```

---

## Utilisation avec Docker

Pour garantir la reproductibilité du projet sur n'importe quelle machine Linux, nous fournissons un environnement Docker.

Pour plus de détails sur les versions exactes de tous les composants (GCC, Python, bibliothèques), consultez le fichier `VERSIONS.md`.

### Prérequis

- Docker installé (version 20.10 ou supérieure)
- 2 GB d'espace disque disponible

### Installation et Exécution

#### 1. Build de l'image Docker

```bash
docker build -t credit-risk-predictor:latest .
```

Cette commande crée une image Docker contenant :

- GCC 12 (compilateur C)
- Python 3.11 + scikit-learn (pour comparaison)
- Toutes les dépendances nécessaires
- Le projet compilé et prêt à l'emploi

#### 2. Exécution du programme

```bash
# Exécution simple
docker run --rm credit-risk-predictor:latest

# Avec sauvegarde des résultats
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  credit-risk-predictor:latest
```

#### 3. Exécution des tests

```bash
docker run --rm credit-risk-predictor:latest bash -c "cd tests && ./run_tests.sh"
```

#### 4. Comparaison avec scikit-learn

```bash
docker run --rm credit-risk-predictor:latest python3 scripts/compare_with_sklearn.py
```

#### 5. Mode interactif (exploration)

```bash
docker run --rm -it credit-risk-predictor:latest bash
```

### Avantages de Docker pour ce Projet

L'utilisation de Docker apporte plusieurs bénéfices :

- **Reproductibilité** : Environnement identique sur toutes les machines (Ubuntu, Arch, Fedora, etc.)
- **Isolation** : Pas de conflits avec les installations système
- **Simplicité** : Une seule commande pour tout installer et exécuter
- **Professionnalisme** : Démontre la maîtrise des outils DevOps modernes

### Sans Docker

Le projet reste fonctionnel sans Docker sur n'importe quelle distribution Linux avec :

- GCC 7.5+ et Make
- Python 3.8+ (optionnel, pour comparaison uniquement)

Voir la section [Installation](#installation) pour les instructions détaillées.

---

## Licence

Projet académique - M1 Informatique  
**Usage** : Éducatif uniquement

---

## Auteur

**Projet M1** - Techniques d'Apprentissage Artificiel  
Année Académique 2024-2025

---

## Références

- Kaggle Credit Risk Dataset
- Breiman, L. (1984). Classification and Regression Trees (CART)
- Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression
- Andrew Ng - Machine Learning Course (Coursera)
- Bishop, C. (2006). Pattern Recognition and Machine Learning
- Scikit-learn Documentation

---

## Fonctionnalités Implémentées

- Encodage catégoriel automatique
- Tests unitaires complets (31 tests)
- Comparaison avec sklearn
- Visualisations professionnelles
- Calcul de l'AUC-ROC
- Régression Logistique
- Arbre de Décision CART
- Sauvegarde et chargement des modèles
- Comparaison automatique des modèles

## Perspectives Futures

- Régularisation L2
- Grid search hyperparamètres
- K-Fold Cross-Validation
- Random Forest (ensemble d'arbres)
- Gradient Boosting (XGBoost)

---

**Dernière mise à jour** : Décembre 2025

**Modèle recommandé** : Arbre de Décision (92.68% accuracy, 90.42% AUC-ROC)
