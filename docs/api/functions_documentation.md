# Documentation API - Credit Risk Prediction

Documentation complète de toutes les fonctions publiques du projet.

---

## Table des Matières

1. [Utils](#1-utils)
2. [Memory Manager](#2-memory-manager)
3. [CSV Parser](#3-csv-parser)
4. [Data Loader](#4-data-loader)
5. [Data Splitter](#5-data-splitter)
6. [Preprocessing](#6-preprocessing)
7. [Scaler](#7-scaler)
8. [Encoder](#8-encoder)
9. [Logistic Regression](#9-logistic-regression)
10. [Metrics](#10-metrics)
11. [Confusion Matrix](#11-confusion-matrix)

---

## 1. Utils

**Fichier** : `src/utils/utils.h` / `utils.c`

### 1.1 Allocation Mémoire

#### `allocate_matrix`
```c
double** allocate_matrix(int rows, int cols);
```
**Description** : Alloue une matrice 2D de doubles.

**Paramètres** :
- `rows` : Nombre de lignes
- `cols` : Nombre de colonnes

**Retour** : Pointeur vers la matrice allouée

**Exemple** :
```c
double** matrix = allocate_matrix(100, 10);
// Utilisation: matrix[i][j]
free_matrix(matrix, 100);
```

#### `free_matrix`
```c
void free_matrix(double** matrix, int rows);
```
**Description** : Libère une matrice allouée avec `allocate_matrix`.

**Paramètres** :
- `matrix` : Matrice à libérer
- `rows` : Nombre de lignes

#### `allocate_vector`
```c
double* allocate_vector(int size);
```
**Description** : Alloue un vecteur de doubles.

**Paramètres** :
- `size` : Taille du vecteur

**Retour** : Pointeur vers le vecteur

#### `free_vector`
```c
void free_vector(double* vector);
```
**Description** : Libère un vecteur.

### 1.2 Affichage

#### `print_matrix`
```c
void print_matrix(double** matrix, int rows, int cols);
```
**Description** : Affiche une matrice (max 5x5 pour la lisibilité).

#### `print_vector`
```c
void print_vector(double* vector, int size);
```
**Description** : Affiche un vecteur (max 10 éléments).

### 1.3 Fichiers

#### `count_lines`
```c
int count_lines(const char* filename);
```
**Description** : Compte le nombre de lignes dans un fichier CSV (sans le header).

**Retour** : Nombre de lignes, -1 si erreur

#### `count_columns`
```c
int count_columns(const char* filename);
```
**Description** : Compte le nombre de colonnes dans un fichier CSV.

**Retour** : Nombre de colonnes, -1 si erreur

---

## 2. Memory Manager

**Fichier** : `src/utils/memory_manager.h` / `memory_manager.c`

#### `safe_malloc`
```c
void* safe_malloc(size_t size);
```
**Description** : Wrapper sécurisé de `malloc` avec vérification d'erreur.

**Paramètres** :
- `size` : Taille en octets à allouer

**Retour** : Pointeur vers la mémoire allouée

**Comportement** : Termine le programme avec `exit(1)` si l'allocation échoue.

#### `safe_calloc`
```c
void* safe_calloc(size_t num, size_t size);
```
**Description** : Wrapper sécurisé de `calloc` (allocation + initialisation à 0).

#### `safe_realloc`
```c
void* safe_realloc(void* ptr, size_t size);
```
**Description** : Wrapper sécurisé de `realloc`.

#### `safe_free`
```c
void safe_free(void* ptr);
```
**Description** : Libère la mémoire si le pointeur n'est pas NULL.

---

## 3. CSV Parser

**Fichier** : `src/utils/csv_parser.h` / `csv_parser.c`

### 3.1 Structures

#### `Dataset`
```c
typedef struct {
    double** data;      // Matrice des features [n_samples x n_features]
    int* labels;        // Vecteur des labels [n_samples]
    int rows;           // Nombre d'échantillons
    int cols;           // Nombre de features
} Dataset;
```

### 3.2 Fonctions

#### `load_csv`
```c
Dataset* load_csv(const char* filename, int has_header, int label_col);
```
**Description** : Charge un fichier CSV avec encodage automatique des variables catégorielles.

**Paramètres** :
- `filename` : Chemin du fichier CSV
- `has_header` : 1 si le fichier a un header, 0 sinon
- `label_col` : Index de la colonne cible (-1 si pas de labels)

**Retour** : Dataset alloué, NULL si erreur

**Encodage automatique** :
- Colonne 2 : `person_home_ownership` (RENT/OWN/MORTGAGE/OTHER → 0/1/2/3)
- Colonne 4 : `loan_intent` (PERSONAL/EDUCATION/etc. → 0/1/2/3/4/5)
- Colonne 5 : `loan_grade` (A/B/C/D/E/F/G → 1/2/3/4/5/6/7)
- Colonne 10 : `cb_person_default_on_file` (N/Y → 0/1)

**Exemple** :
```c
// Charger le dataset avec la colonne 8 comme cible
Dataset* data = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);

printf("Chargé: %d échantillons, %d features\n", data->rows, data->cols);

// Utilisation
double feature_value = data->data[0][5];  // Premier échantillon, 6ème feature
int label = data->labels[0];

free_dataset(data);
```

#### `free_dataset`
```c
void free_dataset(Dataset* dataset);
```
**Description** : Libère toute la mémoire d'un dataset.

#### `parse_csv_line`
```c
char** parse_csv_line(char* line, int* count);
```
**Description** : Parse une ligne CSV en tokens.

**Paramètres** :
- `line` : Ligne à parser (sera modifiée par `strtok`)
- `count` : Pointeur pour stocker le nombre de tokens

**Retour** : Array de strings (tokens)

#### `free_parsed_line`
```c
void free_parsed_line(char** tokens, int count);
```
**Description** : Libère la mémoire d'une ligne parsée.

---

## 4. Data Loader

**Fichier** : `src/data/data_loader.h` / `data_loader.c`

#### `load_train_data`
```c
Dataset* load_train_data(const char* filename);
```
**Description** : Charge les données d'entraînement (wrapper de `load_csv`).

#### `load_test_data`
```c
Dataset* load_test_data(const char* filename);
```
**Description** : Charge les données de test.

#### `save_dataset`
```c
void save_dataset(const char* filename, Dataset* dataset);
```
**Description** : Sauvegarde un dataset au format CSV.

**Exemple** :
```c
Dataset* train = load_train_data("data/processed/train.csv");
// Modifications...
save_dataset("data/processed/train_modified.csv", train);
free_dataset(train);
```

---

## 5. Data Splitter

**Fichier** : `src/data/data_splitter.h` / `data_splitter.c`

### 5.1 Structures

#### `SplitData`
```c
typedef struct {
    Dataset* train;
    Dataset* test;
} SplitData;
```

### 5.2 Fonctions

#### `split_dataset`
```c
SplitData* split_dataset(Dataset* dataset, double train_ratio);
```
**Description** : Divise un dataset en train/test.

**Paramètres** :
- `dataset` : Dataset à diviser
- `train_ratio` : Proportion du train (ex: 0.8 pour 80%)

**Retour** : Structure contenant train et test

**Exemple** :
```c
Dataset* full_data = load_csv("data.csv", 1, 8);
shuffle_dataset(full_data);  // Important: mélanger avant split

SplitData* split = split_dataset(full_data, 0.8);

printf("Train: %d samples\n", split->train->rows);
printf("Test: %d samples\n", split->test->rows);

free_split_data(split);
free_dataset(full_data);
```

#### `shuffle_dataset`
```c
void shuffle_dataset(Dataset* dataset);
```
**Description** : Mélange aléatoirement les lignes du dataset (Fisher-Yates).

**Important** : Appeler AVANT `split_dataset` pour un split aléatoire.

#### `free_split_data`
```c
void free_split_data(SplitData* split);
```
**Description** : Libère la mémoire d'un split (train + test).

---

## 6. Preprocessing

**Fichier** : `src/preprocessing/preprocessing.h` / `preprocessing.c`

#### `preprocess_dataset`
```c
Dataset* preprocess_dataset(Dataset* dataset);
```
**Description** : Pipeline de prétraitement complet (valeurs manquantes).

**Retour** : Le dataset modifié

**Exemple** :
```c
Dataset* data = load_csv("data.csv", 1, 8);
preprocess_dataset(data);  // Traite les valeurs manquantes
```

#### `handle_missing_values`
```c
void handle_missing_values(Dataset* dataset);
```
**Description** : Remplace les valeurs NaN par la moyenne de chaque colonne.

#### `normalize_features`
```c
void normalize_features(Dataset* dataset, Scaler* scaler);
```
**Description** : Normalise les features avec un scaler (wrapper de `transform_dataset`).

---

## 7. Scaler

**Fichier** : `src/preprocessing/scaler.h` / `scaler.c`

### 7.1 Structures

#### `Scaler`
```c
typedef struct {
    double* mean;        // Moyennes [n_features]
    double* std;         // Écarts-types [n_features]
    int n_features;      // Nombre de features
} Scaler;
```

### 7.2 Fonctions

#### `fit_scaler`
```c
Scaler* fit_scaler(Dataset* dataset);
```
**Description** : Calcule les paramètres de normalisation (moyenne et écart-type) sur le dataset.

**Paramètres** :
- `dataset` : Dataset d'entraînement

**Retour** : Scaler avec paramètres calculés

**Important** : Utiliser UNIQUEMENT sur le train set !

#### `transform_dataset`
```c
void transform_dataset(Dataset* dataset, Scaler* scaler);
```
**Description** : Normalise un dataset avec les paramètres du scaler.  
Formule : `x_normalized = (x - mean) / std`

**Paramètres** :
- `dataset` : Dataset à normaliser (sera modifié)
- `scaler` : Scaler avec paramètres (fit sur train)

**Exemple complet** :
```c
// 1. Fit sur train
Scaler* scaler = fit_scaler(train_data);

// 2. Transform train
transform_dataset(train_data, scaler);

// 3. Transform test (avec les MÊMES paramètres)
transform_dataset(test_data, scaler);

// 4. Sauvegarder pour utilisation ultérieure
save_scaler("scaler_params.txt", scaler);

free_scaler(scaler);
```

#### `save_scaler`
```c
void save_scaler(const char* filename, Scaler* scaler);
```
**Description** : Sauvegarde les paramètres du scaler dans un fichier texte.

#### `load_scaler`
```c
Scaler* load_scaler(const char* filename);
```
**Description** : Charge un scaler depuis un fichier.

#### `free_scaler`
```c
void free_scaler(Scaler* scaler);
```
**Description** : Libère la mémoire d'un scaler.

---

## 8. Encoder

**Fichier** : `src/preprocessing/encoder.h` / `encoder.c`

### 8.1 Fonctions d'Encodage Spécifiques

Ces fonctions sont utilisées automatiquement par `load_csv`.

#### `encode_home_ownership`
```c
int encode_home_ownership(const char* value);
```
**Description** : Encode la variable `person_home_ownership`.

**Mapping** :
- "RENT" → 0
- "OWN" → 1
- "MORTGAGE" → 2
- "OTHER" → 3

#### `encode_loan_intent`
```c
int encode_loan_intent(const char* value);
```
**Description** : Encode la variable `loan_intent`.

**Mapping** :
- "PERSONAL" → 0
- "EDUCATION" → 1
- "MEDICAL" → 2
- "VENTURE" → 3
- "HOMEIMPROVEMENT" → 4
- "DEBTCONSOLIDATION" → 5

#### `encode_loan_grade`
```c
int encode_loan_grade(const char* value);
```
**Description** : Encode la variable `loan_grade` (ordinal).

**Mapping** :
- "A" → 1 (meilleur grade)
- "B" → 2
- "C" → 3
- "D" → 4
- "E" → 5
- "F" → 6
- "G" → 7 (pire grade)

#### `encode_default_on_file`
```c
int encode_default_on_file(const char* value);
```
**Description** : Encode la variable `cb_person_default_on_file`.

**Mapping** :
- "N" → 0 (pas de défaut)
- "Y" → 1 (défaut)

### 8.2 Encodeur Générique

#### `LabelEncoder`
```c
typedef struct {
    int* mapping;
    int n_classes;
} LabelEncoder;
```

#### `fit_label_encoder`
```c
LabelEncoder* fit_label_encoder(int* labels, int n_samples);
```
**Description** : Crée un encodeur pour les labels.

#### `encode_labels`
```c
void encode_labels(int* labels, int n_samples, LabelEncoder* encoder);
```
**Description** : Encode les labels avec l'encodeur.

#### `free_label_encoder`
```c
void free_label_encoder(LabelEncoder* encoder);
```

---

## 9. Logistic Regression

**Fichier** : `src/models/logistic_regression.h` / `logistic_regression.c`

### 9.1 Structures

#### `LogisticRegression`
```c
typedef struct {
    double* weights;        // Poids [n_features]
    double bias;            // Biais
    int n_features;         // Nombre de features
    double learning_rate;   // Taux d'apprentissage
    int max_iterations;     // Nombre d'itérations max
} LogisticRegression;
```

### 9.2 Fonctions

#### `create_logistic_regression`
```c
LogisticRegression* create_logistic_regression(int n_features, double learning_rate, int max_iterations);
```
**Description** : Crée et initialise un modèle de régression logistique.

**Paramètres** :
- `n_features` : Nombre de features
- `learning_rate` : Taux d'apprentissage (typiquement 0.001 à 0.1)
- `max_iterations` : Nombre d'itérations de gradient descent

**Retour** : Modèle avec poids initialisés à 0

**Exemple** :
```c
LogisticRegression* model = create_logistic_regression(
    11,      // 11 features dans notre dataset
    0.01,    // Learning rate
    1000     // Iterations
);
```

#### `train_logistic_regression`
```c
void train_logistic_regression(LogisticRegression* model, Dataset* dataset);
```
**Description** : Entraîne le modèle sur un dataset par gradient descent.

**Algorithme** :
1. Pour chaque itération :
   - Calculer les prédictions (sigmoïde)
   - Calculer les gradients
   - Mettre à jour les poids et biais
   - Calculer et afficher le coût (tous les 100 iterations)

**Paramètres** :
- `model` : Modèle à entraîner
- `dataset` : Dataset d'entraînement (déjà normalisé)

**Affichage** : Coût tous les 100 iterations

**Exemple complet** :
```c
// 1. Créer le modèle
LogisticRegression* model = create_logistic_regression(11, 0.01, 1000);

// 2. Entraîner
train_logistic_regression(model, train_data);
// Output: Iteration 0, Cost: 0.693147
//         Iteration 100, Cost: 0.597342
//         ...

// 3. Sauvegarder
save_model("models/model.bin", model);

free_logistic_regression(model);
```

#### `predict`
```c
int* predict(LogisticRegression* model, Dataset* dataset);
```
**Description** : Prédit les classes (0 ou 1) pour un dataset.

**Paramètres** :
- `model` : Modèle entraîné
- `dataset` : Dataset de test (déjà normalisé)

**Retour** : Array de prédictions [n_samples] (à libérer avec `free`)

**Seuil** : 0.5 (si probabilité >= 0.5 → classe 1)

**Exemple** :
```c
int* predictions = predict(model, test_data);

for (int i = 0; i < test_data->rows; i++) {
    printf("Sample %d: Prédit=%d, Réel=%d\n", 
           i, predictions[i], test_data->labels[i]);
}

free(predictions);
```

#### `predict_proba`
```c
double* predict_proba(LogisticRegression* model, Dataset* dataset);
```
**Description** : Prédit les probabilités (entre 0 et 1) pour un dataset.

**Retour** : Array de probabilités [n_samples] (à libérer avec `free`)

**Exemple** :
```c
double* probas = predict_proba(model, test_data);

for (int i = 0; i < test_data->rows; i++) {
    printf("Sample %d: Proba=%.3f\n", i, probas[i]);
}

free(probas);
```

#### `save_model`
```c
void save_model(const char* filename, LogisticRegression* model);
```
**Description** : Sauvegarde le modèle en format binaire.

**Format** : `n_features | bias | weights[n_features]`

#### `load_model`
```c
LogisticRegression* load_model(const char* filename);
```
**Description** : Charge un modèle depuis un fichier.

**Exemple** :
```c
// Sauvegarder
save_model("models/my_model.bin", model);

// Charger plus tard
LogisticRegression* loaded_model = load_model("models/my_model.bin");
int* predictions = predict(loaded_model, test_data);
```

#### `free_logistic_regression`
```c
void free_logistic_regression(LogisticRegression* model);
```
**Description** : Libère la mémoire du modèle.

---

## 10. Metrics

**Fichier** : `src/evaluation/metrics.h` / `metrics.c`

#### `compute_accuracy`
```c
double compute_accuracy(int* y_true, int* y_pred, int n_samples);
```
**Description** : Calcule l'accuracy (taux de bonnes prédictions).

**Formule** : `Accuracy = (TP + TN) / Total`

**Retour** : Valeur entre 0 et 1

#### `compute_precision`
```c
double compute_precision(int* y_true, int* y_pred, int n_samples);
```
**Description** : Calcule la précision.

**Formule** : `Precision = TP / (TP + FP)`

**Signification** : Parmi les prédictions positives, combien sont correctes ?

#### `compute_recall`
```c
double compute_recall(int* y_true, int* y_pred, int n_samples);
```
**Description** : Calcule le recall (sensibilité).

**Formule** : `Recall = TP / (TP + FN)`

**Signification** : Parmi les vrais positifs, combien sont détectés ?

#### `compute_f1_score`
```c
double compute_f1_score(int* y_true, int* y_pred, int n_samples);
```
**Description** : Calcule le F1-Score (moyenne harmonique de Precision et Recall).

**Formule** : `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

#### `print_metrics`
```c
void print_metrics(int* y_true, int* y_pred, int n_samples);
```
**Description** : Affiche toutes les métriques.

**Exemple** :
```c
int* predictions = predict(model, test_data);

print_metrics(test_data->labels, predictions, test_data->rows);
// Output:
// Accuracy:  0.8090
// Precision: 0.4976
// Recall:    0.5077
// F1-Score:  0.5026

free(predictions);
```

#### `save_metrics`
```c
void save_metrics(const char* filename, int* y_true, int* y_pred, int n_samples);
```
**Description** : Sauvegarde les métriques dans un fichier texte.

---

## 11. Confusion Matrix

**Fichier** : `src/evaluation/confusion_matrix.h` / `confusion_matrix.c`

### 11.1 Structures

#### `ConfusionMatrix`
```c
typedef struct {
    int tp;  // True Positives
    int tn;  // True Negatives
    int fp;  // False Positives
    int fn;  // False Negatives
} ConfusionMatrix;
```

### 11.2 Fonctions

#### `compute_confusion_matrix`
```c
ConfusionMatrix* compute_confusion_matrix(int* y_true, int* y_pred, int n_samples);
```
**Description** : Calcule la matrice de confusion.

**Paramètres** :
- `y_true` : Labels réels
- `y_pred` : Labels prédits
- `n_samples` : Nombre d'échantillons

**Retour** : Structure avec TP, TN, FP, FN

**Exemple** :
```c
int* predictions = predict(model, test_data);
ConfusionMatrix* cm = compute_confusion_matrix(
    test_data->labels, predictions, test_data->rows
);

printf("TP: %d, TN: %d, FP: %d, FN: %d\n", 
       cm->tp, cm->tn, cm->fp, cm->fn);

free_confusion_matrix(cm);
free(predictions);
```

#### `print_confusion_matrix`
```c
void print_confusion_matrix(ConfusionMatrix* cm);
```
**Description** : Affiche la matrice de confusion formatée.

**Affichage** :
```
Confusion Matrix:
                Predicted
              Neg    Pos
Actual Neg   4643    635
       Pos    614    625
```

#### `save_confusion_matrix`
```c
void save_confusion_matrix(const char* filename, ConfusionMatrix* cm);
```
**Description** : Sauvegarde la matrice dans un fichier.

#### `free_confusion_matrix`
```c
void free_confusion_matrix(ConfusionMatrix* cm);
```
**Description** : Libère la mémoire de la matrice.

---

## Workflow Complet - Exemple

```c
#include "data/data_loader.h"
#include "data/data_splitter.h"
#include "preprocessing/preprocessing.h"
#include "preprocessing/scaler.h"
#include "models/logistic_regression.h"
#include "evaluation/metrics.h"
#include "evaluation/confusion_matrix.h"

int main() {
    // 1. Charger les données
    Dataset* dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
    
    // 2. Prétraiter
    preprocess_dataset(dataset);
    
    // 3. Split train/test
    shuffle_dataset(dataset);
    SplitData* split = split_dataset(dataset, 0.8);
    
    // 4. Normaliser
    Scaler* scaler = fit_scaler(split->train);
    transform_dataset(split->train, scaler);
    transform_dataset(split->test, scaler);
    
    // 5. Entraîner le modèle
    LogisticRegression* model = create_logistic_regression(
        split->train->cols, 0.01, 1000
    );
    train_logistic_regression(model, split->train);
    
    // 6. Évaluer
    int* predictions = predict(model, split->test);
    print_metrics(split->test->labels, predictions, split->test->rows);
    
    ConfusionMatrix* cm = compute_confusion_matrix(
        split->test->labels, predictions, split->test->rows
    );
    print_confusion_matrix(cm);
    
    // 7. Sauvegarder
    save_model("models/model.bin", model);
    save_scaler("models/scaler.txt", scaler);
    
    // 8. Nettoyer
    free(predictions);
    free_confusion_matrix(cm);
    free_logistic_regression(model);
    free_scaler(scaler);
    free_split_data(split);
    free_dataset(dataset);
    
    return 0;
}
```

---

**Dernière mise à jour** : Novembre 2024  
**Version** : 1.0

