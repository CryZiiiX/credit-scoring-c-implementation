#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double compute_accuracy(int* y_true, int* y_pred, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return (double)correct / n_samples;
}

double compute_precision(int* y_true, int* y_pred, int n_samples) {
    int tp = 0, fp = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_pred[i] == 1) {
            if (y_true[i] == 1) tp++;
            else fp++;
        }
    }
    return (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
}

double compute_recall(int* y_true, int* y_pred, int n_samples) {
    int tp = 0, fn = 0;
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1) {
            if (y_pred[i] == 1) tp++;
            else fn++;
        }
    }
    return (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
}

double compute_f1_score(int* y_true, int* y_pred, int n_samples) {
    double precision = compute_precision(y_true, y_pred, n_samples);
    double recall = compute_recall(y_true, y_pred, n_samples);
    
    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}

void print_metrics(int* y_true, int* y_pred, int n_samples) {
    printf("Accuracy:  %.4f\n", compute_accuracy(y_true, y_pred, n_samples));
    printf("Precision: %.4f\n", compute_precision(y_true, y_pred, n_samples));
    printf("Recall:    %.4f\n", compute_recall(y_true, y_pred, n_samples));
    printf("F1-Score:  %.4f\n", compute_f1_score(y_true, y_pred, n_samples));
}

void save_metrics(const char* filename, int* y_true, int* y_pred, int n_samples) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Accuracy: %.6f\n", compute_accuracy(y_true, y_pred, n_samples));
    fprintf(file, "Precision: %.6f\n", compute_precision(y_true, y_pred, n_samples));
    fprintf(file, "Recall: %.6f\n", compute_recall(y_true, y_pred, n_samples));
    fprintf(file, "F1-Score: %.6f\n", compute_f1_score(y_true, y_pred, n_samples));
    
    fclose(file);
}

void save_metrics_with_auc(const char* filename, int* y_true, int* y_pred, int n_samples, double auc_roc) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Accuracy: %.6f\n", compute_accuracy(y_true, y_pred, n_samples));
    fprintf(file, "Precision: %.6f\n", compute_precision(y_true, y_pred, n_samples));
    fprintf(file, "Recall: %.6f\n", compute_recall(y_true, y_pred, n_samples));
    fprintf(file, "F1-Score: %.6f\n", compute_f1_score(y_true, y_pred, n_samples));
    fprintf(file, "AUC-ROC: %.6f\n", auc_roc);
    
    fclose(file);
}

/*
 * Structure pour associer probabilite et label vrai
 */
typedef struct {
    double probability;
    int true_label;
} PredictionPair;

/*
 * Fonction de comparaison pour qsort
 * Tri par probabilite decroissante
 */
static int compare_predictions_desc(const void* a, const void* b) {
    PredictionPair* pair_a = (PredictionPair*)a;
    PredictionPair* pair_b = (PredictionPair*)b;
    
    if (pair_a->probability > pair_b->probability) return -1;
    if (pair_a->probability < pair_b->probability) return 1;
    return 0;
}

/*
 * Calcule l'AUC-ROC (Area Under the ROC Curve)
 * 
 * L'AUC-ROC mesure la capacite du modele a discriminer les classes.
 * Une valeur de 1.0 indique une discrimination parfaite.
 * Une valeur de 0.5 indique une performance aleatoire.
 * 
 * Algorithme:
 * 1. Trier les predictions par probabilite decroissante
 * 2. Calculer TPR et FPR pour chaque seuil
 * 3. Calculer l'aire sous la courbe par methode des trapezes
 * 
 * Complexite temporelle: O(n log n) (tri)
 * Complexite spatiale: O(n)
 */
double compute_auc_roc(double* probabilities, int* y_true, int n_samples) {
    /* Etape 1: Creer et trier les paires (probabilite, label) */
    PredictionPair* pairs = (PredictionPair*)malloc(n_samples * sizeof(PredictionPair));
    if (!pairs) {
        fprintf(stderr, "Erreur allocation memoire pour AUC-ROC\n");
        return 0.5;
    }
    
    for (int i = 0; i < n_samples; i++) {
        pairs[i].probability = probabilities[i];
        pairs[i].true_label = y_true[i];
    }
    
    qsort(pairs, n_samples, sizeof(PredictionPair), compare_predictions_desc);
    
    /* Etape 2: Compter le nombre de positifs et negatifs */
    int n_positive = 0;
    int n_negative = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1) n_positive++;
        else n_negative++;
    }
    
    /* Cas limite: pas de positifs ou pas de negatifs */
    if (n_positive == 0 || n_negative == 0) {
        free(pairs);
        return 0.5;
    }
    
    /* Etape 3: Allouer tableaux pour TPR et FPR */
    double* tpr_values = (double*)malloc(n_samples * sizeof(double));
    double* fpr_values = (double*)malloc(n_samples * sizeof(double));
    
    if (!tpr_values || !fpr_values) {
        fprintf(stderr, "Erreur allocation memoire pour TPR/FPR\n");
        free(pairs);
        if (tpr_values) free(tpr_values);
        if (fpr_values) free(fpr_values);
        return 0.5;
    }
    
    /* Etape 4: Calculer TPR et FPR pour chaque seuil */
    int tp = 0;
    int fp = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (pairs[i].true_label == 1) {
            tp++;
        } else {
            fp++;
        }
        
        tpr_values[i] = (double)tp / n_positive;
        fpr_values[i] = (double)fp / n_negative;
    }
    
    /* Etape 5: Calculer l'aire sous la courbe (methode des trapezes) */
    double auc = 0.0;
    
    for (int i = 1; i < n_samples; i++) {
        double width = fpr_values[i] - fpr_values[i-1];
        double avg_height = (tpr_values[i] + tpr_values[i-1]) / 2.0;
        auc += width * avg_height;
    }
    
    /* Nettoyage */
    free(pairs);
    free(tpr_values);
    free(fpr_values);
    
    return auc;
}

