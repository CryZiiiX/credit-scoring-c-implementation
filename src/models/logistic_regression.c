/*****************************************************************************************************

Nom : src/models/logistic_regression.c

Rôle : Implémentation from scratch de la régression logistique (entraînement, prédiction, sauvegarde)

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage : Pour compiler : make
        Pour executer : N/A

******************************************************************************************************/

#include "logistic_regression.h"
#include "../utils/memory_manager.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdio.h>

/* **************************************************
 * # --- FONCTIONS MATHÉMATIQUES --- #
 * ************************************************** */

/**
 * Fonction : sigmoid
 * Rôle     : Calcule la fonction sigmoïde 1/(1+exp(-z)) utilisée pour la régression logistique
 * Param    : z (valeur d'entrée)
 * Retour   : double (valeur sigmoïde entre 0 et 1)
 */
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/* **************************************************
 * # --- CRÉATION ET INITIALISATION --- #
 * ************************************************** */

/**
 * Fonction : create_logistic_regression
 * Rôle     : Crée et initialise un modèle de régression logistique avec poids à zéro
 * Param    : n_features (nombre de features), learning_rate (taux d'apprentissage), max_iterations (nombre max d'itérations)
 * Retour   : LogisticRegression* (modèle initialisé)
 */
LogisticRegression* create_logistic_regression(int n_features, double learning_rate, int max_iterations) {
    LogisticRegression* model = (LogisticRegression*)safe_malloc(sizeof(LogisticRegression));
    model->n_features = n_features;
    model->learning_rate = learning_rate;
    model->max_iterations = max_iterations;
    model->weights = allocate_vector(n_features);
    model->bias = 0.0;
    
    // Initialize weights to zero
    for (int i = 0; i < n_features; i++) {
        model->weights[i] = 0.0;
    }
    
    return model;
}

/* **************************************************
 * # --- ENTRAÎNEMENT --- #
 * ************************************************** */

/**
 * Fonction : train_logistic_regression
 * Rôle     : Entraîne le modèle de régression logistique par descente de gradient
 * Param    : model (modèle à entraîner), dataset (dataset d'entraînement)
 * Retour   : void
 */
void train_logistic_regression(LogisticRegression* model, Dataset* dataset) {
    int n_samples = dataset->rows;
    int n_features = dataset->cols;
    
    for (int iter = 0; iter < model->max_iterations; iter++) {
        double* gradients = allocate_vector(n_features);
        double bias_gradient = 0.0;
        double cost = 0.0;
        
        // Compute gradients
        for (int i = 0; i < n_samples; i++) {
            double z = model->bias;
            for (int j = 0; j < n_features; j++) {
                z += model->weights[j] * dataset->data[i][j];
            }
            
            double prediction = sigmoid(z);
            double error = prediction - dataset->labels[i];
            
            // Accumulate gradients
            for (int j = 0; j < n_features; j++) {
                gradients[j] += error * dataset->data[i][j];
            }
            bias_gradient += error;
            
            // Compute cost
            double y = dataset->labels[i];
            cost += -(y * log(prediction + 1e-15) + (1 - y) * log(1 - prediction + 1e-15));
        }
        
        // Update weights
        for (int j = 0; j < n_features; j++) {
            model->weights[j] -= model->learning_rate * gradients[j] / n_samples;
        }
        model->bias -= model->learning_rate * bias_gradient / n_samples;
        
        free_vector(gradients);
        
        if (iter % 100 == 0) {
            printf("Iteration %d, Cost: %.6f\n", iter, cost / n_samples);
        }
    }
}

/* **************************************************
 * # --- PRÉDICTION --- #
 * ************************************************** */

/**
 * Fonction : predict
 * Rôle     : Prédit les classes binaires (0 ou 1) pour un dataset en utilisant un seuil de 0.5
 * Param    : model (modèle entraîné), dataset (dataset à prédire)
 * Retour   : int* (tableau de prédictions binaires)
 */
int* predict(LogisticRegression* model, Dataset* dataset) {
    int* predictions = (int*)safe_malloc(dataset->rows * sizeof(int));
    
    for (int i = 0; i < dataset->rows; i++) {
        double z = model->bias;
        for (int j = 0; j < model->n_features; j++) {
            z += model->weights[j] * dataset->data[i][j];
        }
        predictions[i] = sigmoid(z) >= 0.5 ? 1 : 0;
    }
    
    return predictions;
}

/**
 * Fonction : predict_proba
 * Rôle     : Calcule les probabilités de classe positive pour chaque échantillon
 * Param    : model (modèle entraîné), dataset (dataset à prédire)
 * Retour   : double* (tableau de probabilités entre 0 et 1)
 */
double* predict_proba(LogisticRegression* model, Dataset* dataset) {
    double* probas = allocate_vector(dataset->rows);
    
    for (int i = 0; i < dataset->rows; i++) {
        double z = model->bias;
        for (int j = 0; j < model->n_features; j++) {
            z += model->weights[j] * dataset->data[i][j];
        }
        probas[i] = sigmoid(z);
    }
    
    return probas;
}

/* **************************************************
 * # --- SAUVEGARDE/CHARGEMENT --- #
 * ************************************************** */

/**
 * Fonction : save_model
 * Rôle     : Sauvegarde un modèle de régression logistique dans un fichier binaire
 * Param    : filename (nom du fichier de destination), model (modèle à sauvegarder)
 * Retour   : void
 */
void save_model(const char* filename, LogisticRegression* model) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fwrite(&model->n_features, sizeof(int), 1, file);
    fwrite(&model->bias, sizeof(double), 1, file);
    fwrite(model->weights, sizeof(double), model->n_features, file);
    
    fclose(file);
}

/**
 * Fonction : load_model
 * Rôle     : Charge un modèle de régression logistique depuis un fichier binaire
 * Param    : filename (nom du fichier source)
 * Retour   : LogisticRegression* (modèle chargé, NULL en cas d'erreur)
 */
LogisticRegression* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    LogisticRegression* model = (LogisticRegression*)safe_malloc(sizeof(LogisticRegression));
    
    if (fread(&model->n_features, sizeof(int), 1, file) != 1 ||
        fread(&model->bias, sizeof(double), 1, file) != 1) {
        fclose(file);
        free(model);
        return NULL;
    }
    
    model->weights = allocate_vector(model->n_features);
    if (fread(model->weights, sizeof(double), model->n_features, file) != (size_t)model->n_features) {
        fclose(file);
        free_logistic_regression(model);
        return NULL;
    }
    
    fclose(file);
    return model;
}

/**
 * Fonction : free_logistic_regression
 * Rôle     : Libère complètement la mémoire allouée pour un modèle de régression logistique
 * Param    : model (modèle à libérer)
 * Retour   : void
 */
void free_logistic_regression(LogisticRegression* model) {
    if (model) {
        free_vector(model->weights);
        safe_free(model);
    }
}
