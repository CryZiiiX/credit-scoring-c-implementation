/*****************************************************************************************************

Nom : src/data/data_loader.c

Rôle : Chargement et gestion des datasets depuis fichiers CSV

Auteur : Maxime BRONNY

Version : V1

Licence : Réalisé dans le cadre du cours Technique d'intelligence artificiel M1 INFORMATIQUE BIG-DATA

Usage : Pour compiler : make
        Pour executer : N/A

******************************************************************************************************/

#include "data_loader.h"
#include <stdio.h>

// #include "../utils/memory_manager.h"

/* **************************************************
 * # --- CHARGEMENT DONNÉES --- #
 * ************************************************** */

/**
 * Fonction : load_train_data
 * Rôle     : Charge un dataset d'entraînement depuis un fichier CSV
 * Param    : filename (nom du fichier CSV à charger)
 * Retour   : Dataset* (structure Dataset contenant les données)
 */
Dataset* load_train_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

/**
 * Fonction : load_test_data
 * Rôle     : Charge un dataset de test depuis un fichier CSV
 * Param    : filename (nom du fichier CSV à charger)
 * Retour   : Dataset* (structure Dataset contenant les données)
 */
Dataset* load_test_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

/**
 * Fonction : save_dataset
 * Rôle     : Sauvegarde un dataset dans un fichier CSV
 * Param    : filename (nom du fichier de destination), dataset (dataset à sauvegarder)
 * Retour   : void
 */
void save_dataset(const char* filename, Dataset* dataset) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    // Write data
    for (int i = 0; i < dataset->rows; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            fprintf(file, "%.6f", dataset->data[i][j]);
            if (j < dataset->cols - 1) fprintf(file, ",");
        }
        if (dataset->labels) {
            fprintf(file, ",%d", dataset->labels[i]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}
