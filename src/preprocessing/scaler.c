#include "scaler.h"
#include "../utils/memory_manager.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdio.h>

Scaler* fit_scaler(Dataset* dataset) {
    Scaler* scaler = (Scaler*)safe_malloc(sizeof(Scaler));
    scaler->n_features = dataset->cols;
    scaler->mean = allocate_vector(dataset->cols);
    scaler->std = allocate_vector(dataset->cols);
    
    // Calculate mean
    for (int j = 0; j < dataset->cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < dataset->rows; i++) {
            sum += dataset->data[i][j];
        }
        scaler->mean[j] = sum / dataset->rows;
    }
    
    // Calculate std
    for (int j = 0; j < dataset->cols; j++) {
        double sum_sq = 0.0;
        for (int i = 0; i < dataset->rows; i++) {
            double diff = dataset->data[i][j] - scaler->mean[j];
            sum_sq += diff * diff;
        }
        scaler->std[j] = sqrt(sum_sq / dataset->rows);
        if (scaler->std[j] < 1e-8) scaler->std[j] = 1.0;
    }
    
    return scaler;
}

void transform_dataset(Dataset* dataset, Scaler* scaler) {
    for (int i = 0; i < dataset->rows; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            dataset->data[i][j] = (dataset->data[i][j] - scaler->mean[j]) / scaler->std[j];
        }
    }
}

void save_scaler(const char* filename, Scaler* scaler) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "%d\n", scaler->n_features);
    for (int i = 0; i < scaler->n_features; i++) {
        fprintf(file, "%.10f %.10f\n", scaler->mean[i], scaler->std[i]);
    }
    
    fclose(file);
}

Scaler* load_scaler(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    Scaler* scaler = (Scaler*)safe_malloc(sizeof(Scaler));
    if (fscanf(file, "%d", &scaler->n_features) != 1) {
        fclose(file);
        free(scaler);
        return NULL;
    }
    
    scaler->mean = allocate_vector(scaler->n_features);
    scaler->std = allocate_vector(scaler->n_features);
    
    for (int i = 0; i < scaler->n_features; i++) {
        if (fscanf(file, "%lf %lf", &scaler->mean[i], &scaler->std[i]) != 2) {
            fclose(file);
            free_scaler(scaler);
            return NULL;
        }
    }
    
    fclose(file);
    return scaler;
}

void free_scaler(Scaler* scaler) {
    if (scaler) {
        free_vector(scaler->mean);
        free_vector(scaler->std);
        safe_free(scaler);
    }
}

