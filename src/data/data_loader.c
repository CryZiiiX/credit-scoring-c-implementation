#include "data_loader.h"
#include <stdio.h>

// #include "../utils/memory_manager.h"

Dataset* load_train_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

Dataset* load_test_data(const char* filename) {
    return load_csv(filename, 1, -1);
}

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
