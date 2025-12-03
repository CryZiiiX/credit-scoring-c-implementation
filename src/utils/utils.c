#include "utils.h"

double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double* allocate_vector(int size) {
    return (double*)malloc(size * sizeof(double));
}

void free_vector(double* vector) {
    free(vector);
}

void print_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_vector(double* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.4f ", vector[i]);
    }
    printf("\n");
}

int count_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;
    
    int count = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }
    fclose(file);
    return count - 1; // Exclude header
}

int count_columns(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;
    
    char buffer[4096];
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        return -1;
    }
    
    int count = 1;
    for (char* p = buffer; *p; p++) {
        if (*p == ',') count++;
    }
    fclose(file);
    return count;
}

