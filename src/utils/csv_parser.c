#include "csv_parser.h"
#include "memory_manager.h"
#include "utils.h"
#include "../preprocessing/encoder.h"
#include <string.h>
#include <stdlib.h>

char** parse_csv_line(char* line, int* count) {
    int capacity = 10;
    char** tokens = (char**)safe_malloc(capacity * sizeof(char*));
    *count = 0;
    
    char* token = strtok(line, ",");
    while (token) {
        if (*count >= capacity) {
            capacity *= 2;
            tokens = (char**)safe_realloc(tokens, capacity * sizeof(char*));
        }
        
        // Remove newline
        size_t len = strlen(token);
        if (len > 0 && token[len-1] == '\n') {
            token[len-1] = '\0';
        }
        if (len > 1 && token[len-2] == '\r') {
            token[len-2] = '\0';
        }
        
        tokens[*count] = strdup(token);
        (*count)++;
        token = strtok(NULL, ",");
    }
    
    return tokens;
}

void free_parsed_line(char** tokens, int count) {
    for (int i = 0; i < count; i++) {
        free(tokens[i]);
    }
    free(tokens);
}

Dataset* load_csv(const char* filename, int has_header, int label_col) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    Dataset* dataset = (Dataset*)safe_malloc(sizeof(Dataset));
    char buffer[8192];
    
    // Skip header if present
    if (has_header) {
        if (!fgets(buffer, sizeof(buffer), file)) {
            fclose(file);
            free(dataset);
            return NULL;
        }
    }
    
    // Count rows and columns
    dataset->rows = 0;
    dataset->cols = 0;
    
    long pos = ftell(file);
    while (fgets(buffer, sizeof(buffer), file)) {
        if (dataset->rows == 0) {
            int count;
            char temp[8192];
            strcpy(temp, buffer);
            char** tokens = parse_csv_line(temp, &count);
            dataset->cols = (label_col >= 0) ? count - 1 : count;
            free_parsed_line(tokens, count);
        }
        dataset->rows++;
    }
    fseek(file, pos, SEEK_SET);
    
    // Allocate memory
    dataset->data = allocate_matrix(dataset->rows, dataset->cols);
    dataset->labels = (int*)safe_malloc(dataset->rows * sizeof(int));
    
    // Read data avec encodage catégoriel
    int row = 0;
    while (fgets(buffer, sizeof(buffer), file) && row < dataset->rows) {
        int count;
        char** tokens = parse_csv_line(buffer, &count);
        
        int col_idx = 0;
        for (int i = 0; i < count; i++) {
            if (i == label_col) {
                dataset->labels[row] = atoi(tokens[i]);
            } else {
                // Encodage spécifique selon la colonne
                if (i == 2) {
                    // person_home_ownership (catégoriel)
                    dataset->data[row][col_idx++] = (double)encode_home_ownership(tokens[i]);
                } else if (i == 4) {
                    // loan_intent (catégoriel)
                    dataset->data[row][col_idx++] = (double)encode_loan_intent(tokens[i]);
                } else if (i == 5) {
                    // loan_grade (catégoriel ordinal)
                    dataset->data[row][col_idx++] = (double)encode_loan_grade(tokens[i]);
                } else if (i == 10) {
                    // cb_person_default_on_file (catégoriel)
                    dataset->data[row][col_idx++] = (double)encode_default_on_file(tokens[i]);
                } else {
                    // Colonne numérique standard
                dataset->data[row][col_idx++] = atof(tokens[i]);
                }
            }
        }
        
        free_parsed_line(tokens, count);
        row++;
    }
    
    fclose(file);
    return dataset;
}

void free_dataset(Dataset* dataset) {
    if (dataset) {
        free_matrix(dataset->data, dataset->rows);
        safe_free(dataset->labels);
        safe_free(dataset);
    }
}

