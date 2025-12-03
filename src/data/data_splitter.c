#include "data_splitter.h"
#include "../utils/memory_manager.h"
#include "../utils/utils.h"
#include <time.h>
#include <stdlib.h>

void shuffle_dataset(Dataset* dataset) {
    srand(time(NULL));
    
    for (int i = dataset->rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap rows
        double* temp_row = dataset->data[i];
        dataset->data[i] = dataset->data[j];
        dataset->data[j] = temp_row;
        
        // Swap labels
        int temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}

SplitData* split_dataset(Dataset* dataset, double train_ratio) {
    SplitData* split = (SplitData*)safe_malloc(sizeof(SplitData));
    
    int train_size = (int)(dataset->rows * train_ratio);
    int test_size = dataset->rows - train_size;
    
    // Allocate train dataset
    split->train = (Dataset*)safe_malloc(sizeof(Dataset));
    split->train->rows = train_size;
    split->train->cols = dataset->cols;
    split->train->data = allocate_matrix(train_size, dataset->cols);
    split->train->labels = (int*)safe_malloc(train_size * sizeof(int));
    
    // Allocate test dataset
    split->test = (Dataset*)safe_malloc(sizeof(Dataset));
    split->test->rows = test_size;
    split->test->cols = dataset->cols;
    split->test->data = allocate_matrix(test_size, dataset->cols);
    split->test->labels = (int*)safe_malloc(test_size * sizeof(int));
    
    // Copy data to train
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            split->train->data[i][j] = dataset->data[i][j];
        }
        split->train->labels[i] = dataset->labels[i];
    }
    
    // Copy data to test
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            split->test->data[i][j] = dataset->data[train_size + i][j];
        }
        split->test->labels[i] = dataset->labels[train_size + i];
    }
    
    return split;
}

void free_split_data(SplitData* split) {
    if (split) {
        free_dataset(split->train);
        free_dataset(split->test);
        safe_free(split);
    }
}

