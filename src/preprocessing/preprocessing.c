#include "preprocessing.h"
#include <math.h>

void handle_missing_values(Dataset* dataset) {
    // Replace NaN with column mean
    for (int j = 0; j < dataset->cols; j++) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < dataset->rows; i++) {
            if (!isnan(dataset->data[i][j])) {
                sum += dataset->data[i][j];
                count++;
            }
        }
        
        double mean = (count > 0) ? sum / count : 0.0;
        
        for (int i = 0; i < dataset->rows; i++) {
            if (isnan(dataset->data[i][j])) {
                dataset->data[i][j] = mean;
            }
        }
    }
}

void normalize_features(Dataset* dataset, Scaler* scaler) {
    transform_dataset(dataset, scaler);
}

Dataset* preprocess_dataset(Dataset* dataset) {
    handle_missing_values(dataset);
    return dataset;
}

