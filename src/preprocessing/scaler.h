#ifndef SCALER_H
#define SCALER_H

#include "../utils/csv_parser.h"

typedef struct {
    double* mean;
    double* std;
    int n_features;
} Scaler;

Scaler* fit_scaler(Dataset* dataset);
void transform_dataset(Dataset* dataset, Scaler* scaler);
void save_scaler(const char* filename, Scaler* scaler);
Scaler* load_scaler(const char* filename);
void free_scaler(Scaler* scaler);

#endif

