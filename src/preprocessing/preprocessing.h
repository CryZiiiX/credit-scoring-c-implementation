#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "../utils/csv_parser.h"
#include "scaler.h"

void handle_missing_values(Dataset* dataset);
void normalize_features(Dataset* dataset, Scaler* scaler);
Dataset* preprocess_dataset(Dataset* dataset);

#endif
