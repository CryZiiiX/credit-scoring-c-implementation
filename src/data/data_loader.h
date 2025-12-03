#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "../utils/csv_parser.h"

Dataset* load_train_data(const char* filename);
Dataset* load_test_data(const char* filename);
void save_dataset(const char* filename, Dataset* dataset);

#endif
