#ifndef DATA_SPLITTER_H
#define DATA_SPLITTER_H

#include "../utils/csv_parser.h"

typedef struct {
    Dataset* train;
    Dataset* test;
} SplitData;

SplitData* split_dataset(Dataset* dataset, double train_ratio);
void free_split_data(SplitData* split);
void shuffle_dataset(Dataset* dataset);

#endif
