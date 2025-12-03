#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stdio.h>

typedef struct {
    double** data;
    int* labels;
    int rows;
    int cols;
} Dataset;

Dataset* load_csv(const char* filename, int has_header, int label_col);
void free_dataset(Dataset* dataset);
char** parse_csv_line(char* line, int* count);
void free_parsed_line(char** tokens, int count);

#endif

