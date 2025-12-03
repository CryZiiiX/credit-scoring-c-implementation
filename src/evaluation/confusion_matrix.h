#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

typedef struct {
    int tp;  // True Positives
    int tn;  // True Negatives
    int fp;  // False Positives
    int fn;  // False Negatives
} ConfusionMatrix;

ConfusionMatrix* compute_confusion_matrix(int* y_true, int* y_pred, int n_samples);
void print_confusion_matrix(ConfusionMatrix* cm);
void save_confusion_matrix(const char* filename, ConfusionMatrix* cm);
void free_confusion_matrix(ConfusionMatrix* cm);

#endif

