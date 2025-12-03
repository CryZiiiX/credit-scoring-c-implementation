#include "confusion_matrix.h"
#include "../utils/memory_manager.h"
#include <stdio.h>

ConfusionMatrix* compute_confusion_matrix(int* y_true, int* y_pred, int n_samples) {
    ConfusionMatrix* cm = (ConfusionMatrix*)safe_malloc(sizeof(ConfusionMatrix));
    cm->tp = cm->tn = cm->fp = cm->fn = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) cm->tp++;
        else if (y_true[i] == 0 && y_pred[i] == 0) cm->tn++;
        else if (y_true[i] == 0 && y_pred[i] == 1) cm->fp++;
        else if (y_true[i] == 1 && y_pred[i] == 0) cm->fn++;
    }
    
    return cm;
}

void print_confusion_matrix(ConfusionMatrix* cm) {
    printf("\nConfusion Matrix:\n");
    printf("                Predicted\n");
    printf("              Neg    Pos\n");
    printf("Actual Neg  %5d  %5d\n", cm->tn, cm->fp);
    printf("       Pos  %5d  %5d\n", cm->fn, cm->tp);
}

void save_confusion_matrix(const char* filename, ConfusionMatrix* cm) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        return;
    }
    
    fprintf(file, "Confusion Matrix:\n");
    fprintf(file, "TN: %d, FP: %d\n", cm->tn, cm->fp);
    fprintf(file, "FN: %d, TP: %d\n", cm->fn, cm->tp);
    
    fclose(file);
}

void free_confusion_matrix(ConfusionMatrix* cm) {
    safe_free(cm);
}

