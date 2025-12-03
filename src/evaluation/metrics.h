#ifndef METRICS_H
#define METRICS_H

double compute_accuracy(int* y_true, int* y_pred, int n_samples);
double compute_precision(int* y_true, int* y_pred, int n_samples);
double compute_recall(int* y_true, int* y_pred, int n_samples);
double compute_f1_score(int* y_true, int* y_pred, int n_samples);
double compute_auc_roc(double* probabilities, int* y_true, int n_samples);
void print_metrics(int* y_true, int* y_pred, int n_samples);
void save_metrics(const char* filename, int* y_true, int* y_pred, int n_samples);
void save_metrics_with_auc(const char* filename, int* y_true, int* y_pred, int n_samples, double auc_roc);

#endif
