#include <stdio.h>
#include <stdlib.h>
// #include "data/data_loader.h"
#include "data/data_splitter.h"
#include "preprocessing/preprocessing.h"
#include "preprocessing/scaler.h"
#include "models/logistic_regression.h"
#include "models/decision_tree.h"
#include "evaluation/metrics.h"
#include "evaluation/confusion_matrix.h"

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    printf("Credit Risk Prediction - Machine Learning in C\n");
    printf("===============================================\n\n");
    
    // Load dataset
    printf("Loading dataset...\n");
    Dataset* dataset = load_csv("data/raw/credit_risk_dataset.csv", 1, 8);
    if (!dataset) {
        fprintf(stderr, "Error loading dataset\n");
        return 1;
    }
    printf("Dataset loaded: %d samples, %d features\n", dataset->rows, dataset->cols);
    
    // Afficher statistiques des classes
    int class_0 = 0, class_1 = 0;
    for (int i = 0; i < dataset->rows; i++) {
        if (dataset->labels[i] == 0) class_0++;
        else class_1++;
    }
    printf("Balance des classes: Classe 0 (pas de défaut) = %d (%.1f%%), Classe 1 (défaut) = %d (%.1f%%)\n\n",
           class_0, 100.0 * class_0 / dataset->rows,
           class_1, 100.0 * class_1 / dataset->rows);
    
    // Preprocess
    printf("Preprocessing data...\n");
    preprocess_dataset(dataset);
    
    // Split data
    printf("Splitting dataset (80%% train, 20%% test)...\n");
    shuffle_dataset(dataset);
    SplitData* split = split_dataset(dataset, 0.8);
    printf("Train set: %d samples\n", split->train->rows);
    printf("Test set: %d samples\n\n", split->test->rows);
    
    // Fit scaler on training data
    printf("Fitting scaler...\n");
    Scaler* scaler = fit_scaler(split->train);
    transform_dataset(split->train, scaler);
    transform_dataset(split->test, scaler);
    save_scaler("data/processed/scaler_params.txt", scaler);
    
    // Train model
    printf("\nTraining Logistic Regression...\n");
    LogisticRegression* model = create_logistic_regression(
        split->train->cols, 
        0.01,    // learning rate
        1000     // max iterations
    );
    train_logistic_regression(model, split->train);
    
    // Save model
    save_model("models/logistic_model.bin", model);
    printf("\nModel saved to models/logistic_model.bin\n");
    
    // Evaluate on train set
    printf("\n--- Training Set Evaluation ---\n");
    int* train_predictions = predict(model, split->train);
    print_metrics(split->train->labels, train_predictions, split->train->rows);
    save_metrics("results/metrics/train_metrics.txt", split->train->labels, train_predictions, split->train->rows);
    
    ConfusionMatrix* train_cm = compute_confusion_matrix(split->train->labels, train_predictions, split->train->rows);
    print_confusion_matrix(train_cm);
    
    // Evaluate on test set
    printf("\n--- Test Set Evaluation ---\n");
    int* test_predictions = predict(model, split->test);
    print_metrics(split->test->labels, test_predictions, split->test->rows);
    
    // Compute AUC-ROC
    double* test_probabilities = predict_proba(model, split->test);
    double lr_auc_roc = compute_auc_roc(test_probabilities, split->test->labels, split->test->rows);
    printf("AUC-ROC:   %.4f\n", lr_auc_roc);
    
    save_metrics_with_auc("results/metrics/test_metrics.txt", split->test->labels, test_predictions, split->test->rows, lr_auc_roc);
    
    ConfusionMatrix* test_cm = compute_confusion_matrix(split->test->labels, test_predictions, split->test->rows);
    print_confusion_matrix(test_cm);
    save_confusion_matrix("results/metrics/confusion_matrix.txt", test_cm);
    
    double lr_train_acc = compute_accuracy(split->train->labels, train_predictions, split->train->rows);
    double lr_test_acc = compute_accuracy(split->test->labels, test_predictions, split->test->rows);
    double lr_test_f1 = compute_f1_score(split->test->labels, test_predictions, split->test->rows);
    
    // Train Decision Tree
    printf("\n\n=== DECISION TREE ===\n");
    printf("Training Decision Tree (max_depth=7, min_samples_split=20, min_samples_leaf=10, criterion=GINI)...\n");
    DecisionTree* dt = create_decision_tree(7, 20, 10, GINI);
    train_decision_tree(dt, split->train);
    
    printf("Decision Tree trained successfully!\n");
    printf("Actual tree depth: %d\n", get_tree_depth(dt));
    printf("Total nodes: %d\n\n", count_tree_nodes(dt));
    
    // Evaluate Decision Tree on train set
    printf("--- Decision Tree: Training Set ---\n");
    int* dt_train_pred = predict_tree_dataset(dt, split->train);
    print_metrics(split->train->labels, dt_train_pred, split->train->rows);
    
    // Evaluate Decision Tree on test set
    printf("\n--- Decision Tree: Test Set ---\n");
    int* dt_test_pred = predict_tree_dataset(dt, split->test);
    print_metrics(split->test->labels, dt_test_pred, split->test->rows);
    
    double* dt_test_proba = get_tree_probabilities(dt, split->test);
    double dt_auc_roc = compute_auc_roc(dt_test_proba, split->test->labels, split->test->rows);
    printf("AUC-ROC:   %.4f\n", dt_auc_roc);
    
    save_metrics_with_auc("results/metrics/dt_test_metrics.txt", split->test->labels, dt_test_pred, split->test->rows, dt_auc_roc);
    
    ConfusionMatrix* dt_test_cm = compute_confusion_matrix(split->test->labels, dt_test_pred, split->test->rows);
    print_confusion_matrix(dt_test_cm);
    save_confusion_matrix("results/metrics/dt_confusion_matrix.txt", dt_test_cm);
    
    save_decision_tree("models/decision_tree_model.bin", dt);
    
    double dt_train_acc = compute_accuracy(split->train->labels, dt_train_pred, split->train->rows);
    double dt_test_acc = compute_accuracy(split->test->labels, dt_test_pred, split->test->rows);
    double dt_test_f1 = compute_f1_score(split->test->labels, dt_test_pred, split->test->rows);
    
    // Comparison
    printf("\n\n=== MODEL COMPARISON ===\n");
    printf("+-----------------------+----------+----------+----------+\n");
    printf("| Model                 | Accuracy | F1-Score | AUC-ROC  |\n");
    printf("+-----------------------+----------+----------+----------+\n");
    printf("| Logistic Regression   | %.4f   | %.4f   | %.4f   |\n", lr_test_acc, lr_test_f1, lr_auc_roc);
    printf("| Decision Tree         | %.4f   | %.4f   | %.4f   |\n", dt_test_acc, dt_test_f1, dt_auc_roc);
    printf("+-----------------------+----------+----------+----------+\n");
    
    printf("\nTrain vs Test Accuracy:\n");
    printf("Logistic Regression: Train=%.4f, Test=%.4f, Gap=%.4f\n", 
           lr_train_acc, lr_test_acc, lr_train_acc - lr_test_acc);
    printf("Decision Tree:       Train=%.4f, Test=%.4f, Gap=%.4f\n", 
           dt_train_acc, dt_test_acc, dt_train_acc - dt_test_acc);
    
    printf("\n\nResults saved in results/ and models/ directories\n");
    
    // Cleanup
    free(train_predictions);
    free(test_predictions);
    free(test_probabilities);
    free(dt_train_pred);
    free(dt_test_pred);
    free(dt_test_proba);
    free_confusion_matrix(train_cm);
    free_confusion_matrix(test_cm);
    free_confusion_matrix(dt_test_cm);
    free_decision_tree(dt);
    free_logistic_regression(model);
    free_scaler(scaler);
    free_split_data(split);
    free_dataset(dataset);
    
    return 0;
}
