#include "encoder.h"
#include "../utils/memory_manager.h"
#include <string.h>

LabelEncoder* fit_label_encoder(int* labels, int n_samples) {
    LabelEncoder* encoder = (LabelEncoder*)safe_malloc(sizeof(LabelEncoder));
    
    // Find unique labels
    int max_label = labels[0];
    for (int i = 1; i < n_samples; i++) {
        if (labels[i] > max_label) {
            max_label = labels[i];
        }
    }
    
    encoder->n_classes = max_label + 1;
    encoder->mapping = (int*)safe_malloc(encoder->n_classes * sizeof(int));
    
    for (int i = 0; i < encoder->n_classes; i++) {
        encoder->mapping[i] = i;
    }
    
    return encoder;
}

void encode_labels(int* labels, int n_samples, LabelEncoder* encoder) {
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] >= 0 && labels[i] < encoder->n_classes) {
            labels[i] = encoder->mapping[labels[i]];
        }
    }
}

void free_label_encoder(LabelEncoder* encoder) {
    if (encoder) {
        safe_free(encoder->mapping);
        safe_free(encoder);
    }
}

// ============= ENCODAGE CATÉGORIEL SPÉCIFIQUE AU DATASET =============

/**
 * Encode la variable person_home_ownership
 * RENT=0, OWN=1, MORTGAGE=2, OTHER=3
 */
int encode_home_ownership(const char* value) {
    if (strcmp(value, "RENT") == 0) return 0;
    if (strcmp(value, "OWN") == 0) return 1;
    if (strcmp(value, "MORTGAGE") == 0) return 2;
    if (strcmp(value, "OTHER") == 0) return 3;
    return 0;  // Défaut : RENT
}

/**
 * Encode la variable loan_intent
 * PERSONAL=0, EDUCATION=1, MEDICAL=2, VENTURE=3, 
 * HOMEIMPROVEMENT=4, DEBTCONSOLIDATION=5
 */
int encode_loan_intent(const char* value) {
    if (strcmp(value, "PERSONAL") == 0) return 0;
    if (strcmp(value, "EDUCATION") == 0) return 1;
    if (strcmp(value, "MEDICAL") == 0) return 2;
    if (strcmp(value, "VENTURE") == 0) return 3;
    if (strcmp(value, "HOMEIMPROVEMENT") == 0) return 4;
    if (strcmp(value, "DEBTCONSOLIDATION") == 0) return 5;
    return 0;  // Défaut : PERSONAL
}

/**
 * Encode la variable loan_grade (ordinal)
 * A=1, B=2, C=3, D=4, E=5, F=6, G=7
 */
int encode_loan_grade(const char* value) {
    if (strcmp(value, "A") == 0) return 1;
    if (strcmp(value, "B") == 0) return 2;
    if (strcmp(value, "C") == 0) return 3;
    if (strcmp(value, "D") == 0) return 4;
    if (strcmp(value, "E") == 0) return 5;
    if (strcmp(value, "F") == 0) return 6;
    if (strcmp(value, "G") == 0) return 7;
    return 3;  // Défaut : C (milieu de gamme)
}

/**
 * Encode la variable cb_person_default_on_file
 * N=0, Y=1
 */
int encode_default_on_file(const char* value) {
    if (strcmp(value, "N") == 0) return 0;
    if (strcmp(value, "Y") == 0) return 1;
    return 0;  // Défaut : N (pas de défaut)
}

