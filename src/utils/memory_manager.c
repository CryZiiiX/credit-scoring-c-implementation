#include "memory_manager.h"
#include <stdio.h>

void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void* safe_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr && num > 0 && size > 0) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "Memory reallocation failed\n");
        exit(1);
    }
    return new_ptr;
}

void safe_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

