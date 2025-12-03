#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <stdlib.h>

void* safe_malloc(size_t size);
void* safe_calloc(size_t num, size_t size);
void* safe_realloc(void* ptr, size_t size);
void safe_free(void* ptr);

#endif

