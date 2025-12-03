#include <stdio.h>
#include <stddef.h>

typedef struct {
    const double* ptr;
    size_t len;
} Slice;

void process_array(Slice slice) {
    printf("Received array of length %zu:\n", slice.len);
    for (size_t i = 0; i < slice.len; i++) {
        printf("%f ", slice.ptr[i]);
    }
    printf("\n");
}
