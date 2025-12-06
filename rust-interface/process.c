#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>

typedef struct {
    const uint64_t* ptr;
    size_t len;
} Slice;

void process_array(Slice slice) {
    printf("Received array of length %zu:\n", slice.len);
    for (size_t i = 0; i < slice.len; i++) {
        printf("%" PRIu64 " ", slice.ptr[i]);
    }
    printf("\n");
}
