#include <stdio.h>
#include "../tools/common.cuh"

void addFromCPU(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}