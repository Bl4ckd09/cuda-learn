/*
 * common.h — Shared utilities for Phase 2 elementwise kernels
 */

#ifndef ELEM_COMMON_H
#define ELEM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/* Fill with random values in [-1, 1] */
static void fill_random(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

/* Fill with random positive values in [0.1, 2] (for testing norms etc.) */
static void fill_random_positive(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 0.1f + 1.9f * (float)rand() / RAND_MAX;
}

/* Compare GPU output to CPU reference */
static int verify(const float *gpu, const float *ref, int n, float tol, const char *name) {
    float max_err = 0.0f;
    int max_idx = 0;
    int fail_count = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - ref[i]);
        float denom = fmaxf(fabsf(ref[i]), 1e-6f);
        if (err / denom > tol && err > tol) {
            fail_count++;
            if (err > max_err) {
                max_err = err;
                max_idx = i;
            }
        }
    }
    if (fail_count > 0) {
        printf("  %-20s FAIL: %d/%d errors (max err=%e at [%d]: gpu=%.6f ref=%.6f)\n",
               name, fail_count, n, max_err, max_idx, gpu[max_idx], ref[max_idx]);
        return 0;
    }
    printf("  %-20s PASS (max error within tol=%.0e)\n", name, tol);
    return 1;
}

/* Time a kernel (returns ms) */
static float time_kernel(void (*fn)(void), int warmup, int runs) {
    for (int i = 0; i < warmup; i++) { fn(); }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) { fn(); }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / runs;
}

#endif /* ELEM_COMMON_H */
