/*
 * common.h — Shared utilities for Phase 3: Attention kernels
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

/* ── Error checking ─────────────────────────────────────── */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/* ── Random fill ────────────────────────────────────────── */
inline void fill_random(float *p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

/* ── Verification ───────────────────────────────────────── */
inline void verify(const float *gpu, const float *ref, int n,
                   float tol, const char *label) {
    float max_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - ref[i]);
        if (err > max_err) { max_err = err; worst_idx = i; }
    }
    if (max_err <= tol) {
        printf("  %-30s PASS (max err=%.2e at idx %d)\n", label, max_err, worst_idx);
    } else {
        printf("  %-30s FAIL (max err=%.2e at idx %d, ref=%.6f, gpu=%.6f)\n",
               label, max_err, worst_idx, ref[worst_idx], gpu[worst_idx]);
    }
}

/* ── CPU reference: scaled dot-product attention ────────── */
/*
 * Q: [N, d]   K: [N, d]   V: [N, d]   O: [N, d]
 * Single-head, single-batch for simplicity.
 *
 * S = Q @ K^T / sqrt(d)     [N, N]
 * P = softmax(S, dim=-1)    [N, N]  (row-wise)
 * O = P @ V                 [N, d]
 */
inline void attention_cpu(const float *Q, const float *K, const float *V,
                          float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);

    /* Allocate S and P matrices [N, N] */
    float *S = (float *)malloc((size_t)N * N * sizeof(float));
    float *P = (float *)malloc((size_t)N * N * sizeof(float));

    /* S = Q @ K^T * scale */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = dot * scale;
        }
    }

    /* P = softmax(S, dim=-1) */
    for (int i = 0; i < N; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++)
            max_val = fmaxf(max_val, S[i * N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val);
            sum += P[i * N + j];
        }
        for (int j = 0; j < N; j++)
            P[i * N + j] /= sum;
    }

    /* O = P @ V */
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += P[i * N + j] * V[j * d + k];
            O[i * d + k] = acc;
        }
    }

    free(S);
    free(P);
}

/* ── Timing helper ──────────────────────────────────────── */
inline float time_kernel(void (*launch)(void), int warmup, int runs) {
    for (int i = 0; i < warmup; i++) launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / runs;
}
