/*
 * common.h — Shared utilities for all matmul implementations
 *
 * Provides: error checking, timing, CPU reference, verification, benchmarking.
 */

#ifndef MATMUL_COMMON_H
#define MATMUL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

/* ===== Error Checking ===== */
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/* ===== CPU Reference Matmul ===== */
static void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ===== Verification ===== */
static int verify_matmul(const float *gpu, const float *cpu, int size, float tol) {
    float max_err = 0.0f;
    int max_idx = 0;
    int fail_count = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        /* Use relative error for large values */
        float denom = fmaxf(fabsf(cpu[i]), 1.0f);
        float rel_err = err / denom;
        if (rel_err > tol) {
            fail_count++;
            if (err > max_err) {
                max_err = err;
                max_idx = i;
            }
        }
    }
    if (fail_count > 0) {
        printf("  VERIFY FAIL: %d/%d elements exceed tol=%.0e (max err=%e at [%d])\n",
               fail_count, size, tol, max_err, max_idx);
        return 0;
    }
    return 1;
}

/* ===== Benchmark result ===== */
typedef struct {
    float avg_ms;
    float min_ms;
    double gflops;
    double pct_peak;
    int pass;
} BenchResult;

/* RTX 4070 FP32 peak: 29.15 TFLOPS */
#define RTX4070_PEAK_TFLOPS 29.15

/*
 * Generic benchmark function.
 *
 * kernel_fn: a function that launches the kernel given (d_A, d_B, d_C, M, K, N).
 * Runs warmup, then `runs` iterations, returns timing & GFLOPS.
 */
typedef void (*KernelLauncher)(const float *d_A, const float *d_B, float *d_C,
                               int M, int K, int N);

static BenchResult benchmark_kernel(KernelLauncher launcher,
                                    int M, int K, int N,
                                    const float *h_A, const float *h_B,
                                    const float *h_C_ref, /* NULL to skip verify */
                                    int warmup_runs, int bench_runs) {
    size_t A_bytes = (size_t)M * K * sizeof(float);
    size_t B_bytes = (size_t)K * N * sizeof(float);
    size_t C_bytes = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));

    /* Warmup */
    for (int r = 0; r < warmup_runs; r++) {
        launcher(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Benchmark */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float min_ms = FLT_MAX;
    float total_ms = 0;

    for (int r = 0; r < bench_runs; r++) {
        CHECK_CUDA(cudaEventRecord(start));
        launcher(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    BenchResult result;
    result.avg_ms = total_ms / bench_runs;
    result.min_ms = min_ms;

    double flops = 2.0 * M * N * K;
    result.gflops = flops / (result.avg_ms / 1000.0) / 1e9;
    result.pct_peak = result.gflops / (RTX4070_PEAK_TFLOPS * 1000) * 100;

    /* Verify (if reference provided) */
    result.pass = -1;  // -1 = not tested
    if (h_C_ref) {
        float *h_C_gpu = (float *)malloc(C_bytes);
        CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, C_bytes, cudaMemcpyDeviceToHost));
        result.pass = verify_matmul(h_C_gpu, h_C_ref, M * N, 1e-3f);
        free(h_C_gpu);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return result;
}

static void print_result(const char *name, int M, int K, int N, BenchResult r) {
    const char *status = r.pass == 1 ? "PASS" : (r.pass == 0 ? "FAIL" : "N/A");
    printf("  %-20s | %4dx%4dx%4d | %8.3f ms | %8.1f GFLOPS | %5.1f%% | %s\n",
           name, M, K, N, r.avg_ms, r.gflops, r.pct_peak, status);
}

/* ===== Random initialization ===== */
static void fill_random(float *data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

#endif /* MATMUL_COMMON_H */
