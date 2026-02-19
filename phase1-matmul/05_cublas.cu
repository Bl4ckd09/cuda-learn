/*
 * 05_cublas.cu — cuBLAS baseline: NVIDIA's optimized matmul
 *
 * cuBLAS is NVIDIA's closed-source BLAS library. It achieves near-peak
 * performance through hand-tuned assembly for each GPU architecture.
 *
 * This is the "100% reference" we compare our kernels against.
 *
 * Key cuBLAS concepts:
 *   - Column-major by default (Fortran convention)
 *   - But we can use transpose flags to work with row-major data!
 *   - cublasSgemm = FP32 matmul, cublasHgemm = FP16 matmul
 *
 * The trick for row-major:
 *   We want: C = A @ B  (row-major)
 *   cuBLAS thinks in column-major, so we compute: C^T = B^T @ A^T
 *   Since row-major C is the same memory layout as column-major C^T,
 *   we just swap A and B and pass CUBLAS_OP_N (no transpose).
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o 05_cublas 05_cublas.cu
 */

#include "common.h"
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) do {                                     \
    cublasStatus_t status = call;                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",      \
                __FILE__, __LINE__, status);                        \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

static cublasHandle_t cublas_handle = NULL;

void ensure_cublas() {
    if (!cublas_handle) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
    }
}

/* ===== FP32 cuBLAS (cublasSgemm) ===== */
void launch_cublas_sgemm(const float *d_A, const float *d_B, float *d_C,
                         int M, int K, int N) {
    ensure_cublas();

    float alpha = 1.0f;
    float beta = 0.0f;

    /*
     * Row-major trick:
     *   C[M×N] = A[M×K] @ B[K×N]     (what we want, row-major)
     *
     * cuBLAS column-major interpretation:
     *   cublasSgemm(handle,
     *     CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose on either
     *     N, M, K,                    // note: N and M swapped!
     *     &alpha,
     *     d_B, N,                     // B is "first matrix" with leading dim N
     *     d_A, K,                     // A is "second matrix" with leading dim K
     *     &beta,
     *     d_C, N)                     // C with leading dim N
     *
     * This computes C^T = B^T @ A^T in column-major, which gives
     * us C = A @ B in row-major layout.
     */
    CHECK_CUBLAS(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

/* ===== FP16 cuBLAS (cublasHgemm) ===== */
/*
 * FP16 (half precision) is what's used in actual LLM training.
 * RTX 4070 FP16 peak: ~58.3 TFLOPS (2x FP32 peak)
 * With tensor cores: up to 166 TFLOPS (FP16 with FP32 accumulate)
 */
#include <cuda_fp16.h>

void launch_cublas_hgemm(const __half *d_A, const __half *d_B, __half *d_C,
                         int M, int K, int N) {
    ensure_cublas();

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

/* FP16 conversion kernels */
__global__ void float_to_half_kernel(const float *src, __half *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

__global__ void half_to_float_kernel(const __half *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

int main() {
    printf("=== Phase 1.5: cuBLAS Baseline ===\n");
    printf("RTX 4070 FP32 peak: %.1f TFLOPS\n", RTX4070_PEAK_TFLOPS);
    printf("RTX 4070 FP16 peak: ~58.3 TFLOPS (tensor cores: ~166 TFLOPS)\n\n");
    printf("  %-20s | %-14s | %10s | %14s | %5s | %s\n",
           "Kernel", "Size", "Time", "GFLOPS", "Peak%", "Check");
    printf("  --------------------|----------------|------------|----------------|-------|------\n");

    ensure_cublas();

    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 5; i++) {
        int S = sizes[i];
        size_t size = (size_t)S * S;
        size_t bytes = size * sizeof(float);

        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C_ref = NULL;
        fill_random(h_A, size);
        fill_random(h_B, size);

        if (S <= 1024) {
            h_C_ref = (float *)malloc(bytes);
            matmul_cpu(h_A, h_B, h_C_ref, S, S, S);
        }

        /* --- FP32 cuBLAS --- */
        BenchResult r32 = benchmark_kernel(launch_cublas_sgemm, S, S, S,
                                           h_A, h_B, h_C_ref, 5, 20);
        print_result("cuBLAS-FP32", S, S, S, r32);

        /* --- FP16 cuBLAS --- */
        {
            float *d_A_f, *d_B_f;
            __half *d_A_h, *d_B_h, *d_C_h;
            float *d_C_f;

            CHECK_CUDA(cudaMalloc(&d_A_f, bytes));
            CHECK_CUDA(cudaMalloc(&d_B_f, bytes));
            CHECK_CUDA(cudaMalloc(&d_A_h, size * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_B_h, size * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_C_h, size * sizeof(__half)));
            CHECK_CUDA(cudaMalloc(&d_C_f, bytes));

            CHECK_CUDA(cudaMemcpy(d_A_f, h_A, bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_B_f, h_B, bytes, cudaMemcpyHostToDevice));

            int bs = 256;
            int nb = CEIL_DIV(size, bs);
            float_to_half_kernel<<<nb, bs>>>(d_A_f, d_A_h, size);
            float_to_half_kernel<<<nb, bs>>>(d_B_f, d_B_h, size);
            CHECK_CUDA(cudaDeviceSynchronize());

            /* Warmup */
            for (int w = 0; w < 5; w++)
                launch_cublas_hgemm(d_A_h, d_B_h, d_C_h, S, S, S);
            CHECK_CUDA(cudaDeviceSynchronize());

            /* Benchmark */
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            int runs = 20;
            CHECK_CUDA(cudaEventRecord(start));
            for (int r = 0; r < runs; r++)
                launch_cublas_hgemm(d_A_h, d_B_h, d_C_h, S, S, S);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float total_ms;
            CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
            float avg_ms = total_ms / runs;
            double flops = 2.0 * S * S * S;
            double gflops = flops / (avg_ms / 1000.0) / 1e9;

            /* FP16 peak is different, but we still report as GFLOPS */
            printf("  %-20s | %4dx%4dx%4d | %8.3f ms | %8.1f GFLOPS | %5.1f%% | N/A\n",
                   "cuBLAS-FP16", S, S, S, avg_ms, gflops,
                   gflops / (58.3 * 1000) * 100);

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaFree(d_A_f));
            CHECK_CUDA(cudaFree(d_B_f));
            CHECK_CUDA(cudaFree(d_A_h));
            CHECK_CUDA(cudaFree(d_B_h));
            CHECK_CUDA(cudaFree(d_C_h));
            CHECK_CUDA(cudaFree(d_C_f));
        }

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    if (cublas_handle) {
        CHECK_CUBLAS(cublasDestroy(cublas_handle));
        cublas_handle = NULL;
    }

    printf("\ncuBLAS represents the performance ceiling for hand-written CUDA.\n");
    printf("Our goal: get within 2-3x of cuBLAS with our optimized kernels.\n");

    printf("\n=== cuBLAS benchmark complete! ===\n");
    return 0;
}
