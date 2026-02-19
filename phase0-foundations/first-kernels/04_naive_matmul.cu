/*
 * 04_naive_matmul.cu — Naive matrix multiplication on GPU
 *
 * C[M,N] = A[M,K] @ B[K,N]
 *
 * Each thread computes ONE output element by computing
 * the dot product of a row of A with a column of B:
 *
 *   C[row][col] = sum_{k=0}^{K-1} A[row][k] * B[k][col]
 *
 * This is the simplest possible GPU matmul. It's slow because:
 *   1. Every thread reads an entire row of A and column of B from GLOBAL memory
 *   2. Adjacent threads don't share any data (no data reuse)
 *   3. Memory access pattern for B is strided (poor coalescing for columns)
 *
 * We'll optimize this in Phase 1 with shared memory tiling.
 *
 * Why matmul matters for your LLM:
 *   - Every nn.Linear layer is a matmul: output = input @ weight^T
 *   - Attention: Q@K^T and P@V are matmuls
 *   - For nanochat (d=1024), each transformer layer has ~6 matmuls
 *   - Training speed is dominated by matmul throughput
 *
 * Compile: nvcc -arch=sm_89 -o 04_naive_matmul 04_naive_matmul.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* ========== CPU matmul (reference) ========== */
void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N) {
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

/* ========== GPU naive matmul ========== */
/*
 * 2D thread mapping:
 *   row = blockIdx.y * blockDim.y + threadIdx.y  → which row of C
 *   col = blockIdx.x * blockDim.x + threadIdx.x  → which column of C
 *
 * Why (x → col, y → row)?
 * Because threadIdx.x changes fastest, and we want adjacent threads
 * to access adjacent columns (adjacent memory) in C for coalescing.
 *
 * Memory access pattern for this kernel:
 *   - Each thread reads K elements from A (one row) → OK (sequential)
 *   - Each thread reads K elements from B (one column) → BAD (stride N)
 *   - Each thread writes 1 element to C → OK (coalesced if x varies)
 */
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* ========== Verification ========== */
int verify(const float *gpu, const float *cpu, int size, float tol) {
    float max_err = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        if (err > max_err) {
            max_err = err;
            max_idx = i;
        }
    }
    int pass = max_err < tol;
    if (!pass) {
        printf("  FAIL: max error = %e at index %d (gpu=%.6f, cpu=%.6f)\n",
               max_err, max_idx, gpu[max_idx], cpu[max_idx]);
    }
    return pass;
}

void benchmark_matmul(int M, int K, int N) {
    size_t A_bytes = M * K * sizeof(float);
    size_t B_bytes = K * N * sizeof(float);
    size_t C_bytes = M * N * sizeof(float);

    /* Allocate host */
    float *h_A = (float *)malloc(A_bytes);
    float *h_B = (float *)malloc(B_bytes);
    float *h_C_cpu = (float *)malloc(C_bytes);
    float *h_C_gpu = (float *)malloc(C_bytes);

    /* Initialize with small random values */
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    /* CPU reference (only for small matrices) */
    int do_cpu = (M <= 1024 && K <= 1024 && N <= 1024);
    if (do_cpu) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    }

    /* Allocate device */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));

    /* Launch config */
    int BLOCK_SIZE = 16;  // 16x16 = 256 threads per block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Warmup */
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Benchmark */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int runs = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / runs;

    /* Performance metrics */
    // FLOPS for matmul: 2 * M * N * K (multiply + add per element)
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    // RTX 4070: ~29.1 TFLOPS FP32 peak
    double peak_tflops = 29.1;
    double efficiency = gflops / (peak_tflops * 1000) * 100;

    /* Verify */
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, C_bytes, cudaMemcpyDeviceToHost));
    const char *status = "N/A";
    if (do_cpu) {
        status = verify(h_C_gpu, h_C_cpu, M * N, 1e-3f) ? "PASS" : "FAIL";
    }

    printf("  %4dx%4dx%4d | %7.2f ms | %8.1f GFLOPS | %5.1f%% peak | %s\n",
           M, K, N, avg_ms, gflops, efficiency, status);

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
}

int main() {
    printf("=== Naive GPU Matrix Multiplication ===\n");
    printf("C[M,N] = A[M,K] @ B[K,N], one thread per output element\n");
    printf("RTX 4070 FP32 peak: ~29.1 TFLOPS\n\n");

    printf("  Size          |    Time   |    GFLOPS   | %% Peak | Check\n");
    printf("  --------------|-----------|-------------|--------|------\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n; i++) {
        benchmark_matmul(sizes[i], sizes[i], sizes[i]);
    }

    printf("\nObservations:\n");
    printf("  - Naive matmul achieves only ~1-5%% of peak TFLOPS\n");
    printf("  - The bottleneck is global memory access (no data reuse)\n");
    printf("  - Each output element reads 2*K floats from global memory\n");
    printf("  - Phase 1 will fix this with shared memory tiling\n");

    printf("\n=== Naive matmul complete! ===\n");
    return 0;
}
