/*
 * 05_nvtx_matmul.cu — Profiling matmul with NVTX annotations
 *
 * NVTX (NVIDIA Tools Extension) lets you mark code regions with names
 * that show up in Nsight Systems / Nsight Compute profilers.
 *
 * This is how you profile real CUDA applications:
 *   1. Annotate code with nvtxRangePush/Pop
 *   2. Run under profiler: nsys profile ./05_nvtx_matmul
 *   3. Open the .qdrep file in Nsight Systems GUI
 *   4. See exactly how long each phase takes
 *
 * Compile: nvcc -arch=sm_89 -lnvToolsExt -o 05_nvtx_matmul 05_nvtx_matmul.cu
 * Profile: nsys profile -o matmul_profile ./05_nvtx_matmul
 *
 * Note: If nvtx headers aren't found, the program still works without
 * profiling annotations. Set USE_NVTX=0 to disable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Try to include NVTX; gracefully degrade if not available */
#define USE_NVTX 1
#if USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP()      ((void)0)
#endif

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    printf("=== NVTX-Annotated Matmul Profiling ===\n\n");

    int M = 1024, K = 1024, N = 1024;
    size_t A_bytes = M * K * sizeof(float);
    size_t B_bytes = K * N * sizeof(float);
    size_t C_bytes = M * N * sizeof(float);

    /* ---- Phase: Host allocation ---- */
    NVTX_PUSH("Host Allocation");
    float *h_A = (float *)malloc(A_bytes);
    float *h_B = (float *)malloc(B_bytes);
    float *h_C = (float *)malloc(C_bytes);
    for (int i = 0; i < M * K; i++) h_A[i] = 0.01f * (i % 100);
    for (int i = 0; i < K * N; i++) h_B[i] = 0.01f * (i % 100);
    NVTX_POP();

    /* ---- Phase: Device allocation ---- */
    NVTX_PUSH("cudaMalloc");
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, B_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, C_bytes));
    NVTX_POP();

    /* ---- Phase: H2D transfer ---- */
    NVTX_PUSH("H2D Transfer");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));
    NVTX_POP();

    /* ---- Phase: Kernel execution ---- */
    NVTX_PUSH("Kernel: matmul_naive");
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    NVTX_POP();

    /* ---- Phase: D2H transfer ---- */
    NVTX_PUSH("D2H Transfer");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));
    NVTX_POP();

    /* ---- Phase: Verification ---- */
    NVTX_PUSH("Verification");
    printf("C[0][0] = %.4f (sanity check)\n", h_C[0]);
    printf("C[%d][%d] = %.4f\n", M - 1, N - 1, h_C[(M - 1) * N + (N - 1)]);
    NVTX_POP();

    /* ---- Cleanup ---- */
    NVTX_PUSH("Cleanup");
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    NVTX_POP();

    printf("\nTo profile, run:\n");
    printf("  nsys profile -o matmul_profile ./05_nvtx_matmul\n");
    printf("  nsys-ui matmul_profile.qdrep\n");
    printf("\nYou'll see colored regions for each NVTX range in the timeline.\n");

    printf("\n=== NVTX profiling demo complete! ===\n");
    return 0;
}
