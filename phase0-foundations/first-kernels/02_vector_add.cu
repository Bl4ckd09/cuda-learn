/*
 * 02_vector_add.cu — Three versions of vector addition
 *
 * The "hello world" of GPU computing. Each version teaches
 * a different kernel launch pattern:
 *
 *   v1: Single block (limited to 1024 elements)
 *   v2: Multi-block (scales to millions, but fragile)
 *   v3: Grid-stride loop (THE robust pattern — use this always)
 *
 * Compile: nvcc -arch=sm_89 -o 02_vector_add 02_vector_add.cu
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

/* ========== CPU reference implementation ========== */
void vector_add_cpu(const float *a, const float *b, float *c, int N) {
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

/* ========== v1: Single block ========== */
/*
 * Limitation: only works for N <= 1024 (max threads per block).
 * Each thread computes exactly one element.
 *
 * Launch: <<<1, N>>>
 */
__global__ void vector_add_v1(const float *a, const float *b, float *c, int N) {
    int i = threadIdx.x;   // no blockIdx since there's only 1 block
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

/* ========== v2: Multi-block ========== */
/*
 * Scales beyond 1024 by using multiple blocks.
 * Each thread still computes exactly one element.
 *
 * Launch: <<<CEIL_DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>
 *
 * IMPORTANT: The `if (i < N)` guard is essential!
 * Without it, threads in the last block may access out-of-bounds memory.
 */
__global__ void vector_add_v2(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

/* ========== v3: Grid-stride loop ========== */
/*
 * THE recommended pattern for production CUDA code.
 *
 * Why? A grid-stride loop decouples the number of threads launched
 * from the data size. Benefits:
 *   1. Works for ANY N, regardless of grid size
 *   2. Can launch fewer threads than elements → less launch overhead
 *   3. Each thread processes multiple elements → better amortized overhead
 *   4. Naturally handles edge cases (N not divisible by grid size)
 *
 * The stride = blockDim.x * gridDim.x = total threads in the grid.
 * Each thread starts at its global_id and strides by the total grid size.
 *
 * Example with 4 threads, N=10:
 *   Thread 0: processes elements 0, 4, 8
 *   Thread 1: processes elements 1, 5, 9
 *   Thread 2: processes elements 2, 6
 *   Thread 3: processes elements 3, 7
 */
__global__ void vector_add_v3(const float *a, const float *b, float *c, int N) {
    int stride = blockDim.x * gridDim.x;  // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

/* ========== Verification ========== */
int verify(const float *gpu_result, const float *cpu_result, int N) {
    for (int i = 0; i < N; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at i=%d: gpu=%.6f, cpu=%.6f\n",
                    i, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}

/* ========== Timing helper ========== */
typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

void timer_create(GpuTimer *t) {
    CHECK_CUDA(cudaEventCreate(&t->start));
    CHECK_CUDA(cudaEventCreate(&t->stop));
}

void timer_start(GpuTimer *t) {
    CHECK_CUDA(cudaEventRecord(t->start));
}

float timer_stop(GpuTimer *t) {
    CHECK_CUDA(cudaEventRecord(t->stop));
    CHECK_CUDA(cudaEventSynchronize(t->stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}

void timer_destroy(GpuTimer *t) {
    CHECK_CUDA(cudaEventDestroy(t->start));
    CHECK_CUDA(cudaEventDestroy(t->stop));
}

int main() {
    printf("=== Vector Addition: 3 Kernel Patterns ===\n\n");

    const int N = 10000000;  // 10M elements
    const int BLOCK_SIZE = 256;
    size_t bytes = N * sizeof(float);

    printf("N = %d (%.1f MB per vector)\n\n", N, bytes / 1e6);

    /* --- Allocate host memory --- */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    /* --- Initialize --- */
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }

    /* --- CPU reference --- */
    vector_add_cpu(h_a, h_b, h_c_cpu, N);

    /* --- Allocate device memory --- */
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    /* --- Copy data to GPU --- */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer_create(&timer);

    /* ===== v1: Single block (only 1024 elements) ===== */
    {
        int small_N = 1024;
        CHECK_CUDA(cudaMemset(d_c, 0, bytes));

        timer_start(&timer);
        vector_add_v1<<<1, small_N>>>(d_a, d_b, d_c, small_N);
        float ms = timer_stop(&timer);

        CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, small_N * sizeof(float), cudaMemcpyDeviceToHost));
        int ok = verify(h_c_gpu, h_c_cpu, small_N);
        printf("v1 (single block, N=%d): %.3f ms — %s\n", small_N, ms, ok ? "PASS" : "FAIL");
        printf("   Limitation: max N = 1024 (max threads per block)\n\n");
    }

    /* ===== v2: Multi-block ===== */
    {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_CUDA(cudaMemset(d_c, 0, bytes));

        timer_start(&timer);
        vector_add_v2<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        float ms = timer_stop(&timer);

        CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
        int ok = verify(h_c_gpu, h_c_cpu, N);
        printf("v2 (multi-block, N=%d): %.3f ms — %s\n", N, ms, ok ? "PASS" : "FAIL");
        printf("   Blocks: %d, Threads/block: %d\n\n", num_blocks, BLOCK_SIZE);
    }

    /* ===== v3: Grid-stride loop ===== */
    {
        // Use fewer blocks than v2 to show the loop handles it
        int num_blocks = 256;  // much less than CEIL_DIV(N, 256) = 39063
        CHECK_CUDA(cudaMemset(d_c, 0, bytes));

        timer_start(&timer);
        vector_add_v3<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        float ms = timer_stop(&timer);

        CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
        int ok = verify(h_c_gpu, h_c_cpu, N);
        printf("v3 (grid-stride, N=%d): %.3f ms — %s\n", N, ms, ok ? "PASS" : "FAIL");
        printf("   Only %d blocks (vs %d for v2), each thread does multiple elements\n",
               num_blocks, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Show elements per thread
        int total_threads = num_blocks * BLOCK_SIZE;
        printf("   Elements per thread: ~%d\n\n", (N + total_threads - 1) / total_threads);
    }

    /* ===== v3 with full blocks (for fair comparison) ===== */
    {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_CUDA(cudaMemset(d_c, 0, bytes));

        timer_start(&timer);
        vector_add_v3<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        float ms = timer_stop(&timer);

        CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
        int ok = verify(h_c_gpu, h_c_cpu, N);
        printf("v3 (grid-stride, full blocks): %.3f ms — %s\n\n", ms, ok ? "PASS" : "FAIL");
    }

    /* --- Bandwidth analysis --- */
    printf("=== Bandwidth Analysis ===\n");
    printf("Vector add reads 2 arrays + writes 1 array = 3 * %.1f MB = %.1f MB\n",
           bytes / 1e6, 3 * bytes / 1e6);
    printf("RTX 4070 theoretical memory bandwidth: ~504 GB/s\n");
    printf("At 0.1ms, effective bandwidth would be: %.1f GB/s\n",
           3 * bytes / 1e6 / 0.1);

    /* --- Cleanup --- */
    timer_destroy(&timer);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    printf("\n=== Vector add complete! ===\n");
    return 0;
}
