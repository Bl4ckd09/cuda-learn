/*
 * 04_cuda_hello.cu — Your first CUDA program
 *
 * This is the simplest possible CUDA program:
 *   1. Allocate GPU memory
 *   2. Launch a kernel (GPU function)
 *   3. Copy result back
 *   4. Print it
 *
 * Compile: nvcc -arch=sm_89 -o 04_cuda_hello 04_cuda_hello.cu
 * Run:     ./04_cuda_hello
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* ---------- CUDA error checking macro ---------- */
/* You'll use this in every CUDA program from now on */
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/*
 * __global__ means: this function runs on the GPU, called from CPU.
 *
 * Three function qualifiers in CUDA:
 *   __global__  — runs on GPU, called from CPU (or GPU in dynamic parallelism)
 *   __device__  — runs on GPU, called from GPU only
 *   __host__    — runs on CPU, called from CPU (default, can be omitted)
 *
 * __global__ functions must return void.
 */
__global__ void hello_kernel(int *d_result) {
    // This code runs on the GPU!
    // threadIdx.x = index of this thread within its block
    d_result[0] = 42;  // write to GPU memory
    printf("Hello from GPU! Thread %d in block %d\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    printf("=== First CUDA Program ===\n\n");

    // --- Query GPU info ---
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Number of CUDA devices: %d\n", device_count);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max block dims: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid dims: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Warp size: %d\n\n", prop.warpSize);

    // --- Allocate GPU memory ---
    int *d_result;  // d_ prefix = device (GPU) memory, convention
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    // --- Launch kernel ---
    // <<<grid_dim, block_dim>>> is the CUDA launch configuration
    // <<<1, 4>>> means: 1 block, 4 threads per block
    printf("Launching kernel with 1 block, 4 threads:\n");
    hello_kernel<<<1, 4>>>(d_result);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Copy result back ---
    int h_result;   // h_ prefix = host (CPU) memory, convention
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nResult from GPU: %d\n", h_result);

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_result));

    printf("\n=== CUDA hello complete! ===\n");
    return 0;
}
