/*
 * 01_thread_indexing.cu — Understanding CUDA thread/block/grid hierarchy
 *
 * The CUDA execution model:
 *
 *   Grid (of blocks)
 *   ┌─────────┬─────────┬─────────┐
 *   │ Block 0 │ Block 1 │ Block 2 │  ← gridDim.x = 3
 *   │ [t0..t3]│ [t0..t3]│ [t0..t3]│  ← blockDim.x = 4
 *   └─────────┴─────────┴─────────┘
 *
 *   Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 *
 *   For Block 1, Thread 2:
 *     global_id = 1 * 4 + 2 = 6
 *
 * Compile: nvcc -arch=sm_89 -o 01_thread_indexing 01_thread_indexing.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* ---------- 1D indexing ---------- */
__global__ void who_am_i_1d() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  Block %d, Thread %d → Global ID %d\n",
           blockIdx.x, threadIdx.x, global_id);
}

/* ---------- 2D indexing (used in matmul) ---------- */
/*
 * For 2D problems (matrices), use 2D blocks:
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Each thread handles one element: C[row][col]
 */
__global__ void who_am_i_2d() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Only the first few print to avoid flood
    if (row < 2 && col < 2) {
        printf("  Block(%d,%d) Thread(%d,%d) → row=%d, col=%d\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col);
    }
}

/* ---------- 3D indexing ---------- */
/*
 * The full 3D hierarchy — rarely needed but good to understand.
 *
 * Each block has a 3D position: (blockIdx.x, blockIdx.y, blockIdx.z)
 * Each thread has a 3D position: (threadIdx.x, threadIdx.y, threadIdx.z)
 *
 * Unique thread within block:
 *   local = threadIdx.z * (blockDim.x * blockDim.y)
 *         + threadIdx.y * blockDim.x
 *         + threadIdx.x
 *
 * Unique block in grid:
 *   block = blockIdx.z * (gridDim.x * gridDim.y)
 *         + blockIdx.y * gridDim.x
 *         + blockIdx.x
 */
__global__ void who_am_i_3d() {
    int local_id = threadIdx.z * (blockDim.x * blockDim.y)
                 + threadIdx.y * blockDim.x
                 + threadIdx.x;

    int block_id = blockIdx.z * (gridDim.x * gridDim.y)
                 + blockIdx.y * gridDim.x
                 + blockIdx.x;

    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int global_id = block_id * threads_per_block + local_id;

    // Only print from a few threads to avoid flood
    if (global_id < 8) {
        printf("  Block(%d,%d,%d) Thread(%d,%d,%d) → local=%d, block=%d, global=%d\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z,
               local_id, block_id, global_id);
    }
}

int main() {
    printf("=== CUDA Thread Indexing ===\n\n");

    /* --- 1D: 3 blocks of 4 threads = 12 total --- */
    printf("1D: <<<3, 4>>> (12 threads):\n");
    who_am_i_1d<<<3, 4>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    /* --- 2D: 2x2 blocks of 2x2 threads = 16 total --- */
    printf("\n2D: <<<dim3(2,2), dim3(2,2)>>> (16 threads):\n");
    dim3 grid2d(2, 2);     // 2x2 grid of blocks
    dim3 block2d(2, 2);    // 2x2 threads per block
    who_am_i_2d<<<grid2d, block2d>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    /* --- 3D: 2x2x1 blocks of 2x2x2 threads = 32 total --- */
    printf("\n3D: <<<dim3(2,2,1), dim3(2,2,2)>>> (32 threads, showing first 8):\n");
    dim3 grid3d(2, 2, 1);
    dim3 block3d(2, 2, 2);
    who_am_i_3d<<<grid3d, block3d>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    /* --- Show the launch config math --- */
    printf("\n=== Launch Configuration Math ===\n");
    printf("For a vector of N=%d elements with BLOCK_SIZE=%d:\n", 10000, 256);
    int N = 10000, BS = 256;
    int num_blocks = (N + BS - 1) / BS;  // CEIL_DIV
    printf("  num_blocks = CEIL_DIV(%d, %d) = %d\n", N, BS, num_blocks);
    printf("  Total threads launched: %d\n", num_blocks * BS);
    printf("  Wasted threads: %d (must bounds-check in kernel!)\n", num_blocks * BS - N);

    printf("\nFor a 1024x1024 matrix with TILE_SIZE=16:\n");
    int M = 1024;
    dim3 grid_example((M + 15) / 16, (M + 15) / 16);
    printf("  grid = (%d, %d), total blocks = %d\n",
           grid_example.x, grid_example.y, grid_example.x * grid_example.y);

    printf("\n=== Thread indexing complete! ===\n");
    return 0;
}
