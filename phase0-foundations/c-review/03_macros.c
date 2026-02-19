/*
 * 03_macros.c — Preprocessor macros and conditional compilation
 *
 * Why this matters for CUDA:
 *   - CUDA error checking uses macros: CHECK_CUDA(cudaMalloc(...))
 *   - Conditional compilation: #ifdef __CUDA_ARCH__ (inside device code)
 *   - Header guards prevent double includes in multi-file projects
 *   - Compile-time constants: #define BLOCK_SIZE 256
 *
 * Compile: gcc -Wall -o 03_macros 03_macros.c
 *     or:  gcc -Wall -DDEBUG -o 03_macros 03_macros.c   (with DEBUG)
 * Run:     ./03_macros
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---------- Compile-time constants ---------- */
/* These become literal values — zero runtime cost */
#define BLOCK_SIZE 256
#define TILE_SIZE 16
#define PI 3.14159265358979f

/* ---------- Macro functions ---------- */
/* Careful: no type checking, args evaluated multiple times */
#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))  /* Used EVERYWHERE in CUDA */

/* ---------- CUDA error checking macro (simulated) ---------- */
/*
 * The real version:
 *   #define CHECK_CUDA(call) do {                          \
 *       cudaError_t err = call;                            \
 *       if (err != cudaSuccess) {                          \
 *           fprintf(stderr, "CUDA error at %s:%d: %s\n",  \
 *                   __FILE__, __LINE__,                    \
 *                   cudaGetErrorString(err));              \
 *           exit(EXIT_FAILURE);                            \
 *       }                                                  \
 *   } while(0)
 *
 * Usage: CHECK_CUDA(cudaMalloc(&d_A, size));
 */
#define CHECK_CALL(call, msg) do {                              \
    int result = call;                                          \
    if (result != 0) {                                          \
        fprintf(stderr, "Error at %s:%d: %s (code %d)\n",      \
                __FILE__, __LINE__, msg, result);               \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* ---------- Conditional compilation ---------- */
/* Set via compiler flag: gcc -DDEBUG ... */
#ifdef DEBUG
    #define DBG_PRINT(fmt, ...) fprintf(stderr, "[DEBUG %s:%d] " fmt "\n", \
                                        __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define DBG_PRINT(fmt, ...) ((void)0)  /* compiles to nothing */
#endif

/* ---------- Header guard pattern ---------- */
/*
 * Every .h file should have:
 *   #ifndef MY_HEADER_H
 *   #define MY_HEADER_H
 *   ... declarations ...
 *   #endif
 */

int main() {
    printf("=== Preprocessor Macros for CUDA ===\n\n");

    /* --- Constants --- */
    printf("Constants:\n");
    printf("  BLOCK_SIZE = %d\n", BLOCK_SIZE);
    printf("  TILE_SIZE  = %d\n", TILE_SIZE);
    printf("  PI         = %.6f\n\n", PI);

    /* --- CEIL_DIV: the CUDA grid-sizing formula --- */
    /*
     * When launching a kernel for N elements with BLOCK_SIZE threads:
     *   int num_blocks = CEIL_DIV(N, BLOCK_SIZE);
     *   kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, N);
     *
     * This ensures enough blocks to cover all N elements,
     * even when N isn't divisible by BLOCK_SIZE.
     */
    printf("CEIL_DIV (grid sizing formula):\n");
    int test_sizes[] = {100, 256, 1000, 1024, 10000};
    for (int i = 0; i < 5; i++) {
        int N = test_sizes[i];
        int num_blocks = CEIL_DIV(N, BLOCK_SIZE);
        printf("  N=%5d, blocks=%3d, threads_launched=%5d, wasted=%3d\n",
               N, num_blocks, num_blocks * BLOCK_SIZE,
               num_blocks * BLOCK_SIZE - N);
    }

    /* --- 2D grid sizing (for matmul) --- */
    printf("\n2D grid sizing (1024x1024 matmul):\n");
    int M = 1024, N = 1024;
    int grid_x = CEIL_DIV(N, TILE_SIZE);
    int grid_y = CEIL_DIV(M, TILE_SIZE);
    printf("  Matrix: %dx%d, Tile: %dx%d\n", M, N, TILE_SIZE, TILE_SIZE);
    printf("  Grid: (%d, %d), Total blocks: %d\n\n", grid_x, grid_y, grid_x * grid_y);

    /* --- MIN/MAX --- */
    printf("MIN/MAX:\n");
    printf("  MIN(3, 7) = %d, MAX(3, 7) = %d\n\n", MIN(3, 7), MAX(3, 7));

    /* --- Error checking macro --- */
    printf("Error checking macro:\n");
    int simulated_success = 0;
    CHECK_CALL(simulated_success, "simulated CUDA call");
    printf("  CHECK_CALL passed (returned 0 = success)\n");
    printf("  If it failed, would print: Error at %s:%d: ...\n\n", __FILE__, __LINE__);

    /* --- Debug printing --- */
    printf("Debug printing (compile with -DDEBUG to see):\n");
    DBG_PRINT("This only prints in debug mode, x=%d", 42);
#ifdef DEBUG
    printf("  DEBUG is defined — you should see debug output above\n\n");
#else
    printf("  DEBUG is not defined — recompile with: gcc -DDEBUG ...\n\n");
#endif

    /* --- Predefined macros --- */
    printf("Predefined macros:\n");
    printf("  __FILE__: %s\n", __FILE__);
    printf("  __LINE__: %d\n", __LINE__);
    printf("  __DATE__: %s\n", __DATE__);
    printf("  __TIME__: %s\n", __TIME__);
    printf("  (In CUDA: __CUDA_ARCH__ is defined inside device code)\n\n");

    printf("=== All macro exercises complete! ===\n");
    return 0;
}
