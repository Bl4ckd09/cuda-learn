# Phase 0A: C/C++ Crash Course — Notes

## Key Patterns for CUDA

### Memory Model (CPU ↔ GPU)
```
CPU (Host)                          GPU (Device)
─────────                           ─────────
float *h_A = malloc(size)           float *d_A; cudaMalloc(&d_A, size)
h_A[i] = value                     (can't access d_A from CPU!)
                  cudaMemcpy(d_A, h_A, size, H2D)
                  kernel<<<grid,block>>>(d_A, N)
                  cudaMemcpy(h_A, d_A, size, D2H)
free(h_A)                           cudaFree(d_A)
```

### Naming Convention
- `h_` prefix = host (CPU) memory
- `d_` prefix = device (GPU) memory

### CEIL_DIV — Used Everywhere
```c
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

int num_blocks = CEIL_DIV(N, BLOCK_SIZE);
kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, N);
```

### Row-Major 2D Indexing
Matrices in CUDA are flat 1D arrays:
```c
// matrix[row][col] becomes:
matrix[row * num_cols + col]
```

### Error Checking Macro
```c
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)
```

## RTX 4070 Specs (sm_89)
- Compute capability: 8.9 (Ada Lovelace)
- SMs: 46
- Max threads/block: 1024
- Shared memory/block: 48 KB
- Global memory: 12.9 GB
- Warp size: 32

## Exercises Completed
1. `01_pointers.c` — Pointer basics, double pointers, arithmetic, malloc/free, void*, function pointers
2. `02_structs_and_types.c` — sizeof, structs, typedef, alignment/padding
3. `03_macros.c` — Constants, CEIL_DIV, error checking, conditional compilation
4. `04_cuda_hello.cu` — First CUDA kernel, device query, CHECK_CUDA macro
5. `Makefile` — Mixed C/CUDA compilation with gcc + nvcc
