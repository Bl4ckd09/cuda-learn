# CUDA Learning Insights

A collection of key insights from the learning journey — useful for revision.

---

## Phase 0A: C/C++ Foundations

### RTX 4070 Hardware Profile (sm_89, Ada Lovelace)
- **46 SMs** with max 1024 threads/block → up to 47,104 concurrent threads
- **48 KB shared memory per block** — this is the fast on-chip SRAM you'll use for tiled matmul and Flash Attention
- **Warp size 32** — this is the fundamental execution unit. 32 threads execute in lockstep (SIMT). All reduction operations and shuffle instructions work at warp granularity.
- **12.9 GB global memory** — enough for your 700M parameter nanochat model (~2.8 GB in FP32)

### Phase 0A Design Philosophy
These C exercises aren't just "review" — they build the exact mental model you need for CUDA. Every CUDA kernel operates through pointers (`cudaMalloc` returns `void*`), every memory transfer is `memcpy`-style, and understanding how `malloc`/`free` map to `cudaMalloc`/`cudaFree` is the foundation. The Makefile exercise teaches `nvcc` compilation flags you'll use for every kernel going forward.

---

## Phase 0B: First CUDA Kernels

### The CUDA Execution Model
- A **kernel** is a function that runs on the GPU, executed by thousands of threads simultaneously
- Threads are organized into **blocks** (up to 1024 threads), blocks form a **grid**
- Each thread knows its identity via built-in variables: `threadIdx.x`, `blockIdx.x`, `blockDim.x`
- The global thread index `idx = blockIdx.x * blockDim.x + threadIdx.x` maps each thread to its data element
- **Grid-stride loops** let a fixed number of threads process arbitrarily large data — the most robust pattern

### Memory Timing: The Critical Truth
At N=10M, the kernel takes only 0.14ms (2% of total time!) while PCIe transfers take 6.85ms. The kernel achieves 533 GB/s bandwidth — exceeding the 504 GB/s spec due to cache effects. PCIe tops out at ~12 GB/s pageable, ~17 GB/s pinned — a 30x gap vs GPU memory. This is exactly why PyTorch's `.to('cuda')` is a one-time cost and tensors stay on GPU throughout training.

---

## Phase 1: Matrix Multiplication

### Why Tiled Matmul is THE Key Optimization
The naive matmul reads 2K global memory values per output element (one row of A + one column of B). For K=4096, that's 32KB of reads for one multiply-add. But adjacent threads in the same block need the SAME rows/columns! By loading a tile into shared memory (48KB on your RTX 4070), all threads in the block can reuse it. This converts global memory reads from O(K) per element to O(K/TILE_SIZE) per element — a TILE_SIZE-fold reduction. This is the single most important GPU optimization pattern.

### Arithmetic Intensity: The Roofline Model
- **Naive matmul:** 2K FLOPs per output element, 2K×4 bytes read → 0.5 FLOP/byte → bandwidth-bound
- **RTX 4070 balance point:** 29.15 TFLOPS / 504 GB/s ≈ 58 FLOP/byte needed to be compute-bound
- **Tiled matmul (TILE=16):** reduces memory reads by 16x → ~8 FLOP/byte → still bandwidth-bound but much better
- **Goal:** Increase arithmetic intensity until kernel becomes compute-bound, not memory-bound

### cuBLAS Column-Major Layout
cuBLAS expects column-major (Fortran-style) layout by default. Row-major `A[i][j]` is stored at `A[i * N + j]`; column-major at `A[j * M + i]`. You can avoid transposing by using the `CUBLAS_OP_T` flag.

### Thread Coarsening: More Work Per Thread
Instead of each thread computing 1 output element, have each thread compute a TM×TN tile (e.g., 8×8). Benefits:
- More arithmetic per thread → better amortizes overhead
- Fewer threads needed → less scheduling overhead
- Values stay in registers (fastest memory) longer
- Register pressure becomes the limiting factor

### Vectorized Loads (float4)
GPU memory transactions are 128 bits (16 bytes). Loading a single `float` (4 bytes) wastes 75% of the transaction. Loading `float4` (16 bytes) uses the full transaction width. This gives ~1.5-2x bandwidth improvement for memory-bound kernels.

### Phase 1 Benchmark Results (RTX 4070, 4096x4096)
- **Naive:** 1,646 GFLOPS (5.6% peak) — global memory bottleneck
- **Tiled (shared mem):** 2,275 GFLOPS (7.8%) — 1.4x speedup, shared memory helps but tile-per-thread is still 1:1
- **Coarsened 4x4:** 9,706 GFLOPS (33.3%) — **5.9x over naive!** Register tiling is the big win
- **Coarsened 8x8:** 9,174 GFLOPS (31.5%) — slightly slower than 4x4 due to register pressure at this config
- **cuBLAS FP32:** 21,900 GFLOPS (75.1%) — 13.3x over naive, our coarsened kernel is ~44% of cuBLAS
- **cuBLAS FP16:** 99,503 GFLOPS (170.7% of FP32 peak) — tensor cores in action!

The coarsened kernel is the sweet spot for hand-written CUDA — getting to ~33% peak with a readable kernel. cuBLAS gets to 75% through auto-tuned assembly and hardware-specific tricks.

### Row-Major cuBLAS Trick
To compute `C = A @ B` in row-major with cuBLAS (which expects column-major):
```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
// Swap A↔B and swap M↔N in dimensions. That's it!
```

---

## General GPU Programming Principles

### Memory Hierarchy (fastest to slowest)
1. **Registers** — per-thread, ~256KB per SM, no latency
2. **Shared memory** — per-block, 48KB, ~20-30 cycles latency
3. **L1 cache** — per-SM, shares hardware with shared memory
4. **L2 cache** — shared across all SMs, ~200 cycles
5. **Global memory (GDDR6X)** — 12.9 GB, ~400-600 cycles
6. **PCIe (CPU↔GPU)** — ~12-17 GB/s, milliseconds for large transfers

### Coalesced Memory Access
When threads in a warp (32 threads) access adjacent memory addresses, the hardware combines them into a single wide transaction. This is "coalesced access" and is critical for bandwidth. Thread `i` should access `data[i]`, not `data[i * stride]`.

### The Three Pillars of GPU Performance
1. **Expose parallelism** — launch enough threads to keep all SMs busy
2. **Minimize memory traffic** — use shared memory, registers; reduce global reads
3. **Coalesce memory access** — adjacent threads read adjacent addresses
