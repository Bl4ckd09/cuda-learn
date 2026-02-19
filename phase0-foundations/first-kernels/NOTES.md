# Phase 0B+0C: First CUDA Kernels — Notes

## RTX 4070 Benchmark Results

### Vector Add (10M elements, 40 MB per vector)
| Version | Time | Notes |
|---------|------|-------|
| v1 (single block, 1024 elements) | 0.272 ms | Limited to 1024 elements |
| v2 (multi-block) | 0.263 ms | 39063 blocks |
| v3 (grid-stride, 256 blocks) | 0.268 ms | Each thread processes ~153 elements |
| v3 (grid-stride, full blocks) | 0.262 ms | Same performance as v2 |

**Takeaway:** Grid-stride loop matches multi-block performance with far fewer blocks.

### Memory Timing (scale by 2.0 kernel)
| N | H2D | Kernel | D2H | Total | Kernel % |
|---|-----|--------|-----|-------|----------|
| 1K | 0.022 ms | 0.007 ms | 0.032 ms | 0.061 ms | 11% |
| 100K | 0.048 ms | 0.019 ms | 0.091 ms | 0.158 ms | 12% |
| 1M | 0.304 ms | 0.023 ms | 0.383 ms | 0.710 ms | 3% |
| 10M | 3.461 ms | 0.140 ms | 3.392 ms | 6.992 ms | 2% |
| 50M | 15.94 ms | 0.837 ms | 15.98 ms | 32.76 ms | 3% |

**Key insight:** Kernel computation is 2-3% of total time. PCIe transfers dominate.
- PCIe pageable: ~12 GB/s
- PCIe pinned: ~17 GB/s (1.4x faster)
- GPU memory bandwidth: ~530 GB/s (kernel achieved)

### Naive Matmul (C = A @ B, all square)
| Size | Time | GFLOPS | % of 29.1 TFLOPS peak |
|------|------|--------|-----------------------|
| 256 | 0.03 ms | 1321 | 4.5% |
| 512 | 0.16 ms | 1671 | 5.7% |
| 1024 | 1.21 ms | 1769 | 6.1% |
| 2048 | 9.55 ms | 1799 | 6.2% |
| 4096 | 83.25 ms | 1651 | 5.7% |

**Naive matmul = ~6% peak.** Phase 1 goal: tiled matmul → 50%+.

### Atomics
- Non-atomic: 41 / 1,000,000 survived (99.996% lost!)
- Atomic: 1,000,000 / 1,000,000 (correct)
- Histogram (10M elements, 10 bins): 1.77 ms

### Streams
- Sequential 4 chunks: 14.47 ms
- Overlapped 4 streams: 13.59 ms (1.06x speedup)
- Priority range: [-5 (highest), 0 (lowest)]

## Key Patterns Learned

### Grid-stride loop (always use this)
```cuda
__global__ void kernel(float *data, int N) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        data[i] = ...;
    }
}
```

### 2D indexing (for matmul)
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    C[row * N + col] = ...;
}
```

### CUDA timing with events
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## Exercises Completed
1. `01_thread_indexing.cu` — 1D/2D/3D thread hierarchy
2. `02_vector_add.cu` — Single block, multi-block, grid-stride loop
3. `03_memory_timing.cu` — H2D/D2H timing, pinned vs pageable memory
4. `04_naive_matmul.cu` — One thread per output element, ~6% peak
5. `05_nvtx_matmul.cu` — NVTX profiling annotations
6. `06_atomics.cu` — Race conditions, atomicAdd, histogram
7. `07_streams.cu` — Async execution, stream events, priorities
