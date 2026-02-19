# Phase 1: Matmul Benchmark Results

## RTX 4070 (sm_89, Ada Lovelace)
- FP32 peak: 29.15 TFLOPS
- FP16 peak: 58.3 TFLOPS
- Tensor Core FP16: ~166 TFLOPS (with FP32 accumulate)
- Memory bandwidth: ~504 GB/s

## Full Benchmark Table (GFLOPS)

| Kernel | 256 | 512 | 1024 | 2048 | 4096 |
|--------|-----|-----|------|------|------|
| 1. Naive (16x16) | 1,196 | 1,649 | 1,761 | 1,795 | 1,646 |
| 2. Tiled (32x32 shared mem) | 1,272 | 2,029 | 2,180 | 2,365 | 2,275 |
| 3. Coarsened (4x4/thread) | 1,775 | 5,439 | 8,460 | 10,639 | 9,706 |
| 4. Coarsened (8x8/thread) | 544 | 2,392 | 5,567 | 8,377 | 9,174 |
| 5. cuBLAS FP32 | 2,050 | 7,976 | 15,428 | 17,850 | 21,900 |
| 6. cuBLAS FP16 | 3,947 | 29,454 | 66,156 | 87,770 | 99,503 |

## Percentage of FP32 Peak (at 4096x4096)

| Kernel | % Peak | Speedup vs Naive |
|--------|--------|-----------------|
| 1. Naive | 5.6% | 1.0x |
| 2. Tiled | 7.8% | 1.4x |
| 3. Coarsened 4x4 | 33.3% | 5.9x |
| 4. Coarsened 8x8 | 31.5% | 5.6x |
| 5. cuBLAS FP32 | 75.1% | 13.3x |
| 6. cuBLAS FP16 | 170.7%* | 60.4x |

*FP16 exceeds 100% of FP32 peak because tensor cores double (or more) throughput.

## Key Observations

1. **Naive → Tiled (1.4x):** Shared memory helps, but the improvement is modest because each thread still computes only 1 element. The bottleneck shifts from global memory bandwidth to thread scheduling overhead.

2. **Tiled → Coarsened 4x4 (4.3x at 4096):** The biggest jump. Each thread computing 4×4 = 16 elements dramatically improves arithmetic intensity. Values accumulate in registers (zero latency) instead of shared memory.

3. **Coarsened 4x4 vs 8x8:** At 4096, 4x4 slightly beats 8x8. The 8x8 config uses 128×128 block tiles and 64 registers per thread, causing register pressure. Occupancy drops, reducing the GPU's ability to hide latency.

4. **cuBLAS gap:** Our best kernel (33% peak) vs cuBLAS (75% peak) = ~2.3x gap. cuBLAS achieves more through: double buffering, software pipelining, auto-tuning per matrix size, and hardware-specific assembly.

5. **FP16 tensor cores:** 5x faster than FP32 cuBLAS at 4096. This is what actual LLM training uses. The RTX 4070's tensor cores are the workhorse for training nanochat.
