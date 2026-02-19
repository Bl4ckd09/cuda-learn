# Makefile for Phase 0B + 0C: First CUDA Kernels
#
# All targets are .cu files compiled with nvcc
# RTX 4070 = Ada Lovelace = sm_89

NVCC    = nvcc
CUFLAGS = -arch=sm_89 -O2

# NVTX3 is header-only in modern CUDA (no library to link)
NVTX_FLAGS =

TARGETS = 01_thread_indexing 02_vector_add 03_memory_timing \
          04_naive_matmul 05_nvtx_matmul 06_atomics 07_streams

all: $(TARGETS)
	@echo "Built: $(TARGETS)"

# Standard CUDA targets
%: %.cu
	$(NVCC) $(CUFLAGS) -o $@ $<

# NVTX matmul needs extra library
05_nvtx_matmul: 05_nvtx_matmul.cu
	$(NVCC) $(CUFLAGS) $(NVTX_FLAGS) -o $@ $<

# Run all
run: all
	@for bin in $(TARGETS); do \
		echo ""; \
		echo "════════════ ./$$bin ════════════"; \
		./$$bin; \
	done

# Run individual phases
run-0b: 01_thread_indexing 02_vector_add 03_memory_timing 04_naive_matmul
	@for bin in 01_thread_indexing 02_vector_add 03_memory_timing 04_naive_matmul; do \
		echo ""; \
		echo "════════════ ./$$bin ════════════"; \
		./$$bin; \
	done

run-0c: 05_nvtx_matmul 06_atomics 07_streams
	@for bin in 05_nvtx_matmul 06_atomics 07_streams; do \
		echo ""; \
		echo "════════════ ./$$bin ════════════"; \
		./$$bin; \
	done

clean:
	rm -f $(TARGETS)
	rm -f *.qdrep *.nsys-rep *.sqlite
	@echo "Cleaned."

.PHONY: all run run-0b run-0c clean
