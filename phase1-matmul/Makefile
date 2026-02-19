# Phase 1: Matmul Optimization Ladder

NVCC    = nvcc
CUFLAGS = -arch=sm_89 -O2
CUBLAS  = -lcublas

TARGETS = 01_naive 02_tiled 03_coarsened 04_vectorized 05_cublas benchmark_all

all: $(TARGETS)
	@echo "Built: $(TARGETS)"

01_naive: 01_naive.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

02_tiled: 02_tiled.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

03_coarsened: 03_coarsened.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

04_vectorized: 04_vectorized.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

05_cublas: 05_cublas.cu common.h
	$(NVCC) $(CUFLAGS) $(CUBLAS) -o $@ $<

benchmark_all: benchmark_all.cu common.h
	$(NVCC) $(CUFLAGS) $(CUBLAS) -o $@ $<

bench: benchmark_all
	./benchmark_all

run: all
	@for bin in $(TARGETS); do \
		echo ""; \
		echo "════════════ ./$$bin ════════════"; \
		./$$bin; \
	done

clean:
	rm -f $(TARGETS)

.PHONY: all bench run clean
