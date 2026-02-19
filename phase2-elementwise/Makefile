# Phase 2: Elementwise Kernels & Fusion

NVCC    = nvcc
CUFLAGS = -arch=sm_89 -O2

TARGETS = 01_activations 02_rmsnorm 03_softmax

all: $(TARGETS)
	@echo "Built: $(TARGETS)"

%: %.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

run: all
	@for bin in $(TARGETS); do \
		echo ""; \
		echo "════════════ ./$$bin ════════════"; \
		./$$bin; \
	done

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
