# Phase 3: Attention Kernels

NVCC    = nvcc
CUFLAGS = -arch=sm_89 -O2

TARGETS = 01_naive_attention 02_fused_attention 03_flash_attention

all: $(TARGETS)
	@echo "Built: $(TARGETS)"

01_naive_attention: 01_naive_attention.cu common.h
	$(NVCC) $(CUFLAGS) -lcublas -o $@ $<

02_fused_attention: 02_fused_attention.cu common.h
	$(NVCC) $(CUFLAGS) -o $@ $<

03_flash_attention: 03_flash_attention.cu common.h
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
