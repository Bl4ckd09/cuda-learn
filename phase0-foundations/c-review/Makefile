# Makefile — Compiling mixed C/CUDA projects
#
# Key concepts:
#   - gcc compiles .c files (CPU code)
#   - nvcc compiles .cu files (GPU code, also handles CPU code in .cu)
#   - -arch=sm_89 targets RTX 4070 (Ada Lovelace architecture)
#   - Compute capability table:
#       sm_70 = V100      sm_75 = T4/RTX 2080
#       sm_80 = A100      sm_86 = RTX 3080
#       sm_89 = RTX 4070  sm_90 = H100
#
# Usage:
#   make           — build all
#   make run       — build and run all
#   make clean     — remove binaries
#   make debug     — build with debug flags
#   make 01_pointers — build single target

# --- Compiler settings ---
CC      = gcc
NVCC    = nvcc
CFLAGS  = -Wall -Wextra -O2
CUFLAGS = -arch=sm_89 -O2

# Debug flags (use: make debug)
DEBUG_CFLAGS  = -Wall -Wextra -g -DDEBUG -O0
DEBUG_CUFLAGS = -arch=sm_89 -g -G -DDEBUG -O0
# -g   = host debug info
# -G   = device debug info (for cuda-gdb, disables optimizations)
# -O0  = no optimization (so debugger can step through code)

# --- Source files ---
C_SRCS  = 01_pointers.c 02_structs_and_types.c 03_macros.c
CU_SRCS = 04_cuda_hello.cu

# --- Derive binary names from sources ---
C_BINS  = $(C_SRCS:.c=)
CU_BINS = $(CU_SRCS:.cu=)
ALL_BINS = $(C_BINS) $(CU_BINS)

# --- Default target ---
all: $(ALL_BINS)
	@echo ""
	@echo "Built: $(ALL_BINS)"
	@echo "Run individual: ./01_pointers"
	@echo "Run all: make run"

# --- Pattern rules ---
# Any .c file → binary with same name (strip .c)
%: %.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# Any .cu file → binary with same name (strip .cu)
%: %.cu
	$(NVCC) $(CUFLAGS) -o $@ $<

# --- Convenience targets ---
run: all
	@echo "========================================"
	@echo "Running all exercises..."
	@echo "========================================"
	@for bin in $(C_BINS); do \
		echo ""; \
		echo "────────── ./$$bin ──────────"; \
		./$$bin; \
	done
	@for bin in $(CU_BINS); do \
		echo ""; \
		echo "────────── ./$$bin ──────────"; \
		./$$bin; \
	done

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: CUFLAGS = $(DEBUG_CUFLAGS)
debug: clean all
	@echo "Built with debug flags (-g -G -DDEBUG)"

clean:
	rm -f $(ALL_BINS)
	@echo "Cleaned."

# --- Phony targets (not actual files) ---
.PHONY: all run debug clean

# --- Show compiler info ---
info:
	@echo "CC:      $(CC)"
	@echo "NVCC:    $(NVCC)"
	@echo "CFLAGS:  $(CFLAGS)"
	@echo "CUFLAGS: $(CUFLAGS)"
	@echo "Targets: $(ALL_BINS)"
	@$(NVCC) --version | tail -1
