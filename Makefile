# ============================================================
#  EE451 Ray Tracer — Makefile
#  Targets: make serial | make openmp | make cuda | make all
# ============================================================

CXX       = g++
NVCC      = nvcc

CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra -Iinclude
OMPFLAGS  = $(CXXFLAGS) -fopenmp

# Multi-arch PTX/SASS for CARC GPUs:
#   sm_61 = P100, sm_70 = V100, sm_80 = A100, sm_86 = A40/RTX3xxx
# -rdc=true (relocatable device code) is REQUIRED for device virtual functions
# spanning multiple translation units (DeviceSphere/DeviceQuad vtables).
# NVCCFLAGS = -O3 -std=c++17 -Iinclude -rdc=true \
#             --generate-code arch=compute_61,code=sm_61 \
#             --generate-code arch=compute_70,code=sm_70 \
#             --generate-code arch=compute_80,code=sm_80 \
#             --generate-code arch=compute_86,code=sm_86
#
#
NVCCFLAGS = -O3 -std=c++17 -Iinclude -rdc=true --generate-code arch=compute_86,code=sm_86

LDCUDA    = -lcurand

BINDIR    = bin
OUTDIR    = output

# Source files
SERIAL_SRC   = src/serial/main.cpp
OPENMP_SRC   = src/openmp/main.cpp
CUDA_SRCS    = src/cuda/main.cu \
               src/cuda/render.cu \
               src/cuda/device_scene.cu

# Headers (used as dependencies)
HEADERS = $(wildcard include/*.h) $(wildcard scenes/*.h) \
          src/cuda/render.cuh src/cuda/device_scene.cuh

.PHONY: all serial openmp cuda clean

all: serial openmp cuda

serial: $(BINDIR)/serial_rt
$(BINDIR)/serial_rt: $(SERIAL_SRC) $(HEADERS)
	@mkdir -p $(BINDIR) $(OUTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SERIAL_SRC)
	@echo "Built: $@"

openmp: $(BINDIR)/openmp_rt
$(BINDIR)/openmp_rt: $(OPENMP_SRC) $(HEADERS)
	@mkdir -p $(BINDIR) $(OUTDIR)
	$(CXX) $(OMPFLAGS) -o $@ $(OPENMP_SRC)
	@echo "Built: $@"

cuda: $(BINDIR)/cuda_rt
$(BINDIR)/cuda_rt: $(CUDA_SRCS) $(HEADERS)
	@mkdir -p $(BINDIR) $(OUTDIR)
	$(NVCC) $(NVCCFLAGS) $(LDCUDA) -o $@ $(CUDA_SRCS)
	@echo "Built: $@"

clean:
	rm -rf $(BINDIR)

# Quick test targets (small render to verify correctness)
test-serial: serial
	./$(BINDIR)/serial_rt --scene random --width 200 --spp 10 --depth 10 \
	                       --output $(OUTDIR)/test_serial.ppm
	@echo "Output: $(OUTDIR)/test_serial.ppm"

test-openmp: openmp
	OMP_NUM_THREADS=4 ./$(BINDIR)/openmp_rt --scene random --width 200 --spp 10 \
	                  --depth 10 --output $(OUTDIR)/test_openmp.ppm
	@echo "Output: $(OUTDIR)/test_openmp.ppm"

test-cuda: cuda
	./$(BINDIR)/cuda_rt --scene random --width 1920 --spp 500 --depth 10 \
	                    --output $(OUTDIR)/test_cuda.ppm
	@echo "Output: $(OUTDIR)/test_cuda.ppm"

test-all: test-serial test-openmp test-cuda
