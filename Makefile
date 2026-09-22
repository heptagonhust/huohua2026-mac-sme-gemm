CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native

# The double-buffered RHS band pipeline runs a packing producer thread,
# so compiling and linking both need the thread runtime.
CXXFLAGS += -pthread
LDFLAGS += -pthread

NOBASELINE ?= 0
ifeq ($(NOBASELINE),1)
CXXFLAGS += -DNOBASELINE
endif

DEBUG ?= 0
ifeq ($(DEBUG),1)
CXXFLAGS += -g
endif

# assemble.s is the SME micro-kernel and must be built with the SME ISA.
# C++ is kept without +sme so the compiler never auto-vectorizes scalar loops
# into non-streaming SVE (which faults on Apple M4 outside streaming mode).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
SME_ASM_FLAGS := -march=armv9-a+sme2
else
SME_ASM_FLAGS :=
endif

HEADERS := src/gemm.h src/assemble.s
HALF_FLAGS := -DA_TYPE=__fp16 -DB_TYPE=__fp16 -DC_TYPE=float \
              -DPRECISION_NAME='"FP16 x FP16 -> FP32"'
SINGLE_FLAGS := -DA_TYPE=float -DB_TYPE=float -DC_TYPE=float \
                -DPRECISION_NAME='"FP32 x FP32 -> FP32"'

HALF_OBJ := build/half_2_single/bench.o \
            build/half_2_single/gemm.o \
            build/half_2_single/assemble.o
SINGLE_OBJ := build/single_2_single/bench.o \
              build/single_2_single/gemm.o \
              build/single_2_single/assemble.o

.PHONY: all clean run_half_2_single run_single_2_single

all: bench_half_2_single bench_single_2_single

bench_half_2_single: $(HALF_OBJ)
	$(CXX) $(HALF_OBJ) $(LDFLAGS) -o $@

bench_single_2_single: $(SINGLE_OBJ)
	$(CXX) $(SINGLE_OBJ) $(LDFLAGS) -o $@

build/half_2_single/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(HALF_FLAGS) -c $< -o $@

build/single_2_single/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SINGLE_FLAGS) -c $< -o $@

build/half_2_single/assemble.o: src/assemble.s $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(SME_ASM_FLAGS) -c $< -o $@

build/single_2_single/assemble.o: src/assemble.s $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(SME_ASM_FLAGS) -c $< -o $@

run_half_2_single: bench_half_2_single
	./bench_half_2_single data/test.in

run_single_2_single: bench_single_2_single
	./bench_single_2_single data/test.in

clean:
	rm -rf build bench_half_2_single bench_single_2_single
