CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native

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

# C++ is kept without +sme so scalar loops cannot be auto-vectorized into
# non-streaming SVE. Each assembly kernel receives its exact SME ISA features.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
SME_F32_FLAGS := -march=armv9-a+sme2
SME_F64_FLAGS := -march=armv9-a+sme2+sme-f64f64
else
SME_F32_FLAGS :=
SME_F64_FLAGS :=
endif

HEADERS := src/gemm.h
ASM_SOURCES := src/assemble_f32.s src/assemble_f64.s
HALF_FLAGS := -DA_TYPE=__fp16 -DB_TYPE=__fp16 -DC_TYPE=float \
              -DPRECISION_NAME='"FP16 x FP16 -> FP32"'
SINGLE_FLAGS := -DA_TYPE=float -DB_TYPE=float -DC_TYPE=float \
                -DPRECISION_NAME='"FP32 x FP32 -> FP32"'
DOUBLE_FLAGS := -DA_TYPE=double -DB_TYPE=double -DC_TYPE=double -DHUOHUA_FP64 \
                -DPRECISION_NAME='"FP64 x FP64 -> FP64"'

HALF_OBJ := build/half_2_single/bench.o \
            build/half_2_single/gemm.o \
            build/half_2_single/assemble_f32.o
SINGLE_OBJ := build/single_2_single/bench.o \
              build/single_2_single/gemm.o \
              build/single_2_single/assemble_f32.o
DOUBLE_OBJ := build/double_2_double/bench.o \
              build/double_2_double/gemm.o \
              build/double_2_double/assemble_f32.o \
              build/double_2_double/assemble_f64.o

.PHONY: all clean run_half_2_single run_single_2_single run_double_2_double

all: bench_half_2_single bench_single_2_single bench_double_2_double

bench_half_2_single: $(HALF_OBJ)
	$(CXX) $(HALF_OBJ) $(LDFLAGS) -o $@

bench_single_2_single: $(SINGLE_OBJ)
	$(CXX) $(SINGLE_OBJ) $(LDFLAGS) -o $@

bench_double_2_double: $(DOUBLE_OBJ)
	$(CXX) $(DOUBLE_OBJ) $(LDFLAGS) -o $@

build/half_2_single/%.o: src/%.cpp $(HEADERS) $(ASM_SOURCES)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(HALF_FLAGS) -c $< -o $@

build/single_2_single/%.o: src/%.cpp $(HEADERS) $(ASM_SOURCES)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SINGLE_FLAGS) -c $< -o $@

build/double_2_double/%.o: src/%.cpp $(HEADERS) $(ASM_SOURCES)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DOUBLE_FLAGS) -c $< -o $@

build/half_2_single/assemble_f32.o: src/assemble_f32.s
	@mkdir -p $(@D)
	$(CXX) $(SME_F32_FLAGS) -c $< -o $@

build/single_2_single/assemble_f32.o: src/assemble_f32.s
	@mkdir -p $(@D)
	$(CXX) $(SME_F32_FLAGS) -c $< -o $@

build/double_2_double/assemble_f32.o: src/assemble_f32.s
	@mkdir -p $(@D)
	$(CXX) $(SME_F32_FLAGS) -c $< -o $@

build/double_2_double/assemble_f64.o: src/assemble_f64.s
	@mkdir -p $(@D)
	$(CXX) $(SME_F64_FLAGS) -c $< -o $@

run_half_2_single: bench_half_2_single
	./bench_half_2_single data/test.in

run_single_2_single: bench_single_2_single
	./bench_single_2_single data/test.in

run_double_2_double: bench_double_2_double
	./bench_double_2_double data/test.in

clean:
	rm -rf build bench_half_2_single bench_single_2_single bench_double_2_double
