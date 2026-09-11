CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native

NOBASELINE ?= 0
ifeq ($(NOBASELINE),1)
CXXFLAGS += -DNOBASELINE
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

TARGET := bench
OBJ := build/bench.o build/gemm.o build/assemble.o
HEADERS := src/gemm.h src/assemble.s

.PHONY: all run clean

all: $(TARGET)

build/%.o: src/%.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/assemble.o: src/assemble.s $(HEADERS)
	@mkdir -p build
	$(CXX) $(SME_ASM_FLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@

run: $(TARGET)
	$(TARGET) data/test.in

clean:
	rm -rf build bench