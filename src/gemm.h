#ifndef HUOHUA_GEMM_H
#define HUOHUA_GEMM_H

#include <cstddef>

// Row-major matrices (FP16 inputs, FP32 accumulate & output, per paper):
// A is N x M, B is M x K, C is N x K.
// A[i * M + j], B[j * K + k], C[i * K + k].  Inputs are stored as __fp16.
void baseline_gemm(const __fp16* A, const __fp16* B, float* C,
                   std::size_t N, std::size_t M, std::size_t K);

// Public entry point (FP16 input -> FP32 output, paper FP16->FP32 path).
void gemm_fp16(const __fp16* A, const __fp16* B, float* C,
               std::size_t N, std::size_t M, std::size_t K);

// ---------------------------------------------------------------------------
// SME micro-kernel (assembly, see src/assemble.s).
//
// Accumulates one full 32x32 output tile.  The C orchestrator packs FP16
// inputs into FP32 k-major panels before calling this kernel:
//   C(32x32 within leading dim ldc) += sum_k A_panel(k,:)*B_panel(k,:)
//
//   A_panel: element (k, i) at A_panel[k*lda + i], i in [0, 32)
//   B_panel: element (k, j) at B_panel[k*ldb + j], j in [0, 32)
//   C:       row-major, leading dimension ldc
//
// This is the single asm hot kernel; 32x32 is the SME 2x2 ZA32 tile size.
// Non-32x32 (edge) tiles are handled by a scalar fallback in gemm.cpp.
#if defined(__cplusplus)
extern "C" {
#endif
void huohua_sme_microkernel_32x32(const float* A_panel, int lda,
                                  const float* B_panel, int ldb,
                                  float* C, int ldc, int kc);
#if defined(__cplusplus)
}
#endif

#endif  // HUOHUA_GEMM_H
