#ifndef HUOHUA_GEMM_H
#define HUOHUA_GEMM_H

#include <cstddef>
#include <type_traits>

#ifndef A_TYPE
#define A_TYPE __fp16
#endif
#ifndef B_TYPE
#define B_TYPE __fp16
#endif
#ifndef C_TYPE
#define C_TYPE float
#endif
#ifndef PRECISION_NAME
#define PRECISION_NAME "FP16 x FP16 -> FP32"
#endif

static_assert(std::is_same<A_TYPE, __fp16>::value ||
              std::is_same<A_TYPE, float>::value,
              "A_TYPE must be __fp16 or float");
static_assert(std::is_same<B_TYPE, __fp16>::value ||
              std::is_same<B_TYPE, float>::value,
              "B_TYPE must be __fp16 or float");
static_assert(std::is_same<C_TYPE, float>::value,
              "The current SME kernel requires C_TYPE=float");

// Row-major matrices using the configured input types and FP32 output:
// A is N x M, B is M x K, C is N x K.
// A[i * M + j], B[j * K + k], C[i * K + k].
void baseline_gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
                   std::size_t N, std::size_t M, std::size_t K);

// Public entry point for the configured input types and FP32 output.
void gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
          std::size_t N, std::size_t M, std::size_t K);

// Compatibility entry point retained for existing callers.
void gemm_fp16(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
               std::size_t N, std::size_t M, std::size_t K);

// ---------------------------------------------------------------------------
// SME micro-kernel (assembly, see src/assemble.s).
//
// Computes -- and OVERWRITES, it does not accumulate -- one full 32x32 output
// tile. The orchestrator packs the configured inputs into FP32 k-major panels:
//   C(32x32 within leading dim ldc) = sum_k A_panel(k,:)*B_panel(k,:)
//
//   A_panel: element (k, i) at A_panel[k*lda + i], i in [0, 32)
//   B_panel: element (k, j) at B_panel[k*ldb + j], j in [0, 32)
//   C:       row-major, leading dimension ldc
//
// This is the single asm hot kernel; 32x32 is the SME 2x2 ZA32 tile size.
// Non-32x32 (edge) tiles are handled by a scalar fallback in gemm.cpp.
//
// Requirements imposed by the asm (see the header of src/assemble.s):
//   * must be called from a non-streaming context;
//   * streaming vector length must be 512 bit (huohua_sme_svl_bytes() == 64);
//   * the caller's ZA contents are clobbered (ZA is caller-saved in the SME
//     ABI), while PSTATE, v8-v15 and the general registers are preserved.
#if defined(__cplusplus)
extern "C" {
#endif
void huohua_sme_microkernel_32x32(const C_TYPE* A_panel, int lda,
                                  const C_TYPE* B_panel, int ldb,
                                  C_TYPE* C, int ldc, int kc);

// Streaming SVE vector length in bytes (RDSVL; requires FEAT_SME). The
// micro-kernel above is hard-wired to 64 bytes, so use this to select fallback.
unsigned huohua_sme_svl_bytes(void);
#if defined(__cplusplus)
}
#endif

#endif  // HUOHUA_GEMM_H
