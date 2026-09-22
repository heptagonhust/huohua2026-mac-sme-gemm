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

constexpr bool kHalfToSingle =
    std::is_same<A_TYPE, __fp16>::value &&
    std::is_same<B_TYPE, __fp16>::value &&
    std::is_same<C_TYPE, float>::value;
constexpr bool kSingleToSingle =
    std::is_same<A_TYPE, float>::value &&
    std::is_same<B_TYPE, float>::value &&
    std::is_same<C_TYPE, float>::value;
constexpr bool kDoubleToDouble =
    std::is_same<A_TYPE, double>::value &&
    std::is_same<B_TYPE, double>::value &&
    std::is_same<C_TYPE, double>::value;
static_assert(kHalfToSingle || kSingleToSingle || kDoubleToDouble,
              "Unsupported GEMM precision combination");

// Row-major matrices using the configured input and output types:
// A is N x M, B is M x K, C is N x K.
void baseline_gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
                   std::size_t N, std::size_t M, std::size_t K);

void gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
          std::size_t N, std::size_t M, std::size_t K);

// Compatibility entry point retained for existing FP32-output callers.
void gemm_fp16(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
               std::size_t N, std::size_t M, std::size_t K);

#if defined(__cplusplus)
extern "C" {
#endif

// FP32 packed panels, four ZA32 tiles, 32x32 output tile.
void huohua_sme_microkernel_f32_32x32(const float* A_panel, int lda,
                                      const float* B_panel, int ldb,
                                      float* C, int ldc, int kc);

// FP64 packed panels, four ZA64 tiles, 16x16 output tile.
void huohua_sme_microkernel_f64_16x16(const double* A_panel, int lda,
                                      const double* B_panel, int ldb,
                                      double* C, int ldc, int kc);

// Streaming SVE vector length in bytes (RDSVL; requires FEAT_SME).
unsigned huohua_sme_svl_bytes(void);

#if defined(__cplusplus)
}
#endif

#endif  // HUOHUA_GEMM_H
