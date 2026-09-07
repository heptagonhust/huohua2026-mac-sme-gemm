#include "gemm.h"

#include <cstddef>
#include <cstdlib>

namespace {

// One SME output micro-tile is 32 x 32 (four 16x16 ZA32 tiles).
constexpr std::size_t kTile = 32;
// L2 budget for one RHS column band (paper formula 5).
constexpr std::size_t kL2Bytes = 12u * 1024u * 1024u;

std::size_t align_down(std::size_t value, std::size_t alignment) {
    return value - (value % alignment);
}

// Column-band width along the output-column dimension (paper formula 5).
// RHS is M x K in this project's naming, so the banded side is K and the
// inner (reused) side is M; each FP32 element is 4 bytes.
std::size_t compute_nc(std::size_t N, std::size_t M, std::size_t K) {
    (void)N;
    if (K <= kTile) {
        return K;
    }
    std::size_t ideal = kL2Bytes / (kTile * (M > 0 ? M : 1) * sizeof(float));
    if (ideal < kTile) {
        ideal = kTile;
    }
    if (ideal > K) {
        ideal = K;
    }
    std::size_t nc = align_down(ideal, kTile);
    return nc < kTile ? kTile : nc;
}

// Pack one A row panel (mr rows x full inner M) into a k-major FP32 panel.
// Input is FP16 (paper); values are widened to FP32 for the current kernel.
// A_panel[k * lda + i] = (float)A[(row0 + i) * M + k], lda == kTile.
void pack_a_panel(const __fp16* A, float* A_panel, std::size_t row0,
                  std::size_t mr, std::size_t M, std::size_t lda) {
    for (std::size_t k = 0; k < M; ++k) {
        for (std::size_t i = 0; i < mr; ++i) {
            A_panel[k * lda + i] = static_cast<float>(A[(row0 + i) * M + k]);
        }
    }
}

// Pack one RHS column band (full inner M x ncols columns) into k-major FP32.
// B_panel[k * ldb + j] = (float)B[k * K + (col0 + j)].
void pack_b_band(const __fp16* B, float* B_panel, std::size_t col0,
                 std::size_t ncols, std::size_t M, std::size_t K,
                 std::size_t ldb) {
    for (std::size_t k = 0; k < M; ++k) {
        const __fp16* b_row = B + k * K + col0;
        for (std::size_t j = 0; j < ncols; ++j) {
            B_panel[k * ldb + j] = static_cast<float>(b_row[j]);
        }
    }
}

// Scalar fallback for the edge tiles (mr<32 or nr<32), and for non-aarch64
// builds where the SME asm kernel is not linked.
void scalar_microkernel(const float* A_panel, int lda,
                        const float* B_panel, int ldb,
                        float* C, int ldc,
                        int mr, int nr, int kc) {
    for (int k = 0; k < kc; ++k) {
        const float* ap = A_panel + static_cast<std::size_t>(k) * lda;
        const float* bp = B_panel + static_cast<std::size_t>(k) * ldb;
        for (int i = 0; i < mr; ++i) {
            const float a = ap[i];
            float* crow = C + static_cast<std::size_t>(i) * ldc;
            for (int j = 0; j < nr; ++j) {
                crow[j] += a * bp[j];
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// C orchestrator: L2 column bands + row panels + k-major packing.
//   A: N x M, B: M x K (FP16 inputs), C: N x K (FP32 output, row-major)
//
// The actual arithmetic of every full 32x32 tile is delegated to the SME
// FMOPA micro-kernel in src/assemble.s (huohua_sme_microkernel_32x32).
// ---------------------------------------------------------------------------
void gemm_fp16(const __fp16* A, const __fp16* B, float* C,
               std::size_t N, std::size_t M, std::size_t K) {
    if (N == 0 || M == 0 || K == 0) {
        return;
    }

    const std::size_t Nc = compute_nc(N, M, K);
    const std::size_t lda = kTile;               // A panel width (rows per panel)
    const std::size_t ldb = Nc;                  // B band width (columns)
    const int ldc = static_cast<int>(K);

    // Buffers: one A panel (kTile x M) + one B band (M x Nc).
    float* A_panel = static_cast<float*>(std::malloc(M * lda * sizeof(float)));
    float* B_band = static_cast<float*>(std::malloc(M * ldb * sizeof(float)));
    if (A_panel == nullptr || B_band == nullptr) {
        std::free(A_panel);
        std::free(B_band);
        return;
    }

    for (std::size_t col0 = 0; col0 < K; col0 += Nc) {
        const std::size_t ncols = (col0 + Nc <= K) ? Nc : (K - col0);
        pack_b_band(B, B_band, col0, ncols, M, K, ldb);

        for (std::size_t row0 = 0; row0 < N; row0 += kTile) {
            const std::size_t mr = (row0 + kTile <= N) ? kTile : (N - row0);
            pack_a_panel(A, A_panel, row0, mr, M, lda);

            for (std::size_t j0 = 0; j0 < ncols; j0 += kTile) {
                const std::size_t nr = (j0 + kTile <= ncols) ? kTile : (ncols - j0);
                float* ctile = C + row0 * K + col0 + j0;

                if (mr == kTile && nr == kTile) {
#if defined(__aarch64__)
                    huohua_sme_microkernel_32x32(
                        A_panel, static_cast<int>(lda),
                        B_band + j0, static_cast<int>(ldb),
                        ctile, ldc, static_cast<int>(M));
#else
                    scalar_microkernel(A_panel, static_cast<int>(lda),
                                        B_band + j0, static_cast<int>(ldb),
                                        ctile, ldc,
                                        static_cast<int>(mr), static_cast<int>(nr),
                                        static_cast<int>(M));
#endif
                } else {
                    scalar_microkernel(A_panel, static_cast<int>(lda),
                                       B_band + j0, static_cast<int>(ldb),
                                       ctile, ldc,
                                       static_cast<int>(mr), static_cast<int>(nr),
                                       static_cast<int>(M));
                }
            }
        }
    }

    std::free(A_panel);
    std::free(B_band);
}
