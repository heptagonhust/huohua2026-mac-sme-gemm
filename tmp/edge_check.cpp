// Temporary verification probe: NaN-aware comparison of gemm_fp16 against the
// reference baseline_gemm, including shapes whose dimensions are not a
// multiple of 32 (i.e. shapes that exercise the edge-tile scalar path).
//
// build: clang++ -std=c++17 -O2 -march=native -Isrc tmp/edge_check.cpp [gemm.cpp] tmp/asm.o -o ...
#include "gemm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

void baseline_gemm(const __fp16* A, const __fp16* B, float* C,
                   std::size_t N, std::size_t M, std::size_t K) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < M; ++k) {
                sum += static_cast<float>(A[i * M + k]) *
                       static_cast<float>(B[k * K + j]);
            }
            C[i * K + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100;
    const std::size_t M = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100;
    const std::size_t K = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 100;

    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<__fp16> A(N * M), B(M * K);
    for (__fp16& v : A) {
        v = static_cast<__fp16>(distribution(generator));
    }
    for (__fp16& v : B) {
        v = static_cast<__fp16>(distribution(generator));
    }

    std::vector<float> reference(N * K, 0.0f);
    // Same "garbage" initialisation bench.cpp uses, so any accidental
    // accumulation into C shows up as NaN instead of hiding.
    std::vector<float> result(N * K, std::numeric_limits<float>::quiet_NaN());

    baseline_gemm(A.data(), B.data(), reference.data(), N, M, K);
    gemm_fp16(A.data(), B.data(), result.data(), N, M, K);

    std::size_t nan_count = 0;
    std::size_t mismatch = 0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (std::isnan(result[i])) {
            ++nan_count;
            continue;
        }
        const double diff = std::fabs(static_cast<double>(result[i]) -
                                      static_cast<double>(reference[i]));
        const double scale = std::max(1.0, std::fabs((double)reference[i]));
        if (diff > 1.0e-3 * scale) {
            ++mismatch;
        }
        max_abs = std::max(max_abs, diff);
    }

    std::printf("N=%zu M=%zu K=%zu  edges=%s -> NaN:%zu mismatched:%zu max|diff|=%.3g  %s\n",
                N, M, K, (N % 32 || M % 32 || K % 32) ? "yes" : "no ",
                nan_count, mismatch, max_abs,
                (nan_count == 0 && mismatch == 0) ? "PASS" : "FAIL");
    return (nan_count == 0 && mismatch == 0) ? 0 : 1;
}
