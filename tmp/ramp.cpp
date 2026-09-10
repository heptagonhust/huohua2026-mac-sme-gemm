// Temporary probe: per-iteration time of repeated 256x256x256 gemm_fp16 calls,
// used to see whether the first calls in a process are intrinsically slower
// (clock ramp / core placement / lazy SME state) or whether the kernel itself
// is already warm.
#include "gemm.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

int main() {
    const std::size_t N = 256, M = 256, K = 256;
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<__fp16> A(N * M), B(M * K);
    std::vector<float> C(N * K, 0.0f);
    for (__fp16& v : A) {
        v = static_cast<__fp16>(distribution(generator));
    }
    for (__fp16& v : B) {
        v = static_cast<__fp16>(distribution(generator));
    }

    const int iterations = 400;
    std::vector<double> us;
    us.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        gemm_fp16(A.data(), B.data(), C.data(), N, M, K);
        const auto end = Clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::printf("per-call time of 256x256x256 gemm_fp16 (us):\n");
    for (int i = 0; i < 20; ++i) {
        std::printf("  call %3d: %8.1f\n", i, us[i]);
    }
    std::vector<double> tail(us.begin() + 100, us.end());
    std::sort(tail.begin(), tail.end());
    std::printf("  ...\n  median(calls 100..%d): %.1f us   min: %.1f   max: %.1f\n",
                iterations - 1, tail[tail.size() / 2], tail.front(), tail.back());
    const double flops = 2.0 * N * M * K;
    std::printf("  -> %.1f GFLOP/s at the median\n",
                flops / (tail[tail.size() / 2] * 1.0e3));
    return 0;
}
