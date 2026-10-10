// Low-noise timing harness for the FP16 GEMM path (paper time_best protocol).
// Links against the already-built precision objects; no naive baseline, so it
// iterates in seconds instead of minutes.
#include "gemm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

static double time_best(std::size_t N, std::size_t M, std::size_t K,
                        int reps, double warm_ms) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<A_TYPE> A(N * M), B(M * K);
    std::vector<C_TYPE> C(N * K, 0.0f);
    for (A_TYPE& v : A) v = static_cast<A_TYPE>(dist(gen));
    for (B_TYPE& v : B) v = static_cast<B_TYPE>(dist(gen));

    auto run = [&] {
        const auto t0 = Clock::now();
        gemm(A.data(), B.data(), C.data(), N, M, K);
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };

    const auto w0 = Clock::now();
    do { run(); } while (std::chrono::duration<double, std::milli>(Clock::now() - w0).count() < warm_ms);

    double best = 1e30;
    for (int i = 0; i < reps; ++i) best = std::min(best, run());

    const double flops = 2.0 * N * M * K;
    return flops / (best * 1.0e6);
}

int main(int argc, char** argv) {
    const std::size_t dims[][3] = {
        {1024, 1024, 1024}, {1024, 4096, 1024}, {4096, 4096, 4096},
        {512, 4096, 4096},  {512, 4096, 11008}, {2048, 2048, 64},
        {256, 256, 256},    {128, 128, 128},    {100, 100, 100},
        {232, 505, 129},    {4096, 256, 64},
    };
    const int reps = argc > 1 ? std::atoi(argv[1]) : 15;
    std::printf("reps=%d (time_best per shape)\n", reps);
    std::printf("%7s %7s %7s %12s\n", "N", "M", "K", "GFLOP/s");
    for (const auto& d : dims) {
        const double g = time_best(d[0], d[1], d[2], reps, 300.0);
        std::printf("%7zu %7zu %7zu %12.1f\n", d[0], d[1], d[2], g);
    }
    return 0;
}
