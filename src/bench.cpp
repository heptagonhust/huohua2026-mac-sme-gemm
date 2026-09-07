#include "gemm.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

bool read_shapes(const std::string& path,
                 std::vector<std::size_t>* shapes) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::size_t N;
    std::size_t M;
    std::size_t K;
    while (input >> N >> M >> K) {
        if (N == 0 || M == 0 || K == 0) {
            std::cerr << "invalid shape in " << path
                      << ": dimensions must be positive\n";
            return false;
        }
        shapes->push_back(N);
        shapes->push_back(M);
        shapes->push_back(K);
    }
    return input.eof() && !shapes->empty();
}

bool allocate_size(std::size_t rows, std::size_t columns,
                   std::size_t* result) {
    if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    *result = rows * columns;
    return true;
}

template <typename Function>
double measure_ms(Function function) {
    const auto start = Clock::now();
    function();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool matches(const std::vector<float>& actual,
             const std::vector<float>& expected,
             float absolute_tolerance,
             float relative_tolerance) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float scale = std::max(1.0f, std::fabs(expected[index]));
        if (difference > absolute_tolerance + relative_tolerance * scale) {
            return false;
        }
    }
    return true;
}

}  // namespace

void baseline_gemm(const __fp16* A, const __fp16* B, float* C,
                   std::size_t N, std::size_t M, std::size_t K) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < M; ++k) {
                const float a = static_cast<float>(A[i * M + k]);
                const float b = static_cast<float>(B[k * K + j]);
                sum += a * b;
            }
            C[i * K + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const std::string input_path = argc > 1 ? argv[1] : "data/test.in";
    std::vector<std::size_t> shapes;
    if (!read_shapes(input_path, &shapes)) {
        std::cerr << "failed to read shapes from " << input_path
                  << " (expected one N M K triplet per line)\n";
        return 1;
    }

    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::cout << std::fixed << std::setprecision(3)
              << std::setw(7) << "N"
              << std::setw(7) << "M"
              << std::setw(7) << "K"
              << std::setw(12) << "correct"
              << std::setw(16) << "baseline"
              << std::setw(16) << "optimized"
              << std::setw(10) << "speedup"
              << '\n'
              << std::setw(7) << ""
              << std::setw(7) << ""
              << std::setw(7) << ""
              << std::setw(12) << ""
              << std::setw(16) << "(GFLOP/s)"
              << std::setw(16) << "(GFLOP/s)"
              << std::setw(10) << ""
              << '\n';

    for (std::size_t shape = 0; shape < shapes.size(); shape += 3) {
        const std::size_t N = shapes[shape];
        const std::size_t M = shapes[shape + 1];
        const std::size_t K = shapes[shape + 2];
        std::size_t A_size;
        std::size_t B_size;
        std::size_t C_size;
        if (!allocate_size(N, M, &A_size) ||
            !allocate_size(M, K, &B_size) ||
            !allocate_size(N, K, &C_size)) {
            std::cerr << "shape is too large: " << N << ' ' << M << ' ' << K
                      << '\n';
            return 1;
        }

        std::vector<__fp16> A(A_size);
        std::vector<__fp16> B(B_size);
        std::vector<float> reference(C_size);
        std::vector<float> result(C_size, std::numeric_limits<float>::quiet_NaN());
        // Paper inputs are FP16 row-major: generate FP32 in [-1,1], store FP16.
        for (__fp16& value : A) {
            value = static_cast<__fp16>(distribution(generator));
        }
        for (__fp16& value : B) {
            value = static_cast<__fp16>(distribution(generator));
        }

        const double baseline_ms = measure_ms([&] {
            baseline_gemm(A.data(), B.data(), reference.data(), N, M, K);
        });
        const double optimized_ms = measure_ms([&] {
            gemm_fp16(A.data(), B.data(), result.data(), N, M, K);
        });
        const bool correct = matches(result, reference, 1.0e-4f, 1.0e-4f);

        // FP32 GEMM: 2*N*M*K floating-point operations.
        const double flops = 2.0 * static_cast<double>(N) *
                             static_cast<double>(M) *
                             static_cast<double>(K);
        const double baseline_gflops =
            baseline_ms > 0.0 ? flops / (baseline_ms * 1.0e6) : 0.0;
        const double optimized_gflops =
            optimized_ms > 0.0 ? flops / (optimized_ms * 1.0e6) : 0.0;
        const double speedup =
            optimized_ms > 0.0 ? baseline_ms / optimized_ms
                               : std::numeric_limits<double>::infinity();

        std::cout << std::setw(7) << N
                  << std::setw(7) << M
                  << std::setw(7) << K
                  << std::setw(12) << (correct ? "yes" : "no")
                  << std::setw(16) << baseline_gflops
                  << std::setw(16) << optimized_gflops
                  << std::setw(10) << speedup
                  << '\n';
    }
    return 0;
}
