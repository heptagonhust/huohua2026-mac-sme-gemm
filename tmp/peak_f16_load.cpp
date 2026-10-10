// Temporary probe: realistic K-loop ceiling on this M4 -- the same 4x ld1h +
// 4x widening fmopa body as the real kernel, streaming over an L2-resident
// region (no DRAM), so we can see whether adding loads (vs the pure-ALU peak)
// is what limits the GEMM kernel.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const std::size_t bytes = 128u << 10;                 // 4 MiB per panel (L2)
    __fp16* A = static_cast<__fp16*>(std::aligned_alloc(64, bytes));
    __fp16* B = static_cast<__fp16*>(std::aligned_alloc(64, bytes));
    std::memset(A, 0, bytes);
    std::memset(B, 0, bytes);

    const long long groups_per_pass = static_cast<long long>(bytes / 128);  // 128 B/group
    const long long passes = 4000;
    const long long iters = groups_per_pass * passes;

    const auto t0 = std::chrono::steady_clock::now();
    asm volatile(
        "smstart\n\tzero {za}\n\tptrue p1.h\n\tptrue p2.h\n\t"
        "mov x9,  %0\n\t"                       // A base
        "mov x10, %1\n\t"                       // B base
        "mov x11, %2\n\t"                       // groups per pass
        "mov x12, %3\n\t"                       // passes
        "2:\n\t"
        "mov x13, x9\n\t"
        "mov x14, x10\n\t"
        "mov x15, x11\n\t"
        "1:\n\t"
        "ld1h {z0.h}, p1/z, [x13]\n\t"
        "ld1h {z1.h}, p1/z, [x13, #1, MUL VL]\n\t"
        "ld1h {z2.h}, p1/z, [x14]\n\t"
        "ld1h {z3.h}, p1/z, [x14, #1, MUL VL]\n\t"
        "fmopa za0.s, p1/m, p1/m, z0.h, z2.h\n\t"
        "fmopa za1.s, p1/m, p1/m, z0.h, z3.h\n\t"
        "fmopa za2.s, p1/m, p1/m, z1.h, z2.h\n\t"
        "fmopa za3.s, p1/m, p1/m, z1.h, z3.h\n\t"
        "add x13, x13, #128\n\t"
        "add x14, x14, #128\n\t"
        "subs x15, x15, #1\n\t"
        "bne 1b\n\t"
        "subs x12, x12, #1\n\t"
        "bne 2b\n\t"
        "smstop\n\t"
        :
        : "r"(A), "r"(B), "r"(groups_per_pass), "r"(passes)
        : "p1", "p2", "z0", "z1", "z2", "z3",
          "x9", "x10", "x11", "x12", "x13", "x14", "x15", "za", "memory");
    const auto t1 = std::chrono::steady_clock::now();

    const double s = std::chrono::duration<double>(t1 - t0).count();
    const double flops = 4096.0 * static_cast<double>(iters);
    std::printf("K-loop with loads: %.1f GFLOP/s  (%.3f s)\n",
                flops / s / 1.0e9, s);
    std::printf("  => %.2f cycles per group @3.92GHz  (pure-ALU peak was 8.78)\n",
                3.92e9 * s / static_cast<double>(iters));
    std::free(A);
    std::free(B);
    return 0;
}
