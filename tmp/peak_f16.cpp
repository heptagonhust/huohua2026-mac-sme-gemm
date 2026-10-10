// Temporary probe: instruction-level ceiling of the FP16 widening FMOPA on
// this M4, i.e. 4 independent ZA32 chains with no memory traffic in the loop.
#include <chrono>
#include <cstdio>
#include <cstdint>

int main() {
    static __fp16 buf[64];
    for (int i = 0; i < 64; ++i) buf[i] = (__fp16)1.0f;
    const long long iters = 200000000LL;

    const auto t0 = std::chrono::steady_clock::now();
    asm volatile(
        "smstart\n\tzero {za}\n\tptrue p1.h\n\tptrue p2.h\n\t"
        "ld1h {z0.h}, p2/z, [%0]\n\t"
        "ld1h {z1.h}, p2/z, [%0, #1, MUL VL]\n\t"
        "mov x9, %1\n\t"
        "1:\n\t"
        "fmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
        "fmopa za1.s, p1/m, p1/m, z0.h, z1.h\n\t"
        "fmopa za2.s, p1/m, p1/m, z1.h, z0.h\n\t"
        "fmopa za3.s, p1/m, p1/m, z1.h, z0.h\n\t"
        "subs x9, x9, #1\n\t"
        "bne 1b\n\t"
        "smstop\n\t"
        :
        : "r"(buf), "r"(iters)
        : "p1", "p2", "z0", "z1", "x9", "za", "memory");
    const auto t1 = std::chrono::steady_clock::now();

    const double s = std::chrono::duration<double>(t1 - t0).count();
    const double flops = 4.0 * 1024.0 * static_cast<double>(iters);
    std::printf("widening FMOPA peak: %.1f GFLOP/s  (%.3f s for %lld iters)\n",
                flops / s / 1.0e9, s, iters);
    std::printf("  => %.2f cycles per 4-FMOPA iteration @3.92GHz\n",
                3.92e9 * s / static_cast<double>(iters));
    return 0;
}
