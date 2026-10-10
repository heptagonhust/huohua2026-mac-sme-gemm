// Temporary probe: does the SVE2p1/SME2 multi-vector load
//   ld1h {z0.h-z1.h}, pn8/z, [x]
// execute on this Apple M4 inside streaming mode, and does it load two
// consecutive 64-byte blocks into z0 and z1?
#include <csignal>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <cstdint>

static sigjmp_buf jb;
static void on_sig(int) { siglongjmp(jb, 1); }

static __fp16 src[64];
static __fp16 out0[32], out1[32];

int main() {
    for (int i = 0; i < 64; ++i) src[i] = (__fp16)(i + 1);
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, nullptr);

    if (sigsetjmp(jb, 1) == 0) {
        asm volatile(
            "smstart\n\t"
            "ptrue pn8.h\n\t"
            "ld1h {z0.h-z1.h}, pn8/z, [%0]\n\t"
            "ptrue p0.h\n\t"
            "st1h {z0.h}, p0, [%1]\n\t"
            "st1h {z1.h}, p0, [%2]\n\t"
            "smstop\n\t"
            :: "r"(src), "r"(out0), "r"(out1)
            : "p0", "p8", "z0", "z1", "memory");
        std::printf("multivec OK\n");
        std::printf("  out0[0..3] = %d %d %d %d  (expect 1 2 3 4)\n",
                    (int)out0[0], (int)out0[1], (int)out0[2], (int)out0[3]);
        std::printf("  out1[0..3] = %d %d %d %d  (expect 33 34 35 36)\n",
                    (int)out1[0], (int)out1[1], (int)out1[2], (int)out1[3]);
    } else {
        std::printf("multivec SIGILL (instruction not implemented)\n");
    }
    return 0;
}
