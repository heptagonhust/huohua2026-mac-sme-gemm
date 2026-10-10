// Temporary probe: does the BF16 widening outer product (BFMOPA .S <- .H) work
// on this Apple M4, and does it need the same .h predicate granularity as the
// FP16 widening FMOPA?
#include <cstdio>
#include <cstring>
#include <cstdint>

static void run(const __bf16* A, const __bf16* B, float* row, char pgmode) {
    std::memset(row, 0, 16 * 16 * sizeof(float));
    if (pgmode == 'h') {
        asm volatile(
            "smstart\n\tzero {za}\n\tptrue p1.h\n\tptrue p2.h\n\t"
            "ld1h {z0.h}, p2/z, [%1]\n\tld1h {z1.h}, p2/z, [%2]\n\t"
            "bfmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
            "mov w12, #0\n\tmov x16, %0\n\t"
            "1:\n\tmov z2.s, p1/m, za0h.s[w12, 0]\n\tst1w {z2.s}, p1, [x16]\n\t"
            "add x16, x16, #64\n\tadd w12, w12, #1\n\tcmp w12, #16\n\tblt 1b\n\tsmstop\n\t"
            :: "r"(row), "r"(A), "r"(B)
            : "p1", "p2", "z0", "z1", "z2", "w12", "x16", "za", "memory");
    } else {
        asm volatile(
            "smstart\n\tzero {za}\n\tptrue p1.s\n\tptrue p2.h\n\t"
            "ld1h {z0.h}, p2/z, [%1]\n\tld1h {z1.h}, p2/z, [%2]\n\t"
            "bfmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
            "mov w12, #0\n\tmov x16, %0\n\t"
            "1:\n\tmov z2.s, p1/m, za0h.s[w12, 0]\n\tst1w {z2.s}, p1, [x16]\n\t"
            "add x16, x16, #64\n\tadd w12, w12, #1\n\tcmp w12, #16\n\tblt 1b\n\tsmstop\n\t"
            :: "r"(row), "r"(A), "r"(B)
            : "p1", "p2", "z0", "z1", "z2", "w12", "x16", "za", "memory");
    }
}

int main() {
    __bf16 A[32], B[32];
    // Apple clang 17 cannot lower int->__bf16 (backend crash), so go through
    // float, which is the one conversion it does support.
    for (int i = 0; i < 32; ++i) {
        A[i] = (__bf16)static_cast<float>(i + 1);
        B[i] = (__bf16)static_cast<float>(i + 101);
    }
    float row[16][16];
    run(A, B, &row[0][0], 'h');
    std::printf("bfmopa  pred=.h ramp -> row0[0..3] = %g %g %g %g  (expect 305 311 317 323)\n",
                row[0][0], row[0][1], row[0][2], row[0][3]);
    run(A, B, &row[0][0], 's');
    std::printf("bfmopa  pred=.s ramp -> row0[0..3] = %g %g %g %g\n",
                row[0][0], row[0][1], row[0][2], row[0][3]);
    return 0;
}
