// Temporary probe: decisive test for the 2-element-dot vs even-lane semantics.
#include <cstdio>
#include <cstring>
#include <cstdint>

static void run(const __fp16* A, const __fp16* B, float* row, const char* pgmode) {
    std::memset(row, 0, 16 * 16 * sizeof(float));
    if (pgmode[0] == 'h') {
        asm volatile(
            "smstart\n\tzero {za}\n\tptrue p1.h\n\tptrue p2.h\n\t"
            "ld1h {z0.h}, p2/z, [%1]\n\tld1h {z1.h}, p2/z, [%2]\n\t"
            "fmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
            "mov w12, #0\n\tmov x16, %0\n\t"
            "1:\n\tmov z2.s, p1/m, za0h.s[w12, 0]\n\tst1w {z2.s}, p1, [x16]\n\t"
            "add x16, x16, #64\n\tadd w12, w12, #1\n\tcmp w12, #16\n\tblt 1b\n\tsmstop\n\t"
            :: "r"(row), "r"(A), "r"(B)
            : "p1", "p2", "z0", "z1", "z2", "w12", "x16", "za", "memory");
    } else {
        asm volatile(
            "smstart\n\tzero {za}\n\tptrue p1.s\n\tptrue p2.h\n\t"
            "ld1h {z0.h}, p2/z, [%1]\n\tld1h {z1.h}, p2/z, [%2]\n\t"
            "fmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
            "mov w12, #0\n\tmov x16, %0\n\t"
            "1:\n\tmov z2.s, p1/m, za0h.s[w12, 0]\n\tst1w {z2.s}, p1, [x16]\n\t"
            "add x16, x16, #64\n\tadd w12, w12, #1\n\tcmp w12, #16\n\tblt 1b\n\tsmstop\n\t"
            :: "r"(row), "r"(A), "r"(B)
            : "p1", "p2", "z0", "z1", "z2", "w12", "x16", "za", "memory");
    }
}

int main() {
    __fp16 A[32], B[32];
    std::memset(A, 0, sizeof A);
    std::memset(B, 0, sizeof B);
    A[0] = 1; A[1] = 1; B[0] = 1; B[1] = 1;
    {
        float row[16][16];
        run(A, B, &row[0][0], "s");
        std::printf("pred=s A=[1,1,0..] B=[1,1,0..] -> ZA[0][0]=%g  ZA[0][1]=%g  ZA[1][0]=%g\n",
                    row[0][0], row[0][1], row[1][0]);
        run(A, B, &row[0][0], "h");
        std::printf("pred=h A=[1,1,0..] B=[1,1,0..] -> ZA[0][0]=%g  ZA[0][1]=%g  ZA[1][0]=%g\n",
                    row[0][0], row[0][1], row[1][0]);
    }
    // ramp with predicate .h
    for (int i = 0; i < 32; ++i) { A[i] = (__fp16)(i + 1); B[i] = (__fp16)(i + 101); }
    float row[16][16];
    run(A, B, &row[0][0], "h");
    std::printf("pred=h ramp -> row0[0..3] = %g %g %g %g\n",
                row[0][0], row[0][1], row[0][2], row[0][3]);
    return 0;
}
