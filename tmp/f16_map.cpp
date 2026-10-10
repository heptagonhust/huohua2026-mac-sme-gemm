// Temporary probe: characterise the exact lane mapping of the FP16 widening
// FMOPA (za0.s, p1/m, p1/m, z0.h, z1.h) on Apple M4 with a 32-half A and B
// vector.  Basis-vector inputs reveal which A lane multiplies which B lane.
#include <cstdio>
#include <cstring>
#include <cstdint>

static void run(const __fp16* A, const __fp16* B, float* row) {
    std::memset(row, 0, 16 * 16 * sizeof(float));
    asm volatile(
        "smstart\n\t"
        "zero {za}\n\t"
        "ptrue p1.s\n\t"
        "ptrue p2.h\n\t"
        "ld1h {z0.h}, p2/z, [%1]\n\t"
        "ld1h {z1.h}, p2/z, [%2]\n\t"
        "fmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
        "mov w12, #0\n\t"
        "mov x16, %0\n\t"
        "1:\n\t"
        "mov z2.s, p1/m, za0h.s[w12, 0]\n\t"
        "st1w {z2.s}, p1, [x16]\n\t"
        "add x16, x16, #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "blt 1b\n\t"
        "smstop\n\t"
        :: "r"(row), "r"(A), "r"(B)
        : "p1", "p2", "z0", "z1", "z2", "w12", "x16", "za", "memory");
}

static void show(const char* name, const __fp16* A, const __fp16* B) {
    float row[16][16];
    run(A, B, &row[0][0]);
    std::printf("== %s ==\n", name);
    for (int r = 0; r < 4; ++r) {
        std::printf("  row%d:", r);
        for (int c = 0; c < 6; ++c) std::printf("%6.0f", row[r][c]);
        std::printf(" ...\n");
    }
}

int main() {
    __fp16 rampA[32], rampB[32];
    for (int i = 0; i < 32; ++i) { rampA[i] = (__fp16)(i + 1); rampB[i] = (__fp16)(i + 101); }
    static __fp16 e0[32], e1[32], e16[32], e17[32];
    std::memset(e0, 0, sizeof e0); std::memset(e1, 0, sizeof e1);
    std::memset(e16, 0, sizeof e16); std::memset(e17, 0, sizeof e17);
    e0[0] = 1; e1[1] = 1; e16[16] = 1; e17[17] = 1;
    show("A=e0,  B=ramp", e0, rampB);
    show("A=e1,  B=ramp", e1, rampB);
    show("A=e16, B=ramp", e16, rampB);
    show("A=e17, B=ramp", e17, rampB);
    show("A=ramp, B=e0", rampA, e0);
    show("A=ramp, B=e1", rampA, e1);
    show("A=ramp, B=e16", rampA, e16);
    return 0;
}
