#include <cstdio>
#include <cstring>
#include <cstdint>

int main() {
    __fp16 A[32], B[32];
    for (int i = 0; i < 32; i++) { A[i] = (__fp16)(i + 1);      B[i] = (__fp16)(i + 101); }
    float row[16][16];
    memset(row, 0, sizeof row);
    asm volatile(
        "smstart\n\t"
        "zero {za}\n\t"
        "ptrue p0.h\n\t"
        "ld1h {z0.h}, p0/z, [%1]\n\t"     // A: 32 half
        "ld1h {z1.h}, p0/z, [%2]\n\t"     // B: 32 half
        "fmopa za0.s, p0/m, p0/m, z0.h, z1.h\n\t"
        "mov w12, #0\n\t"
        "mov x16, %0\n\t"
        "1:\n\t"
        "mov z2.s, p0/m, za0h.s[w12, 0]\n\t"
        "st1w {z2.s}, p0, [x16]\n\t"
        "add x16, x16, #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "blt 1b\n\t"
        "smstop\n\t"
        :: "r"(row), "r"(A), "r"(B)
        : "p0","z0","z1","z2","w12","x16","za","memory");
    printf("fp16 widening FMOPA -> 16x16 fp32 tile\n");
    for (int r = 0; r < 16; r++) { for (int c = 0; c < 16; c++) printf("%6.0f", row[r][c]); printf("\n"); }
    return 0;
}
