// Temporary probe: distinguish FMOPA widening semantics.
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

int main() {
    __fp16 Avec[32], Bvec[32];
    // Cases designed to isolate the two-product structure.
    int cases[][2] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}, {2, 2},
        {1, 3}, {3, 3}, {2, 3}, {0, 16}, {16, 0}, {16, 16}, {17, 17},
    };
    for (auto& c : cases) {
        std::memset(Avec, 0, sizeof Avec);
        std::memset(Bvec, 0, sizeof Bvec);
        Avec[c[0]] = 1;
        Bvec[c[1]] = 1;
        float row[16][16];
        run(Avec, Bvec, &row[0][0]);
        std::printf("A=e%-2d B=e%-2d ->", c[0], c[1]);
        bool any = false;
        for (int r = 0; r < 16; ++r)
            for (int col = 0; col < 16; ++col)
                if (row[r][col] != 0.0f) {
                    std::printf(" ZA[%d][%d]=%g", r, col, row[r][col]);
                    any = true;
                }
        if (!any) std::printf(" (all zero)");
        std::printf("\n");
    }
    return 0;
}
