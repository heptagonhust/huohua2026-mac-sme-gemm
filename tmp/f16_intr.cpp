#include <arm_sme.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

__arm_locally_streaming __attribute__((noinline))
void kernel16(__fp16* A, __fp16* B, float* out) {
    svbool_t pgs = svptrue_b32();      // .s predicate for tile rows
    svbool_t pgh = svptrue_b16();      // .h predicate for loads
    svfloat16_t a = svld1_f16(pgh, A);
    svfloat16_t b = svld1_f16(pgh, B);
    svmopa_za32_f16_m(0, pgs, pgs, a, b);
    for (int r = 0; r < 16; r++) {
        svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0.0f), pgs, 0, (uint32_t)r);
        svst1_f32(pgs, out + (size_t)r * 16, row);
    }
}

int main() {
    __fp16 A[32], B[32];
    for (int i = 0; i < 32; i++) { A[i] = (__fp16)(i + 1); B[i] = (__fp16)(i + 101); }
    float out[16][16];
    memset(out, 0, sizeof out);
    kernel16(A, B, &out[0][0]);
    printf("intrinsic fp16 widening -> 16x16\n");
    for (int r = 0; r < 16; r++) { for (int c = 0; c < 16; c++) printf("%6.0f", out[r][c]); printf("\n"); }
    return 0;
}
