// Temporary diagnostic #2: interpose gemm_fp16() and dump the callee-saved
// registers that the caller (bench.cpp -O3) keeps live across the call.
//
// Motivation: in bench.cpp -O3 the accumulated optimized_ms_sum and the
// measurement timestamps live in x20/x27 and in the low 64 bits of v8-v15
// (d12 holds the constant 3.0 divisor).  The AAPCS64 requires the callee to
// preserve those, so if the SME micro-kernel (reached through gemm_fp16)
// clobbers any of them we can see it here.
//
// Build: clang -dynamiclib -O1 -Wl,-undefined,dynamic_lookup -o tmp/interpose2.dylib tmp/interpose2.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void real_gemm(const void*, const void*, void*, unsigned long,
                      unsigned long, unsigned long)
    __asm__("__Z9gemm_fp16PKDhS0_Pfmmm");

static inline unsigned long long rd_x19(void) {
    unsigned long long v; __asm__ volatile("mov %0, x19" : "=r"(v)); return v;
}
static inline unsigned long long rd_x20(void) {
    unsigned long long v; __asm__ volatile("mov %0, x20" : "=r"(v)); return v;
}
static inline unsigned long long rd_x21(void) {
    unsigned long long v; __asm__ volatile("mov %0, x21" : "=r"(v)); return v;
}
static inline unsigned long long rd_x22(void) {
    unsigned long long v; __asm__ volatile("mov %0, x22" : "=r"(v)); return v;
}
static inline unsigned long long rd_x23(void) {
    unsigned long long v; __asm__ volatile("mov %0, x23" : "=r"(v)); return v;
}
static inline unsigned long long rd_x24(void) {
    unsigned long long v; __asm__ volatile("mov %0, x24" : "=r"(v)); return v;
}
static inline unsigned long long rd_x25(void) {
    unsigned long long v; __asm__ volatile("mov %0, x25" : "=r"(v)); return v;
}
static inline unsigned long long rd_x26(void) {
    unsigned long long v; __asm__ volatile("mov %0, x26" : "=r"(v)); return v;
}
static inline unsigned long long rd_x27(void) {
    unsigned long long v; __asm__ volatile("mov %0, x27" : "=r"(v)); return v;
}
static inline unsigned long long rd_x28(void) {
    unsigned long long v; __asm__ volatile("mov %0, x28" : "=r"(v)); return v;
}
static inline unsigned long long rd_d8(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d8" : "=r"(v)); return v;
}
static inline unsigned long long rd_d9(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d9" : "=r"(v)); return v;
}
static inline unsigned long long rd_d10(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d10" : "=r"(v)); return v;
}
static inline unsigned long long rd_d11(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d11" : "=r"(v)); return v;
}
static inline unsigned long long rd_d12(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d12" : "=r"(v)); return v;
}
static inline unsigned long long rd_d13(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d13" : "=r"(v)); return v;
}
static inline unsigned long long rd_d14(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d14" : "=r"(v)); return v;
}
static inline unsigned long long rd_d15(void) {
    unsigned long long v; __asm__ volatile("fmov %0, d15" : "=r"(v)); return v;
}

#define TAKE(name)                     \
    do {                               \
        name[0] = rd_x19();            \
        name[1] = rd_x20();            \
        name[2] = rd_x21();            \
        name[3] = rd_x22();            \
        name[4] = rd_x23();            \
        name[5] = rd_x24();            \
        name[6] = rd_x25();            \
        name[7] = rd_x26();            \
        name[8] = rd_x27();            \
        name[9] = rd_x28();            \
        name[10] = rd_d8();            \
        name[11] = rd_d9();            \
        name[12] = rd_d10();           \
        name[13] = rd_d11();           \
        name[14] = rd_d12();           \
        name[15] = rd_d13();           \
    } while (0)

static const char* k_names[16] = {
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26",
    "x27", "x28", "d8 ", "d9 ", "d10", "d11", "d12", "d13"};

static int calls = 0;

void interposed_gemm(const void* A, const void* B, void* C, unsigned long N,
                     unsigned long M, unsigned long K) {
    unsigned long long before[16];
    unsigned long long after[16];
    TAKE(before);
    real_gemm(A, B, C, N, M, K);
    TAKE(after);

    ++calls;
    fprintf(stderr, "[gemm #%d] N=%lu M=%lu K=%lu\n", calls, N, M, K);
    int bad = 0;
    for (int i = 0; i < 16; ++i) {
        if (before[i] != after[i]) {
            ++bad;
            fprintf(stderr, "    CLOBBERED %s: 0x%016llx -> 0x%016llx\n",
                    k_names[i], before[i], after[i]);
        }
    }
    if (!bad) {
        fprintf(stderr, "    all 16 checked registers preserved\n");
    }
}

#define DYLD_INTERPOSE(replacement, replacee)                                 \
    __attribute__((used)) static struct {                                     \
        const void* replacement;                                              \
        const void* replacee;                                                 \
    } _interpose_##replacee __attribute__((section("__DATA,__interpose"))) = { \
        (const void*)(unsigned long)&replacement,                             \
        (const void*)(unsigned long)&replacee}

DYLD_INTERPOSE(interposed_gemm, real_gemm);
