// Temporary diagnostic: dyld interposer that logs every
// std::chrono::steady_clock::now() call and every gemm_fp16() call.
// It does NOT require recompiling the target, so the target's codegen is
// untouched (the bug under investigation is codegen-sensitive).
//
// Build: clang -dynamiclib -O1 -o tmp/interpose.dylib tmp/interpose.c
// Run:   DYLD_INSERT_LIBRARIES=$PWD/tmp/interpose.dylib ./tmp/dbg_bench tmp/data.in
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Original symbols (mangled C++ names).
extern long long real_now(void)
    __asm__("__ZNSt3__16chrono12steady_clock3nowEv");
extern void real_gemm(const void*, const void*, void*, unsigned long,
                      unsigned long, unsigned long)
    __asm__("__Z9gemm_fp16PKDhS0_Pfmmm");

static int now_calls = 0;
static int gemm_calls = 0;
static long long prev_now = 0;

long long interposed_now(void) {
    const long long v = real_now();
    ++now_calls;
    fprintf(stderr, "[now ] #%02d t=%lld  delta_from_prev=%lld\n", now_calls,
            v, prev_now == 0 ? 0 : v - prev_now);
    prev_now = v;
    return v;
}

void interposed_gemm(const void* A, const void* B, void* C, unsigned long N,
                     unsigned long M, unsigned long K) {
    ++gemm_calls;
    fprintf(stderr, "[gemm] #%d begin N=%lu M=%lu K=%lu\n", gemm_calls, N, M, K);
    real_gemm(A, B, C, N, M, K);
    fprintf(stderr, "[gemm] #%d end\n", gemm_calls);
}

#define DYLD_INTERPOSE(replacement, replacee)                                \
    __attribute__((used)) static struct {                                    \
        const void* replacement;                                             \
        const void* replacee;                                                \
    } _interpose_##replacee __attribute__((section("__DATA,__interpose"))) = {\
        (const void*)(unsigned long)&replacement,                            \
        (const void*)(unsigned long)&replacee}

DYLD_INTERPOSE(interposed_now, real_now);
DYLD_INTERPOSE(interposed_gemm, real_gemm);
