// ---------------------------------------------------------------------------
// assemble.s -- SME FP32 32x32 micro-kernel (single SME worker / P-cluster).
//
// Computes one full 32x32 output tile that was already packed k-major:
//   C(32x32 tile, row-major, leading dim ldc) = sum_k A_panel(k,:) (x) B_panel(k,:)
// NOTE: the tile is OVERWRITTEN, never accumulated into.  In particular
// kc == 0 stores zeros (it does not leave the previous C contents alone).
//
// Layout assumptions (matches gemm.cpp pack_a_panel / pack_b_band):
//   A_panel: FP32, k-major.  Row of inner index k holds 32 A-lanes
//            contiguous at A_panel[k*lda + i], i in [0,32).
//   B_panel: FP32, k-major.  Row of inner index k holds 32 B-lanes
//            contiguous at B_panel[k*ldb + j], j in [0,32).
//   C:       row-major, leading dimension ldc.
//
// One ZA32 tile is 16x16; four tiles in a 2x2 arrangement cover the 32x32
// output (paper Sec. 2.3 / 4.2):
//   ZA0 -> C[ 0..15][ 0..15]   ZA1 -> C[ 0..15][16..31]
//   ZA2 -> C[16..31][ 0..15]   ZA3 -> C[16..31][16..31]
// Every FMOPA step loads a0,a1 (A rows 0-15 / 16-31) and b0,b1 (B cols
// 0-15 / 16-31), then issues the four widening-ish (FP32) outer products.
//
// C ABI entry (aarch64): x0..x6 carry the 7 scalar arguments, so this asm
// needs no stack frame for arguments.  Streaming mode (SMSTART/SMSTOP) is
// entered/exited once around the whole tile so the toggle cost is amortized.
//
// ---------------------------------------------------------------------------
// ABI / encoding notes -- read before editing this file
// ---------------------------------------------------------------------------
// * Only caller-saved state is used: x0-x17, z0-z5 (== v0-v5) and p0.
// * v8-v15 (low 64 bits) are CALLEE-SAVED per AAPCS64, but on Apple M4 the
//   SMSTART/SMSTOP streaming-mode toggle destroys ALL of d8-d15: measured,
//   they read back as 0 and SMSTOP does not restore them.  The prologue and
//   epilogue below therefore spill d8-d15 itself.  That spill MUST happen
//   outside streaming mode: save before SMSTART, restore after SMSTOP.
//   (Without it, a caller compiled at -O3 keeps constants in d8-d15 and dies
//   with wrong results, e.g. a divisor of 3.0 turning into 0.0 -> x/0 = inf.)
// * ZA is enabled, zeroed, accumulated into (ZA0-ZA3) and left zeroed: the
//   caller's ZA contents are NOT preserved.  This is legal -- in the SME ABI
//   ZA is caller-saved unless the callee declares __arm_preserves("za") --
//   but callers holding live data in ZA must save it themselves.
// * PSTATE.SM and PSTATE.ZA are restored to 0 on return by the bare `smstop`
//   (= SMSTOP SM,ZA, encoding 0xD503467F).  So this kernel must only be
//   entered from a non-streaming context, and it must not be called from a
//   function that is itself executing in streaming mode.
// * The bare `smstart` (= SMSTART SM,ZA, encoding 0xD503477F) sets BOTH
//   PSTATE.SM and PSTATE.ZA, which is required for ZERO/FMOPA to be legal.
//   Do not "simplify" it to `smstart sm` (0xD503437F) -- that leaves ZA
//   disabled; `smstart za` is 0xD503457F, and this assembler rejects the
//   spelled-out `smstart sm, za` form entirely.
// * Vector length: the code is hard-wired to a 512-bit streaming vector
//   length (64 bytes == 16 FP32 lanes per Z register; measured with RDSVL on
//   Apple M4).  On any other SVL the entry point bails out without touching
//   C, so callers must use huohua_sme_svl_bytes() to select a fallback.
// ---------------------------------------------------------------------------

    .text
    .p2align 2

// extern "C" void huohua_sme_microkernel_32x32(
//     const float* A_panel,   // x0
//     int lda,                // w1  (A panel row stride, in FP32)
//     const float* B_panel,   // x2
//     int ldb,                // w3  (B panel row stride, in FP32)
//     float* C,               // x4
//     int ldc,                // w5  (C leading dim, in FP32)
//     int kc);                // w6  (inner dimension count)
    .globl _huohua_sme_microkernel_32x32
_huohua_sme_microkernel_32x32:

    // This kernel only implements the 512-bit/32-lane layout.  If the machine
    // has a different streaming vector length, bail out and leave C untouched
    // instead of storing a wrong number of bytes (callers should consult
    // huohua_sme_svl_bytes() and take their scalar fallback path).
    rdsvl x15, #1                   // streaming vector length, in bytes
    cmp  x15, #64
    b.ne 8f                         // unsupported SVL -> no-op

    // AAPCS64 / SVE-SME PCS: the low 64 bits of v8-v15 are callee-saved.
    // On Apple M4 the SMSTART/SMSTOP streaming-mode toggle destroys them
    // (measured: every one of d8-d15 reads back as zero afterwards), so the
    // kernel must spill them itself around the whole streaming section.
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // Byte strides.  lda/ldb/ldc are C `int`, i.e. only w1/w3/w5 are defined
    // by the caller -- sign-extend them before scaling (the upper halves of
    // x1/x3/x5 are unspecified per AAPCS64).
    sxtw x9,  w1
    lsl  x9,  x9, #2                // a_stride = lda * 4
    sxtw x10, w3
    lsl  x10, x10, #2               // b_stride = ldb * 4
    sxtw x11, w5
    lsl  x11, x11, #2               // c_stride = ldc * 4

    smstart                         // enter streaming mode once

    mov  x13, x0                    // a row pointer (k = 0)
    mov  x14, x2                    // b row pointer (k = 0)

    ptrue p0.s
    zero {za}

    cbz  w6, 2f                     // kc == 0 -> straight to epilogue

    // ---------------- K (inner) accumulation loop ----------------
1:  ld1w {z0.s}, p0/z, [x13]            // a0 = A rows 0-15 @ k
    ld1w {z1.s}, p0/z, [x13, #1, MUL VL] // a1 = A rows 16-31 @ k
    ld1w {z2.s}, p0/z, [x14]            // b0 = B cols 0-15 @ k
    ld1w {z3.s}, p0/z, [x14, #1, MUL VL] // b1 = B cols 16-31 @ k

    fmopa za0.s, p0/m, p0/m, z0.s, z2.s // C[0:16][0:16]  += a0*b0
    fmopa za1.s, p0/m, p0/m, z0.s, z3.s // C[0:16][16:32] += a0*b1
    fmopa za2.s, p0/m, p0/m, z1.s, z2.s // C[16:32][0:16] += a1*b0
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s // C[16:32][16:32]+= a1*b1

    add  x13, x13, x9                  // advance a row by lda
    add  x14, x14, x10                 // advance b row by ldb
    subs w6, w6, #1
    bne  1b
2:
    // ---------------- Epilogue: read ZA back into C ----------------
    // Upper 16 rows of the tile come from ZA0 (cols 0-15) and ZA1 (cols
    // 16-31).  Row index is held in w12 (SME2 mova register form).
    mov  w12, #0
    mov  x16, x4                       // cptr = C
3:  mov z4.s, p0/m, za0h.s[w12, 0]       // ZA0 row w12 -> 16 lanes
    mov z5.s, p0/m, za1h.s[w12, 0]       // ZA1 row w12 -> 16 lanes
    st1w {z4.s}, p0, [x16]             // C[r][0:16]
    st1w {z5.s}, p0, [x16, #1, MUL VL] // C[r][16:32]
    add  x16, x16, x11                 // next C row (ldc FP32s)
    add  w12, w12, #1
    cmp  w12, #16
    blt  3b

    // Lower 16 rows come from ZA2 (cols 0-15) and ZA3 (cols 16-31).
    mov  w12, #0
    lsl  x15, x11, #4                  // 16 * ldc (FP32) in bytes
    add  x16, x4, x15                  // cptr = C + 16*ldc
4:  mov z4.s, p0/m, za2h.s[w12, 0]       // ZA2 row w12
    mov z5.s, p0/m, za3h.s[w12, 0]       // ZA3 row w12
    st1w {z4.s}, p0, [x16]             // C[16+r][0:16]
    st1w {z5.s}, p0, [x16, #1, MUL VL] // C[16+r][16:32]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #16
    blt  4b

    smstop                          // leave streaming mode

    // Restore the callee-saved SIMD registers saved in the prologue.
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret

8:  ret                             // unsupported SVL: leave C untouched

// ---------------------------------------------------------------------------
// unsigned huohua_sme_svl_bytes(void)
//   Returns the streaming SVE vector length in bytes (RDSVL).  Readable
//   outside streaming mode; requires FEAT_SME.
// ---------------------------------------------------------------------------
    .globl _huohua_sme_svl_bytes
_huohua_sme_svl_bytes:
    rdsvl x0, #1
    ret
