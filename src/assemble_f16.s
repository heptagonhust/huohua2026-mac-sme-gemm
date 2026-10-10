// ---------------------------------------------------------------------------
// assemble_f16.s -- SME FP16 widening 32x32 micro-kernel (single SME worker).
//
// Computes one full 32x32 output tile from k-major FP16 panels using the
// *widening* FP16->FP32 outer product, kr = 2 (paper Sec. 2.1 / 4.2):
//
//   C(32x32 tile, row-major, leading dim ldc) =
//       sum_g A_panel(g,:) (x) B_panel(g,:)
//
// where each group g covers two adjacent k values (k = 2g, 2g+1) and every
// widening FMOPA contributes a 2-element dot per output element:
//
//   ZA.S[i][j] += Zn.H[2i] * Zm.H[2j] + Zn.H[2i+1] * Zm.H[2j+1]
//
// NOTE: the tile is OVERWRITTEN, never accumulated into.  kc == 0 stores zeros.
//
// ---------------------------------------------------------------------------
// CRITICAL: predicate granularity (measured on Apple M4)
// ---------------------------------------------------------------------------
// The widening FMOPA evaluates its predicates at 16-bit element granularity,
// i.e. it needs a `.h` predicate with ALL 32 half-lanes active.  If a `.s`
// predicate (16 bits) is used instead, only the *even* half-lanes are active
// and the instruction degenerates to a rank-1 update on A[2i]*B[2j] (512 FLOP)
// with the odd lanes silently masked off -- a very confusing silent-wrong
// result.  So the K-loop uses p1.h for both the ld1h loads and the FMOPA,
// while the epilogue uses p2.s for the 32-bit ZA reads / st1w stores.
//
// ---------------------------------------------------------------------------
// Layout assumptions (matches gemm.cpp pack_a_panel / pack_b_band, kr = 2)
// ---------------------------------------------------------------------------
//   A_panel: FP16, one 32-row panel at a time, k-pair interleaved per row:
//            lane 2i   = A[row0+i][2g]
//            lane 2i+1 = A[row0+i][2g+1]
//            address  = A_panel[g*lda + i*2 + lane], lda == 64 halves.
//   B_panel: same layout along the 32 columns of the band:
//            address  = B_panel[g*ldb + j*2 + lane], ldb == band_cols*2.
//   C:       row-major, leading dimension ldc (FP32).
//
// One ZA32 tile is 16x16; four tiles in a 2x2 arrangement cover the 32x32
// output:
//   ZA0 -> C[ 0..15][ 0..15]   ZA1 -> C[ 0..15][16..31]
//   ZA2 -> C[16..31][ 0..15]   ZA3 -> C[16..31][16..31]
//
// ---------------------------------------------------------------------------
// ABI / encoding notes (same constraints as assemble_f32.s)
// ---------------------------------------------------------------------------
// * Only caller-saved state is used: x0-x17, z0-z5 (== v0-v5), p1, p2.
// * v8-v15 (low 64 bits) are CALLEE-SAVED per AAPCS64, but the SMSTART/SMSTOP
//   streaming-mode toggle destroys ALL of d8-d15 on Apple M4 (measured).  The
//   prologue/epilogue therefore spill d8-d15 themselves, outside streaming mode.
// * ZA is enabled, zeroed, accumulated into and left zeroed: caller ZA is not
//   preserved (legal -- ZA is caller-saved in the SME ABI).
// * PSTATE.SM and PSTATE.ZA are restored to 0 by the bare `smstop` (0xD503467F),
//   so this kernel must only be entered from a non-streaming context.
// * Bare `smstart` (0xD503477F) sets BOTH PSTATE.SM and PSTATE.ZA.
// * Hard-wired to a 512-bit streaming vector length (RDSVL == 64); on any other
//   SVL the entry point bails out without touching C.
// ---------------------------------------------------------------------------

    .text
    .p2align 2

// extern "C" void huohua_sme_microkernel_f16_32x32(
//     const __fp16* A_panel,  // x0
//     int lda,                // w1  (A panel row stride, in __fp16)
//     const __fp16* B_panel,  // x2
//     int ldb,                // w3  (B panel row stride, in __fp16)
//     float* C,               // x4
//     int ldc,                // w5  (C leading dim, in float)
//     int kc_groups);         // w6  (number of k-pairs)
    .globl _huohua_sme_microkernel_f16_32x32
_huohua_sme_microkernel_f16_32x32:

    rdsvl x15, #1                   // streaming vector length, in bytes
    cmp  x15, #64
    b.ne 8f                         // unsupported SVL -> no-op

    // Spill the callee-saved (but M4-destroyed) d8-d15 around the streaming
    // section.  Must happen outside streaming mode.
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // Byte strides.  lda/ldb/ldc are C `int` (only w-registers are defined).
    sxtw x9,  w1
    lsl  x9,  x9, #1                // a_stride = lda * 2
    sxtw x10, w3
    lsl  x10, x10, #1               // b_stride = ldb * 2
    sxtw x11, w5
    lsl  x11, x11, #2               // c_stride = ldc * 4

    smstart                         // enter streaming mode once

    mov  x13, x0                    // a row pointer (group = 0)
    mov  x14, x2                    // b row pointer (group = 0)

    ptrue p1.h                      // 32 half-lanes: widening FMOPA predicate
    ptrue pn8.h                     // 32 half-lanes: multi-vector load predicate
    ptrue p2.s                      // 16 word-lanes: epilogue
    zero {za}

    cbz  w6, 2f                     // kc == 0 -> straight to epilogue

    // ---------------- K (inner) accumulation loop, one group = 2 k ----------------
    //
    // SME2/SVE2.1 multi-vector loads (paper 4.2): one `ld1h` pulls a whole
    // 64-half group strip (128 B = 2 x 64 B blocks) into two consecutive Z
    // registers, so the loop issues 2 loads instead of 4 per group.
    //
    // Assembly syntax note: the assembler only accepts the hyphen *range* form
    // with a `pn8`-style predicate (`pn8/z`), i.e. `{z0.h-z1.h}, pn8/z`.  The
    // comma form / a plain `p0/z` is rejected as an invalid operand.  Measured
    // on this M4 these loads execute correctly inside streaming mode (encoding
    // 0xa04021a0 for `ld1h {z0.h-z1.h}, pn8/z, [x13]`).
    //
    // NOTE: explicit software pipelining (double-buffering the whole group in a
    // second register bank) gave no gain -- the core's out-of-order window
    // already hides the load latency, so the simpler loop is kept.
1:  ld1h {z0.h-z1.h}, pn8/z, [x13]        // a0 = A rows 0-15, a1 = A rows 16-31
    ld1h {z2.h-z3.h}, pn8/z, [x14]        // b0 = B cols 0-15, b1 = B cols 16-31

    fmopa za0.s, p1/m, p1/m, z0.h, z2.h // C[0:16][0:16]  += a0*b0 (2-way k)
    fmopa za1.s, p1/m, p1/m, z0.h, z3.h // C[0:16][16:32] += a0*b1
    fmopa za2.s, p1/m, p1/m, z1.h, z2.h // C[16:32][0:16] += a1*b0
    fmopa za3.s, p1/m, p1/m, z1.h, z3.h // C[16:32][16:32]+= a1*b1

    add  x13, x13, x9                  // advance a group by lda halves
    add  x14, x14, x10                 // advance b group by ldb halves
    subs w6, w6, #1
    bne  1b
2:
    // ---------------- Epilogue: read ZA back into C ----------------
    mov  w12, #0
    mov  x16, x4                       // cptr = C
3:  mov z4.s, p2/m, za0h.s[w12, 0]       // ZA0 row w12 -> 16 lanes
    mov z5.s, p2/m, za1h.s[w12, 0]       // ZA1 row w12 -> 16 lanes
    st1w {z4.s}, p2, [x16]             // C[r][0:16]
    st1w {z5.s}, p2, [x16, #1, MUL VL] // C[r][16:32]
    add  x16, x16, x11                 // next C row (ldc FP32s)
    add  w12, w12, #1
    cmp  w12, #16
    blt  3b

    // Lower 16 rows come from ZA2 (cols 0-15) and ZA3 (cols 16-31).
    mov  w12, #0
    lsl  x15, x11, #4                  // 16 * ldc (FP32) in bytes
    add  x16, x4, x15                  // cptr = C + 16*ldc
4:  mov z4.s, p2/m, za2h.s[w12, 0]       // ZA2 row w12
    mov z5.s, p2/m, za3h.s[w12, 0]       // ZA3 row w12
    st1w {z4.s}, p2, [x16]             // C[16+r][0:16]
    st1w {z5.s}, p2, [x16, #1, MUL VL] // C[16+r][16:32]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #16
    blt  4b

    smstop                          // leave streaming mode

    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret

8:  ret                             // unsupported SVL: leave C untouched

// ---------------------------------------------------------------------------
// unsigned huohua_sme_svl_bytes(void)
//   Returns the streaming SVE vector length in bytes (RDSVL).  Readable
//   outside streaming mode; requires FEAT_SME.  (Shared by every SME kernel;
//   provided here so the FP16 build needs no other assembly object.)
// ---------------------------------------------------------------------------
    .globl _huohua_sme_svl_bytes
_huohua_sme_svl_bytes:
    rdsvl x0, #1
    ret
