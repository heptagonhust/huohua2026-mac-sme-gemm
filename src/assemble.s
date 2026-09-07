// ---------------------------------------------------------------------------
// assemble.s -- SME FP32 32x32 micro-kernel (single SME worker / P-cluster).
//
// Computes, for one full 32x32 output tile that was already packed k-major:
//   C(32xKout tile) += sum_k  A_panel(k, 0..31) (x) B_panel(k, 0..31)
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
// entered/exited once around the whole tile so the ~9ns toggle is amortized.
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

    // Byte strides.
    lsl x9,  x1, #2                 // a_stride = lda * 4
    lsl x10, x3, #2                 // b_stride = ldb * 4
    lsl x11, x5, #2                 // c_stride = ldc * 4

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
    ret
