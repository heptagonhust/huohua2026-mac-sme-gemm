// ---------------------------------------------------------------------------
// assemble_f64.s -- SME FP64 16x16 micro-kernel (single SME worker / P-cluster).
//
// Computes and overwrites one full 16x16 output tile from k-major FP64 panels:
//   C(16x16 tile, row-major, leading dim ldc) = sum_k A_panel(k,:) (x) B_panel(k,:)
//
// A 512-bit streaming vector contains 8 FP64 lanes. Four ZA64 8x8 tiles cover
// the output:
//   ZA0 -> C[0..7][0..7]   ZA1 -> C[0..7][8..15]
//   ZA2 -> C[8..15][0..7] ZA3 -> C[8..15][8..15]
//
// The entry point must be called from a non-streaming context. It preserves
// d8-d15 around the streaming-mode transition, clobbers caller-saved ZA, and
// returns with PSTATE.SM and PSTATE.ZA disabled.
// ---------------------------------------------------------------------------

    .text
    .p2align 2

// extern "C" void huohua_sme_microkernel_f64_16x16(
//     const double* A_panel, int lda, const double* B_panel, int ldb,
//     double* C, int ldc, int kc);
    .globl _huohua_sme_microkernel_f64_16x16
_huohua_sme_microkernel_f64_16x16:
    rdsvl x15, #1
    cmp   x15, #64
    b.ne  8f

    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    sxtw x9,  w1
    lsl  x9,  x9, #3                // a_stride = lda * sizeof(double)
    sxtw x10, w3
    lsl  x10, x10, #3               // b_stride = ldb * sizeof(double)
    sxtw x11, w5
    lsl  x11, x11, #3               // c_stride = ldc * sizeof(double)

    smstart

    mov   x13, x0
    mov   x14, x2
    ptrue p0.d
    zero  {za}

    cbz w6, 2f

1:  ld1d {z0.d}, p0/z, [x13]
    ld1d {z1.d}, p0/z, [x13, #1, MUL VL]
    ld1d {z2.d}, p0/z, [x14]
    ld1d {z3.d}, p0/z, [x14, #1, MUL VL]

    fmopa za0.d, p0/m, p0/m, z0.d, z2.d
    fmopa za1.d, p0/m, p0/m, z0.d, z3.d
    fmopa za2.d, p0/m, p0/m, z1.d, z2.d
    fmopa za3.d, p0/m, p0/m, z1.d, z3.d

    add  x13, x13, x9
    add  x14, x14, x10
    subs w6, w6, #1
    bne  1b

2:  mov w12, #0
    mov x16, x4
3:  mov z4.d, p0/m, za0h.d[w12, 0]
    mov z5.d, p0/m, za1h.d[w12, 0]
    st1d {z4.d}, p0, [x16]
    st1d {z5.d}, p0, [x16, #1, MUL VL]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #8
    blt  3b

    mov w12, #0
    lsl x15, x11, #3                 // 8 * ldc * sizeof(double)
    add x16, x4, x15
4:  mov z4.d, p0/m, za2h.d[w12, 0]
    mov z5.d, p0/m, za3h.d[w12, 0]
    st1d {z4.d}, p0, [x16]
    st1d {z5.d}, p0, [x16, #1, MUL VL]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #8
    blt  4b

    smstop

    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret

8:  ret
