// ---------------------------------------------------------------------------
// assemble_i8.s -- SME signed INT8 32x32 micro-kernel with INT32 output.
//
// Each packed K group contains four adjacent K values per output row/column.
// A 64-byte vector therefore covers 16 rows or columns, and one SMOPA adds
// four signed byte products to each ZA32 element. Four ZA32 tiles cover 32x32.
// ---------------------------------------------------------------------------

    .text
    .p2align 2

// extern "C" void huohua_sme_microkernel_i8_32x32(
//     const int8_t* A_panel, int lda, const int8_t* B_panel, int ldb,
//     int32_t* C, int ldc, int kc_groups);
    .globl _huohua_sme_microkernel_i8_32x32
_huohua_sme_microkernel_i8_32x32:
    rdsvl x15, #1
    cmp   x15, #64
    b.ne  8f

    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    sxtw x9,  w1                   // A group stride in bytes
    sxtw x10, w3                   // B group stride in bytes
    sxtw x11, w5
    lsl  x11, x11, #2              // C row stride in bytes

    smstart

    mov   x13, x0
    mov   x14, x2
    ptrue p0.b
    ptrue p1.s
    zero  {za}

    cbz w6, 2f

1:  ld1b {z0.b}, p0/z, [x13]
    ld1b {z1.b}, p0/z, [x13, #1, MUL VL]
    ld1b {z2.b}, p0/z, [x14]
    ld1b {z3.b}, p0/z, [x14, #1, MUL VL]

    smopa za0.s, p0/m, p0/m, z0.b, z2.b
    smopa za1.s, p0/m, p0/m, z0.b, z3.b
    smopa za2.s, p0/m, p0/m, z1.b, z2.b
    smopa za3.s, p0/m, p0/m, z1.b, z3.b

    add  x13, x13, x9
    add  x14, x14, x10
    subs w6, w6, #1
    bne  1b

2:  mov w12, #0
    mov x16, x4
3:  mov z4.s, p1/m, za0h.s[w12, 0]
    mov z5.s, p1/m, za1h.s[w12, 0]
    st1w {z4.s}, p1, [x16]
    st1w {z5.s}, p1, [x16, #1, MUL VL]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #16
    blt  3b

    mov w12, #0
    lsl x15, x11, #4
    add x16, x4, x15
4:  mov z4.s, p1/m, za2h.s[w12, 0]
    mov z5.s, p1/m, za3h.s[w12, 0]
    st1w {z4.s}, p1, [x16]
    st1w {z5.s}, p1, [x16, #1, MUL VL]
    add  x16, x16, x11
    add  w12, w12, #1
    cmp  w12, #16
    blt  4b

    smstop

    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret

8:  ret
