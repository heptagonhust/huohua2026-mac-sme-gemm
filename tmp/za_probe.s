// Temporary diagnostic #3: which SME instruction clobbers v11/v12/v13?
//
// Sets z11/z12/z13 to distinctive patterns, then replays the micro-kernel's
// instruction sequence one step at a time, dumping the low 64 bits of
// z11/z12/z13 after every step.  The first checkpoint whose pattern changed
// identifies the guilty instruction.
//
//   void za_probe(unsigned long long* out);   // out[27], 9 checkpoints x 3
    .data
    .p2align 6
scratch:
    .space 1024

    .text
    .p2align 2

.macro DUMP a, b, c
    fmov x9, d11
    str  x9, [x10, #\a]
    fmov x9, d12
    str  x9, [x10, #\b]
    fmov x9, d13
    str  x9, [x10, #\c]
.endm

// void za_probe(unsigned long long* out);
    .globl _za_probe
_za_probe:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #16
    mov x10, x0                     // out

    adrp x13, scratch@PAGE
    add  x13, x13, scratch@PAGEOFF
    mov  x14, x13
    mov  x16, x13

    // z11 = 0x1111111111111111, z12 = 0x2222..., z13 = 0x3333...
    movz x9, #0x1111, lsl #48
    movk x9, #0x1111, lsl #32
    movk x9, #0x1111, lsl #16
    movk x9, #0x1111
    fmov d11, x9
    dup  v11.2d, v11.d[0]

    movz x9, #0x2222, lsl #48
    movk x9, #0x2222, lsl #32
    movk x9, #0x2222, lsl #16
    movk x9, #0x2222
    fmov d12, x9
    dup  v12.2d, v12.d[0]

    movz x9, #0x3333, lsl #48
    movk x9, #0x3333, lsl #32
    movk x9, #0x3333, lsl #16
    movk x9, #0x3333
    fmov d13, x9
    dup  v13.2d, v13.d[0]

    // A: patterns installed
    DUMP 0, 8, 16

    smstart
    // B: after smstart
    DUMP 24, 32, 40

    .long 0x2598e3e0                // ptrue p0.s
    // C: after ptrue p0.s
    DUMP 48, 56, 64

    mov  w12, #0
    .long 0xc00800ff                // zero {za}  (as emitted by the assembler)
    // D: after zero {za}
    DUMP 72, 80, 88

    .long 0xa540a1a0                // ld1w {z0.s}, p0/z, [x13]
    .long 0xa541a1a1                // ld1w {z1.s}, p0/z, [x13, #1, MUL VL]
    .long 0xa540a1c2                // ld1w {z2.s}, p0/z, [x14]
    .long 0xa541a1c3                // ld1w {z3.s}, p0/z, [x14, #1, MUL VL]
    // E: after the four ld1w
    DUMP 96, 104, 112

    .long 0x80820000                // fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    .long 0x80830001
    .long 0x80820022
    .long 0x80830023
    // F: after the four fmopa
    DUMP 120, 128, 136

    .long 0xc0820004                // mov z4.s, p0/m, za0h.s[w12, 0]
    .long 0xc0820085                // mov z5.s, p0/m, za1h.s[w12, 0]
    // G: after the two mova (ZA -> z4/z5)
    DUMP 144, 152, 160

    .long 0xe540e204                // st1w {z4.s}, p0, [x16]
    .long 0xe541e205                // st1w {z5.s}, p0, [x16, #1, MUL VL]
    // H: after the two st1w
    DUMP 168, 176, 184

    smstop
    // I: after smstop
    DUMP 192, 200, 208

    add sp, sp, #16
    ldp x29, x30, [sp], #16
    ret
