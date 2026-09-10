// Temporary diagnostic #4: complete ABI audit of the streaming-mode toggle.
//
// AAPCS64 (+SVE/SME PCS) callee-saved state that a leaf asm kernel must keep:
//   x19-x28, x29(FP), x30(LR), sp, and the low 64 bits of v8-v15.
//
// This probe fills those registers with distinctive patterns, executes exactly
// SMSTART + SMSTOP (nothing else), and dumps them again.
//
//   void abi_audit(unsigned long long* out);
//     out[ 0..15] : v8..v15 low/high 64 bits, before
//     out[16..31] : v8..v15 low/high 64 bits, after  SMSTART+SMSTOP
//     out[32..41] : x19..x28 before
//     out[42..51] : x19..x28 after
    .text
    .p2align 2

// AAPCS64 only requires the low 64 bits of v8-v15 to be preserved.
.macro DUMPV n, off
    fmov x9, d\n
    str  x9, [x10, #\off]
.endm

.macro DUMPX n, off
    str  x\n, [x10, #\off]
.endm

    .globl _abi_audit
_abi_audit:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #96
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    mov x10, x0                     // out

    // v8..v15 = 0x1111..0x8888 replicated per byte
    movi v8.16b,  #0x11
    movi v9.16b,  #0x22
    movi v10.16b, #0x33
    movi v11.16b, #0x44
    movi v12.16b, #0x55
    movi v13.16b, #0x66
    movi v14.16b, #0x77
    movi v15.16b, #0x88

    // x19..x28 = 0x19..0x28 << 8  (distinct, easy to spot)
    mov x19, #0x1900
    mov x20, #0x2000
    mov x21, #0x2100
    mov x22, #0x2200
    mov x23, #0x2300
    mov x24, #0x2400
    mov x25, #0x2500
    mov x26, #0x2600
    mov x27, #0x2700
    mov x28, #0x2800

    DUMPV 8, 0
    DUMPV 9, 16
    DUMPV 10, 32
    DUMPV 11, 48
    DUMPV 12, 64
    DUMPV 13, 80
    DUMPV 14, 96
    DUMPV 15, 112
    DUMPX 19, 256
    DUMPX 20, 264
    DUMPX 21, 272
    DUMPX 22, 280
    DUMPX 23, 288
    DUMPX 24, 296
    DUMPX 25, 304
    DUMPX 26, 312
    DUMPX 27, 320
    DUMPX 28, 328

    smstart                          // <-- the only SME instruction executed
    smstop

    DUMPV 8, 128
    DUMPV 9, 144
    DUMPV 10, 160
    DUMPV 11, 176
    DUMPV 12, 192
    DUMPV 13, 208
    DUMPV 14, 224
    DUMPV 15, 240
    DUMPX 19, 336
    DUMPX 20, 344
    DUMPX 21, 352
    DUMPX 22, 360
    DUMPX 23, 368
    DUMPX 24, 376
    DUMPX 25, 384
    DUMPX 26, 392
    DUMPX 27, 400
    DUMPX 28, 408

    // the probe deliberately clobbered x19-x28; restore the caller's values
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    add sp, sp, #96
    ldp x29, x30, [sp], #16
    ret
