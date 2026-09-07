	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 26, 2
	.globl	__Z5probeu13__SVFloat32_tu10__SVBool_t ; -- Begin function _Z5probeu13__SVFloat32_tu10__SVBool_t
	.p2align	2
__Z5probeu13__SVFloat32_tu10__SVBool_t: ; @_Z5probeu13__SVFloat32_tu10__SVBool_t
	.cfi_startproc
; %bb.0:
	rdsvl	x9, #1
	lsr	x9, x9, #3
	str	x9, [sp, #-32]!                 ; 8-byte Folded Spill
	.cfi_def_cfa_offset 32
	cntd	x9
	str	x9, [sp, #8]                    ; 8-byte Folded Spill
	str	x28, [sp, #16]                  ; 8-byte Folded Spill
	.cfi_offset w28, -16
	.cfi_offset vg, -24
	addsvl	sp, sp, #-18
	.cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0x90, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; sp + 32 + 144 * VG
	str	p15, [sp, #4, mul vl]           ; 2-byte Folded Spill
	str	p14, [sp, #5, mul vl]           ; 2-byte Folded Spill
	str	p13, [sp, #6, mul vl]           ; 2-byte Folded Spill
	str	p12, [sp, #7, mul vl]           ; 2-byte Folded Spill
	str	p11, [sp, #8, mul vl]           ; 2-byte Folded Spill
	str	p10, [sp, #9, mul vl]           ; 2-byte Folded Spill
	str	p9, [sp, #10, mul vl]           ; 2-byte Folded Spill
	str	p8, [sp, #11, mul vl]           ; 2-byte Folded Spill
	str	p7, [sp, #12, mul vl]           ; 2-byte Folded Spill
	str	p6, [sp, #13, mul vl]           ; 2-byte Folded Spill
	str	p5, [sp, #14, mul vl]           ; 2-byte Folded Spill
	str	p4, [sp, #15, mul vl]           ; 2-byte Folded Spill
	str	z23, [sp, #2, mul vl]           ; 16-byte Folded Spill
	str	z22, [sp, #3, mul vl]           ; 16-byte Folded Spill
	str	z21, [sp, #4, mul vl]           ; 16-byte Folded Spill
	str	z20, [sp, #5, mul vl]           ; 16-byte Folded Spill
	str	z19, [sp, #6, mul vl]           ; 16-byte Folded Spill
	str	z18, [sp, #7, mul vl]           ; 16-byte Folded Spill
	str	z17, [sp, #8, mul vl]           ; 16-byte Folded Spill
	str	z16, [sp, #9, mul vl]           ; 16-byte Folded Spill
	str	z15, [sp, #10, mul vl]          ; 16-byte Folded Spill
	str	z14, [sp, #11, mul vl]          ; 16-byte Folded Spill
	str	z13, [sp, #12, mul vl]          ; 16-byte Folded Spill
	str	z12, [sp, #13, mul vl]          ; 16-byte Folded Spill
	str	z11, [sp, #14, mul vl]          ; 16-byte Folded Spill
	str	z10, [sp, #15, mul vl]          ; 16-byte Folded Spill
	str	z9, [sp, #16, mul vl]           ; 16-byte Folded Spill
	str	z8, [sp, #17, mul vl]           ; 16-byte Folded Spill
	.cfi_escape 0x10, 0x48, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x78, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d8  @ cfa - 32 - 8 * VG
	.cfi_escape 0x10, 0x49, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x70, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d9  @ cfa - 32 - 16 * VG
	.cfi_escape 0x10, 0x4a, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x68, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d10  @ cfa - 32 - 24 * VG
	.cfi_escape 0x10, 0x4b, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x60, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d11  @ cfa - 32 - 32 * VG
	.cfi_escape 0x10, 0x4c, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x58, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d12  @ cfa - 32 - 40 * VG
	.cfi_escape 0x10, 0x4d, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x50, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d13  @ cfa - 32 - 48 * VG
	.cfi_escape 0x10, 0x4e, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x48, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d14  @ cfa - 32 - 56 * VG
	.cfi_escape 0x10, 0x4f, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x40, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; $d15  @ cfa - 32 - 64 * VG
	addsvl	sp, sp, #-8
	.cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0xd0, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; sp + 32 + 208 * VG
	str	p0, [sp, #15, mul vl]           ; 2-byte Folded Spill
	str	z0, [sp]                        ; 16-byte Folded Spill
	smstart	sm
	ldr	z0, [sp]                        ; 16-byte Folded Reload
	ldr	p1, [sp, #15, mul vl]           ; 2-byte Folded Reload
	ptrue	p0.s
	st1w	{ z0.s }, p0, [sp, #7, mul vl]
	str	p1, [sp, #55, mul vl]
	ldr	p1, [sp, #55, mul vl]
	ptrue	p2.h
	ld1h	{ z0.h }, p2/z, [sp, #5, mul vl]
	ld1h	{ z1.h }, p2/z, [sp, #4, mul vl]
	fmopa	za0.s, p1/m, p1/m, z0.h, z1.h
	mov	z0.s, #0                        ; =0x0
	ldr	p1, [sp, #55, mul vl]
	mov	w12, #5                         ; =0x5
	mov	z0.s, p1/m, za0h.s[w12, 0]
	st1w	{ z0.s }, p0, [sp, #3, mul vl]
	ld1w	{ z0.s }, p0/z, [sp, #3, mul vl]
	str	z0, [sp, #2, mul vl]            ; 16-byte Folded Spill
	smstop	sm
	ldr	z0, [sp, #2, mul vl]            ; 16-byte Folded Reload
	addsvl	sp, sp, #8
	.cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0x90, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 ; sp + 32 + 144 * VG
	ldr	z23, [sp, #2, mul vl]           ; 16-byte Folded Reload
	ldr	z22, [sp, #3, mul vl]           ; 16-byte Folded Reload
	ldr	z21, [sp, #4, mul vl]           ; 16-byte Folded Reload
	ldr	z20, [sp, #5, mul vl]           ; 16-byte Folded Reload
	ldr	z19, [sp, #6, mul vl]           ; 16-byte Folded Reload
	ldr	z18, [sp, #7, mul vl]           ; 16-byte Folded Reload
	ldr	z17, [sp, #8, mul vl]           ; 16-byte Folded Reload
	ldr	z16, [sp, #9, mul vl]           ; 16-byte Folded Reload
	ldr	z15, [sp, #10, mul vl]          ; 16-byte Folded Reload
	ldr	z14, [sp, #11, mul vl]          ; 16-byte Folded Reload
	ldr	z13, [sp, #12, mul vl]          ; 16-byte Folded Reload
	ldr	z12, [sp, #13, mul vl]          ; 16-byte Folded Reload
	ldr	z11, [sp, #14, mul vl]          ; 16-byte Folded Reload
	ldr	z10, [sp, #15, mul vl]          ; 16-byte Folded Reload
	ldr	z9, [sp, #16, mul vl]           ; 16-byte Folded Reload
	ldr	z8, [sp, #17, mul vl]           ; 16-byte Folded Reload
	ldr	p15, [sp, #4, mul vl]           ; 2-byte Folded Reload
	ldr	p14, [sp, #5, mul vl]           ; 2-byte Folded Reload
	ldr	p13, [sp, #6, mul vl]           ; 2-byte Folded Reload
	ldr	p12, [sp, #7, mul vl]           ; 2-byte Folded Reload
	ldr	p11, [sp, #8, mul vl]           ; 2-byte Folded Reload
	ldr	p10, [sp, #9, mul vl]           ; 2-byte Folded Reload
	ldr	p9, [sp, #10, mul vl]           ; 2-byte Folded Reload
	ldr	p8, [sp, #11, mul vl]           ; 2-byte Folded Reload
	ldr	p7, [sp, #12, mul vl]           ; 2-byte Folded Reload
	ldr	p6, [sp, #13, mul vl]           ; 2-byte Folded Reload
	ldr	p5, [sp, #14, mul vl]           ; 2-byte Folded Reload
	ldr	p4, [sp, #15, mul vl]           ; 2-byte Folded Reload
	addsvl	sp, sp, #18
	.cfi_def_cfa wsp, 32
	.cfi_restore z8
	.cfi_restore z9
	.cfi_restore z10
	.cfi_restore z11
	.cfi_restore z12
	.cfi_restore z13
	.cfi_restore z14
	.cfi_restore z15
	ldr	x28, [sp, #16]                  ; 8-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w28
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
