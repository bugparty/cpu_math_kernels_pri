; =============================================================================
; transpose_8x8_avx2.asm
;
; Transcribed from profiler disassembly (Block 22).
; Assemble with NASM:  nasm -f elf64   transpose_8x8_avx2.asm  (Linux)
;                      nasm -f macho64 transpose_8x8_avx2.asm  (macOS)
;
; C declaration (put this in a header):
;
;   #include <stddef.h>
;   void transpose_8x8_double_avx2(
;       const double *src,        // 32-byte aligned
;       double       *dst,        // 32-byte aligned
;       ptrdiff_t     src_stride, // row stride in bytes, multiple of 32
;       ptrdiff_t     dst_stride);
;
; Calling convention: System V AMD64  (Linux / macOS)
;   rdi = src   rsi = dst   rdx = src_stride   rcx = dst_stride
; =============================================================================

; -----------------------------------------------------------------------------
; Algorithm overview
; ------------------
; Each source row is split into two 32-byte halves:
;   low  half  col 0-3  (byte offset  0)
;   high half  col 4-7  (byte offset 32 = 0x20)
;
; Each half is transposed in two stages:
;
;   Stage A  vunpcklpd / vunpckhpd
;     Interleave 64-bit elements within each 128-bit lane:
;       unpacklo(a, b) -> [a[0],b[0] | a[2],b[2]]
;       unpackhi(a, b) -> [a[1],b[1] | a[3],b[3]]
;
;   Stage B  vinsertf128 / vperm2f128  (or vinserti128 / vperm2i128)
;     Merge 128-bit lanes across register pairs to produce complete rows:
;       insertf128(ymm_lo, xmm_hi, 1) -> [ymm_lo.lane0 | xmm_hi]
;       perm2f128 (a, b, 0x31)        -> [a.lane1      | b.lane1]
;
;   Stores  vmovntps (low half)  /  vmovntdq (high half)
;     Non-temporal writes bypass the LLC to avoid cache pollution.
;     The high half uses integer-domain instructions (vpunpcklqdq,
;     vinserti128, vmovntdq), matching the original binary.
;
; Live-range constraint
; ---------------------
;   Stage B for rows 0-3 overwrites ymm13, which also holds row7_lo
;   after the initial load.  Therefore Stage A for rows 4-7 MUST
;   complete before Stage B for rows 0-3.  The ordering below enforces
;   this constraint.
;
; Register map ！ low half
; -----------------------
;   Load :  ymm0  row0_lo   ymm1  row1_lo   ymm2  row2_lo   ymm3  row3_lo
;           ymm4  row4_lo   ymm5  row5_lo   ymm6  row6_lo   ymm7  row7_lo
;   Temp :  ymm14  Stage-A scratch (rows 0-3)
;           ymm15  Stage-A scratch (rows 4-7)
;   After Stage B:
;     ymm8  out_row0 col0-3     ymm12 out_row0 col4-7
;     ymm9  out_row1 col0-3     ymm13 out_row1 col4-7
;     ymm10 out_row2 col0-3     ymm14 out_row2 col4-7
;     ymm11 out_row3 col0-3     ymm0  out_row3 col4-7
;
; Register map ！ high half  (ymm0-ymm7 reloaded after low-half stores)
; -----------------------
;   Same layout as low half; after Stage B:
;     ymm8  out_row4 col0-3     ymm12 out_row4 col4-7
;     ymm9  out_row5 col0-3     ymm13 out_row5 col4-7
;     ymm10 out_row6 col0-3     ymm14 out_row6 col4-7
;     ymm11 out_row7 col0-3     ymm0  out_row7 col4-7
; -----------------------------------------------------------------------------

global transpose_8x8_double_avx2

section .text

transpose_8x8_double_avx2:

    ; -------------------------------------------------------------------------
    ; Prologue ！ save callee-saved registers we clobber
    ; -------------------------------------------------------------------------
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; -------------------------------------------------------------------------
    ; Assign register roles and precompute src row offsets
    ;
    ;   r8  = src base       r9  = src_stride
    ;   r15 = dst_stride     rsi = dst base  (already set by ABI)
    ;   r10 = dst byte-offset counter (0 .. 7*dst_stride)
    ;
    ;   rax = 2*src_stride   rbx = 3*src_stride
    ;   r11 = 4*src_stride   r12 = 5*src_stride
    ;   r13 = 6*src_stride   r14 = 7*src_stride
    ; -------------------------------------------------------------------------
    mov     r8,  rdi
    mov     r9,  rdx
    mov     r15, rcx
    xor     r10, r10

    lea     rax, [r9  + r9]
    lea     rbx, [rax + r9]
    lea     r11, [rax + rax]
    lea     r12, [r11 + r9]
    lea     r13, [r11 + rax]
    lea     r14, [r11 + rbx]

    ; -------------------------------------------------------------------------
    ; Prefetch (NTA = Non-Temporal All)
    ;
    ;   +8 skips the first 8 bytes (already pulled in by the vmovups below)
    ;   and targets the second cache line of each source row.  The NTA hint
    ;   loads into L1 without allocating in L2/L3, preventing once-used
    ;   source data from evicting other working-set cache lines.
    ; -------------------------------------------------------------------------
    prefetchnta [r8          + 8]
    prefetchnta [r8 + r9     + 8]
    prefetchnta [r8 + rax    + 8]
    prefetchnta [r8 + rbx    + 8]
    prefetchnta [r8 + r11    + 8]
    prefetchnta [r8 + r12    + 8]
    prefetchnta [r8 + r13    + 8]
    prefetchnta [r8 + r14    + 8]

    ; =========================================================================
    ; LOW HALF  (col 0-3, byte offset +0)
    ; =========================================================================

    ; --- load ----------------------------------------------------------------
    vmovups ymm0, [r8]
    vmovups ymm1, [r8 + r9]
    vmovups ymm2, [r8 + rax]
    vmovdqu ymm3, [r8 + rbx]
    vmovups ymm4, [r8 + r11]
    vmovdqu ymm5, [r8 + r12]
    vmovdqu ymm6, [r8 + r13]
    vmovdqu ymm7, [r8 + r14]

    ; --- Stage A, rows 4-7  (must precede Stage B for rows 0-3) -------------
    ;   ymm15 = [r4c0,r5c0 | r4c2,r5c2]     ymm4  = [r4c1,r5c1 | r4c3,r5c3]
    ;   ymm5  = [r6c0,r7c0 | r6c2,r7c2]     ymm6  = [r6c1,r7c1 | r6c3,r7c3]
    vunpcklpd ymm15, ymm4, ymm5
    vunpckhpd ymm4,  ymm4, ymm5
    vunpcklpd ymm5,  ymm6, ymm7
    vunpckhpd ymm6,  ymm6, ymm7

    ; --- Stage A, rows 0-3 ---------------------------------------------------
    ;   ymm14 = [r0c0,r1c0 | r0c2,r1c2]     ymm0  = [r0c1,r1c1 | r0c3,r1c3]
    ;   ymm1  = [r2c0,r3c0 | r2c2,r3c2]     ymm2  = [r2c1,r3c1 | r2c3,r3c3]
    vunpcklpd ymm14, ymm0, ymm1
    vunpckhpd ymm0,  ymm0, ymm1
    vunpcklpd ymm1,  ymm2, ymm3
    vunpckhpd ymm2,  ymm2, ymm3

    ; --- Stage B, rows 0-3 -> output rows 0-3, col 0-3 ----------------------
    ;   vinsertf128 dst, src_ymm, src_xmm, 1
    ;     dst.lane0 = src_ymm.lane0 ;  dst.lane1 = src_xmm
    ;   vperm2f128 dst, a, b, 0x31
    ;     dst.lane0 = a.lane1       ;  dst.lane1 = b.lane1
    ;
    ;   ymm8  = [r0c0,r1c0,r2c0,r3c0]   ymm9  = [r0c1,r1c1,r2c1,r3c1]
    ;   ymm10 = [r0c2,r1c2,r2c2,r3c2]   ymm11 = [r0c3,r1c3,r2c3,r3c3]
    vinsertf128 ymm8,  ymm14, xmm1,  1
    vperm2f128  ymm10, ymm14, ymm1,  0x31
    vinsertf128 ymm9,  ymm0,  xmm2,  1
    vperm2f128  ymm11, ymm0,  ymm2,  0x31

    ; --- Stage B, rows 4-7 -> output rows 0-3, col 4-7 ----------------------
    ;   ymm12 = [r4c0,r5c0,r6c0,r7c0]   ymm13 = [r4c1,r5c1,r6c1,r7c1]
    ;   ymm14 = [r4c2,r5c2,r6c2,r7c2]   ymm0  = [r4c3,r5c3,r6c3,r7c3]
    vinsertf128 ymm12, ymm15, xmm5,  1
    vperm2f128  ymm14, ymm15, ymm5,  0x31
    vinsertf128 ymm13, ymm4,  xmm6,  1
    vperm2f128  ymm0,  ymm4,  ymm6,  0x31

    ; --- NT stores, output rows 0-3 ------------------------------------------
    ;   vmovntps: non-temporal packed-float store, 32 bytes.
    ;   Two writes per output row (col 0-3 at +0x00, col 4-7 at +0x20).
    vmovntps [rsi + r10],        ymm8
    vmovntps [rsi + r10 + 0x20], ymm12
    add      r10, r15

    vmovntps [rsi + r10],        ymm9
    vmovntps [rsi + r10 + 0x20], ymm13
    add      r10, r15

    vmovntps [rsi + r10],        ymm10
    vmovntps [rsi + r10 + 0x20], ymm14
    add      r10, r15

    vmovntps [rsi + r10],        ymm11
    vmovntps [rsi + r10 + 0x20], ymm0
    add      r10, r15

    ; =========================================================================
    ; HIGH HALF  (col 4-7, byte offset +0x20)
    ;
    ;   ymm0-ymm7 are now free.  Integer-domain instructions are used
    ;   throughout, matching the original binary's second shuffle block.
    ;   vpunpcklqdq / vpunpckhqdq are semantically identical to
    ;   vunpcklpd / vunpckhpd at 64-bit granularity.
    ;   vinserti128 / vperm2i128 are the integer equivalents of
    ;   vinsertf128 / vperm2f128.
    ; =========================================================================

    ; --- load ----------------------------------------------------------------
    vmovdqu ymm0, [r8          + 0x20]
    vmovdqu ymm1, [r8 + r9     + 0x20]
    vmovdqu ymm2, [r8 + rax    + 0x20]
    vmovdqu ymm3, [r8 + rbx    + 0x20]
    vmovdqu ymm4, [r8 + r11    + 0x20]
    vmovdqu ymm5, [r8 + r12    + 0x20]
    vmovdqu ymm6, [r8 + r13    + 0x20]
    vmovdqu ymm7, [r8 + r14    + 0x20]

    ; --- Stage A, rows 4-7 ---------------------------------------------------
    ;   ymm15 = [r4c4,r5c4 | r4c6,r5c6]     ymm4  = [r4c5,r5c5 | r4c7,r5c7]
    ;   ymm5  = [r6c4,r7c4 | r6c6,r7c6]     ymm6  = [r6c5,r7c5 | r6c7,r7c7]
    vpunpcklqdq ymm15, ymm4, ymm5
    vpunpckhqdq ymm4,  ymm4, ymm5
    vpunpcklqdq ymm5,  ymm6, ymm7
    vpunpckhqdq ymm6,  ymm6, ymm7

    ; --- Stage A, rows 0-3 ---------------------------------------------------
    ;   ymm14 = [r0c4,r1c4 | r0c6,r1c6]     ymm0  = [r0c5,r1c5 | r0c7,r1c7]
    ;   ymm1  = [r2c4,r3c4 | r2c6,r3c6]     ymm2  = [r2c5,r3c5 | r2c7,r3c7]
    vpunpcklqdq ymm14, ymm0, ymm1
    vpunpckhqdq ymm0,  ymm0, ymm1
    vpunpcklqdq ymm1,  ymm2, ymm3
    vpunpckhqdq ymm2,  ymm2, ymm3

    ; --- Stage B, rows 0-3 -> output rows 4-7, col 0-3 ----------------------
    ;   ymm8  = [r0c4,r1c4,r2c4,r3c4]   ymm9  = [r0c5,r1c5,r2c5,r3c5]
    ;   ymm10 = [r0c6,r1c6,r2c6,r3c6]   ymm11 = [r0c7,r1c7,r2c7,r3c7]
    vinserti128 ymm8,  ymm14, xmm1,  1
    vperm2i128  ymm10, ymm14, ymm1,  0x31
    vinserti128 ymm9,  ymm0,  xmm2,  1
    vperm2i128  ymm11, ymm0,  ymm2,  0x31

    ; --- Stage B, rows 4-7 -> output rows 4-7, col 4-7 ----------------------
    ;   ymm12 = [r4c4,r5c4,r6c4,r7c4]   ymm13 = [r4c5,r5c5,r6c5,r7c5]
    ;   ymm14 = [r4c6,r5c6,r6c6,r7c6]   ymm0  = [r4c7,r5c7,r6c7,r7c7]
    vinserti128 ymm12, ymm15, xmm5,  1
    vperm2i128  ymm14, ymm15, ymm5,  0x31
    vinserti128 ymm13, ymm4,  xmm6,  1
    vperm2i128  ymm0,  ymm4,  ymm6,  0x31

    ; --- NT stores, output rows 4-7 ------------------------------------------
    ;   vmovntdq: non-temporal integer store, 32 bytes.
    ;   r10 already points at row 4 after the four adds in the low-half block.
    vmovntdq [rsi + r10],        ymm8
    vmovntdq [rsi + r10 + 0x20], ymm12
    add      r10, r15

    vmovntdq [rsi + r10],        ymm9
    vmovntdq [rsi + r10 + 0x20], ymm13
    add      r10, r15

    vmovntdq [rsi + r10],        ymm10
    vmovntdq [rsi + r10 + 0x20], ymm14
    add      r10, r15

    vmovntdq [rsi + r10],        ymm11
    vmovntdq [rsi + r10 + 0x20], ymm0

    ; -------------------------------------------------------------------------
    ; sfence ！ flush NT store buffers before returning.
    ;   Guarantees all vmovntps / vmovntdq writes are globally visible.
    ;   The caller does not need to issue an additional fence.
    ; -------------------------------------------------------------------------
    sfence

    ; -------------------------------------------------------------------------
    ; Epilogue
    ; -------------------------------------------------------------------------
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
