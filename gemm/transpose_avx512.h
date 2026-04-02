#ifndef TRANSPOSE_AVX512_H
#define TRANSPOSE_AVX512_H

#include <immintrin.h>
#include <algorithm>
#include "compiler_compat.h"

#ifdef __AVX512F__
/*
 * In-register transpose of an 8x8 block of doubles using AVX-512.
 *
 * Three rounds of shuffles:
 *   Round 1: _mm512_unpacklo/hi_pd  — interleave pairs of rows within 128-bit lanes
 *   Round 2: _mm512_shuffle_f64x2   — rearrange 128-bit lanes across row-pairs (0,2 vs 1,3)
 *   Round 3: _mm512_shuffle_f64x2   — final lane rearrangement to produce columns
 *
 * IMM8 constants:
 *   0x88 = 0b10_00_10_00  -> select lanes {src1[0], src1[2], src2[0], src2[2]}
 *   0xDD = 0b11_01_11_01  -> select lanes {src1[1], src1[3], src2[1], src2[3]}
 */
static inline void transpose_8x8_kernel(const double* __restrict__ src,
                                         double* __restrict__ dst,
                                         int src_stride, int dst_stride)
{
    // Load 8 rows
    __m512d r0 = _mm512_loadu_pd(src + 0 * src_stride);
    __m512d r1 = _mm512_loadu_pd(src + 1 * src_stride);
    __m512d r2 = _mm512_loadu_pd(src + 2 * src_stride);
    __m512d r3 = _mm512_loadu_pd(src + 3 * src_stride);
    __m512d r4 = _mm512_loadu_pd(src + 4 * src_stride);
    __m512d r5 = _mm512_loadu_pd(src + 5 * src_stride);
    __m512d r6 = _mm512_loadu_pd(src + 6 * src_stride);
    __m512d r7 = _mm512_loadu_pd(src + 7 * src_stride);

    // Round 1: unpack pairs of rows within 128-bit lanes
    __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    // Round 2: shuffle 128-bit lanes across row-group pairs (rows 0-1 with 2-3, 4-5 with 6-7)
    __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    __m512d u1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d u3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    __m512d u5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d u7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    // Round 3: final lane shuffle to produce transposed columns
    __m512d c0 = _mm512_shuffle_f64x2(u0, u4, 0x88);
    __m512d c1 = _mm512_shuffle_f64x2(u1, u5, 0x88);
    __m512d c2 = _mm512_shuffle_f64x2(u2, u6, 0x88);
    __m512d c3 = _mm512_shuffle_f64x2(u3, u7, 0x88);
    __m512d c4 = _mm512_shuffle_f64x2(u0, u4, 0xDD);
    __m512d c5 = _mm512_shuffle_f64x2(u1, u5, 0xDD);
    __m512d c6 = _mm512_shuffle_f64x2(u2, u6, 0xDD);
    __m512d c7 = _mm512_shuffle_f64x2(u3, u7, 0xDD);

    // Store 8 transposed columns as rows of dst
    _mm512_storeu_pd(dst + 0 * dst_stride, c0);
    _mm512_storeu_pd(dst + 1 * dst_stride, c1);
    _mm512_storeu_pd(dst + 2 * dst_stride, c2);
    _mm512_storeu_pd(dst + 3 * dst_stride, c3);
    _mm512_storeu_pd(dst + 4 * dst_stride, c4);
    _mm512_storeu_pd(dst + 5 * dst_stride, c5);
    _mm512_storeu_pd(dst + 6 * dst_stride, c6);
    _mm512_storeu_pd(dst + 7 * dst_stride, c7);
}
#endif // __AVX512F__

inline void transpose_avx512(const double* A, double* AT, int n) noexcept {
#ifdef __AVX512F__
    const int TILE = 8;

    // Full 8x8 blocks
    int n_full = n - (n % TILE);
    for (int ti = 0; ti < n_full; ti += TILE) {
        for (int tj = 0; tj < n_full; tj += TILE) {
            transpose_8x8_kernel(A + ti * n + tj, AT + tj * n + ti, n, n);
        }
    }

    // Right remainder strip: columns [n_full, n), rows [0, n_full)
    for (int i = 0; i < n_full; ++i) {
        for (int j = n_full; j < n; ++j) {
            AT[j * n + i] = A[i * n + j];
        }
    }

    // Bottom remainder strip: rows [n_full, n), all columns
    for (int i = n_full; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT[j * n + i] = A[i * n + j];
        }
    }

#else
    // Scalar fallback with 64x64 tiling
    const int TILE = 64;
    int n_full = n - (n % TILE);

    for (int ti = 0; ti < n_full; ti += TILE) {
        for (int tj = 0; tj < n_full; tj += TILE) {
            for (int i = ti; i < ti + TILE; ++i) {
                for (int j = tj; j < tj + TILE; ++j) {
                    AT[j * n + i] = A[i * n + j];
                }
            }
        }
    }

    // Right remainder strip
    for (int i = 0; i < n_full; ++i) {
        for (int j = n_full; j < n; ++j) {
            AT[j * n + i] = A[i * n + j];
        }
    }

    // Bottom remainder strip
    for (int i = n_full; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT[j * n + i] = A[i * n + j];
        }
    }
#endif
}

#endif // TRANSPOSE_AVX512_H
