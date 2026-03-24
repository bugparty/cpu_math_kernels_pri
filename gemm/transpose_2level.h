#ifndef TRANSPOSE_2LEVEL_H
#define TRANSPOSE_2LEVEL_H

#include <algorithm>
#include <immintrin.h>

/**
 * Two-level hierarchical tiling matrix transpose.
 *
 * L2-tile (outer): 256x256 -- fits in L2 cache (~1MB for src+dst doubles)
 * L1-tile (inner):  32x32  -- fits in L1 cache (~16KB for src+dst doubles)
 * Micro-kernel:     4x4 block load-then-store to reduce read/write dependency
 */
inline void transpose_2level(const double* __restrict__ A,
                             double* __restrict__ AT,
                             int n) noexcept {
    constexpr int L2_TILE = 256;
    constexpr int L1_TILE = 32;

    for (int l2_ti = 0; l2_ti < n; l2_ti += L2_TILE) {
        const int l2_ti_end = std::min(l2_ti + L2_TILE, n);
        for (int l2_tj = 0; l2_tj < n; l2_tj += L2_TILE) {
            const int l2_tj_end = std::min(l2_tj + L2_TILE, n);

            // L1-level tiling within the L2 tile
            for (int l1_ti = l2_ti; l1_ti < l2_ti_end; l1_ti += L1_TILE) {
                const int l1_ti_end = std::min(l1_ti + L1_TILE, l2_ti_end);
                for (int l1_tj = l2_tj; l1_tj < l2_tj_end; l1_tj += L1_TILE) {
                    const int l1_tj_end = std::min(l1_tj + L1_TILE, l2_tj_end);

                    // 4x4 micro-kernel within the L1 tile
                    int ii = l1_ti;
                    for (; ii + 4 <= l1_ti_end; ii += 4) {
                        int jj = l1_tj;
                        for (; jj + 4 <= l1_tj_end; jj += 4) {
                            // Load 4x4 block from A
                            double t00 = A[(ii    ) * n + jj    ];
                            double t01 = A[(ii    ) * n + jj + 1];
                            double t02 = A[(ii    ) * n + jj + 2];
                            double t03 = A[(ii    ) * n + jj + 3];

                            double t10 = A[(ii + 1) * n + jj    ];
                            double t11 = A[(ii + 1) * n + jj + 1];
                            double t12 = A[(ii + 1) * n + jj + 2];
                            double t13 = A[(ii + 1) * n + jj + 3];

                            double t20 = A[(ii + 2) * n + jj    ];
                            double t21 = A[(ii + 2) * n + jj + 1];
                            double t22 = A[(ii + 2) * n + jj + 2];
                            double t23 = A[(ii + 2) * n + jj + 3];

                            double t30 = A[(ii + 3) * n + jj    ];
                            double t31 = A[(ii + 3) * n + jj + 1];
                            double t32 = A[(ii + 3) * n + jj + 2];
                            double t33 = A[(ii + 3) * n + jj + 3];

                            // Store transposed 4x4 block to AT
                            AT[(jj    ) * n + ii    ] = t00;
                            AT[(jj    ) * n + ii + 1] = t10;
                            AT[(jj    ) * n + ii + 2] = t20;
                            AT[(jj    ) * n + ii + 3] = t30;

                            AT[(jj + 1) * n + ii    ] = t01;
                            AT[(jj + 1) * n + ii + 1] = t11;
                            AT[(jj + 1) * n + ii + 2] = t21;
                            AT[(jj + 1) * n + ii + 3] = t31;

                            AT[(jj + 2) * n + ii    ] = t02;
                            AT[(jj + 2) * n + ii + 1] = t12;
                            AT[(jj + 2) * n + ii + 2] = t22;
                            AT[(jj + 2) * n + ii + 3] = t32;

                            AT[(jj + 3) * n + ii    ] = t03;
                            AT[(jj + 3) * n + ii + 1] = t13;
                            AT[(jj + 3) * n + ii + 2] = t23;
                            AT[(jj + 3) * n + ii + 3] = t33;
                        }
                        // Remainder columns (jj boundary)
                        for (; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii    ] = A[(ii    ) * n + jj];
                            AT[jj * n + ii + 1] = A[(ii + 1) * n + jj];
                            AT[jj * n + ii + 2] = A[(ii + 2) * n + jj];
                            AT[jj * n + ii + 3] = A[(ii + 3) * n + jj];
                        }
                    }
                    // Remainder rows (ii boundary)
                    for (; ii < l1_ti_end; ++ii) {
                        for (int jj = l1_tj; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii] = A[ii * n + jj];
                        }
                    }
                }
            }
        }
    }
}

/**
 * Variant with configurable tile sizes via template parameters.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned(const double* __restrict__ A,
                                   double* __restrict__ AT,
                                   int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 4, "L1 tile must be >= 4 for the 4x4 micro-kernel");

    for (int l2_ti = 0; l2_ti < n; l2_ti += L2_TILE) {
        const int l2_ti_end = std::min(l2_ti + L2_TILE, n);
        for (int l2_tj = 0; l2_tj < n; l2_tj += L2_TILE) {
            const int l2_tj_end = std::min(l2_tj + L2_TILE, n);

            for (int l1_ti = l2_ti; l1_ti < l2_ti_end; l1_ti += L1_TILE) {
                const int l1_ti_end = std::min(l1_ti + L1_TILE, l2_ti_end);
                for (int l1_tj = l2_tj; l1_tj < l2_tj_end; l1_tj += L1_TILE) {
                    const int l1_tj_end = std::min(l1_tj + L1_TILE, l2_tj_end);

                    int ii = l1_ti;
                    for (; ii + 4 <= l1_ti_end; ii += 4) {
                        int jj = l1_tj;
                        for (; jj + 4 <= l1_tj_end; jj += 4) {
                            double t00 = A[(ii    ) * n + jj    ];
                            double t01 = A[(ii    ) * n + jj + 1];
                            double t02 = A[(ii    ) * n + jj + 2];
                            double t03 = A[(ii    ) * n + jj + 3];

                            double t10 = A[(ii + 1) * n + jj    ];
                            double t11 = A[(ii + 1) * n + jj + 1];
                            double t12 = A[(ii + 1) * n + jj + 2];
                            double t13 = A[(ii + 1) * n + jj + 3];

                            double t20 = A[(ii + 2) * n + jj    ];
                            double t21 = A[(ii + 2) * n + jj + 1];
                            double t22 = A[(ii + 2) * n + jj + 2];
                            double t23 = A[(ii + 2) * n + jj + 3];

                            double t30 = A[(ii + 3) * n + jj    ];
                            double t31 = A[(ii + 3) * n + jj + 1];
                            double t32 = A[(ii + 3) * n + jj + 2];
                            double t33 = A[(ii + 3) * n + jj + 3];

                            AT[(jj    ) * n + ii    ] = t00;
                            AT[(jj    ) * n + ii + 1] = t10;
                            AT[(jj    ) * n + ii + 2] = t20;
                            AT[(jj    ) * n + ii + 3] = t30;

                            AT[(jj + 1) * n + ii    ] = t01;
                            AT[(jj + 1) * n + ii + 1] = t11;
                            AT[(jj + 1) * n + ii + 2] = t21;
                            AT[(jj + 1) * n + ii + 3] = t31;

                            AT[(jj + 2) * n + ii    ] = t02;
                            AT[(jj + 2) * n + ii + 1] = t12;
                            AT[(jj + 2) * n + ii + 2] = t22;
                            AT[(jj + 2) * n + ii + 3] = t32;

                            AT[(jj + 3) * n + ii    ] = t03;
                            AT[(jj + 3) * n + ii + 1] = t13;
                            AT[(jj + 3) * n + ii + 2] = t23;
                            AT[(jj + 3) * n + ii + 3] = t33;
                        }
                        for (; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii    ] = A[(ii    ) * n + jj];
                            AT[jj * n + ii + 1] = A[(ii + 1) * n + jj];
                            AT[jj * n + ii + 2] = A[(ii + 2) * n + jj];
                            AT[jj * n + ii + 3] = A[(ii + 3) * n + jj];
                        }
                    }
                    for (; ii < l1_ti_end; ++ii) {
                        for (int jj = l1_tj; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii] = A[ii * n + jj];
                        }
                    }
                }
            }
        }
    }
}

/**
 * AVX2 4x4 micro-kernel variant with configurable tile sizes.
 * Keeps the same 2-level tiling as transpose_2level_tuned, but uses
 * 256-bit registers (__m256d) in the inner 4x4 block when AVX2 is available.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx2(const double* __restrict__ A,
                                        double* __restrict__ AT,
                                        int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 4, "L1 tile must be >= 4 for the 4x4 micro-kernel");

#if defined(__AVX2__)
    for (int l2_ti = 0; l2_ti < n; l2_ti += L2_TILE) {
        const int l2_ti_end = std::min(l2_ti + L2_TILE, n);
        for (int l2_tj = 0; l2_tj < n; l2_tj += L2_TILE) {
            const int l2_tj_end = std::min(l2_tj + L2_TILE, n);

            for (int l1_ti = l2_ti; l1_ti < l2_ti_end; l1_ti += L1_TILE) {
                const int l1_ti_end = std::min(l1_ti + L1_TILE, l2_ti_end);
                for (int l1_tj = l2_tj; l1_tj < l2_tj_end; l1_tj += L1_TILE) {
                    const int l1_tj_end = std::min(l1_tj + L1_TILE, l2_tj_end);

                    int ii = l1_ti;
                    for (; ii + 4 <= l1_ti_end; ii += 4) {
                        int jj = l1_tj;
                        for (; jj + 4 <= l1_tj_end; jj += 4) {
                            const __m256d r0 = _mm256_loadu_pd(&A[(ii    ) * n + jj]);
                            const __m256d r1 = _mm256_loadu_pd(&A[(ii + 1) * n + jj]);
                            const __m256d r2 = _mm256_loadu_pd(&A[(ii + 2) * n + jj]);
                            const __m256d r3 = _mm256_loadu_pd(&A[(ii + 3) * n + jj]);

                            const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
                            const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
                            const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
                            const __m256d t3 = _mm256_unpackhi_pd(r2, r3);

                            const __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                            const __m256d c1 = _mm256_permute2f128_pd(t0, t2, 0x31);
                            const __m256d c2 = _mm256_permute2f128_pd(t1, t3, 0x20);
                            const __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                            _mm256_storeu_pd(&AT[(jj    ) * n + ii], c0);
                            _mm256_storeu_pd(&AT[(jj + 1) * n + ii], c2);
                            _mm256_storeu_pd(&AT[(jj + 2) * n + ii], c1);
                            _mm256_storeu_pd(&AT[(jj + 3) * n + ii], c3);
                        }
                        for (; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii    ] = A[(ii    ) * n + jj];
                            AT[jj * n + ii + 1] = A[(ii + 1) * n + jj];
                            AT[jj * n + ii + 2] = A[(ii + 2) * n + jj];
                            AT[jj * n + ii + 3] = A[(ii + 3) * n + jj];
                        }
                    }
                    for (; ii < l1_ti_end; ++ii) {
                        for (int jj = l1_tj; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii] = A[ii * n + jj];
                        }
                    }
                }
            }
        }
    }
#else
    transpose_2level_tuned<L2_TILE, L1_TILE>(A, AT, n);
#endif
}

#if defined(__AVX512F__)
static inline void transpose_8x8_kernel_2level(const double* __restrict__ src,
                                               double* __restrict__ dst,
                                               int src_stride,
                                               int dst_stride) noexcept {
    const __m512d r0 = _mm512_loadu_pd(src + 0 * src_stride);
    const __m512d r1 = _mm512_loadu_pd(src + 1 * src_stride);
    const __m512d r2 = _mm512_loadu_pd(src + 2 * src_stride);
    const __m512d r3 = _mm512_loadu_pd(src + 3 * src_stride);
    const __m512d r4 = _mm512_loadu_pd(src + 4 * src_stride);
    const __m512d r5 = _mm512_loadu_pd(src + 5 * src_stride);
    const __m512d r6 = _mm512_loadu_pd(src + 6 * src_stride);
    const __m512d r7 = _mm512_loadu_pd(src + 7 * src_stride);

    const __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    const __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    const __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    const __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    const __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    const __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    const __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    const __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    const __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    const __m512d u1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    const __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    const __m512d u3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    const __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    const __m512d u5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    const __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    const __m512d u7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    const __m512d c0 = _mm512_shuffle_f64x2(u0, u4, 0x88);
    const __m512d c1 = _mm512_shuffle_f64x2(u1, u5, 0x88);
    const __m512d c2 = _mm512_shuffle_f64x2(u2, u6, 0x88);
    const __m512d c3 = _mm512_shuffle_f64x2(u3, u7, 0x88);
    const __m512d c4 = _mm512_shuffle_f64x2(u0, u4, 0xDD);
    const __m512d c5 = _mm512_shuffle_f64x2(u1, u5, 0xDD);
    const __m512d c6 = _mm512_shuffle_f64x2(u2, u6, 0xDD);
    const __m512d c7 = _mm512_shuffle_f64x2(u3, u7, 0xDD);

    _mm512_storeu_pd(dst + 0 * dst_stride, c0);
    _mm512_storeu_pd(dst + 1 * dst_stride, c1);
    _mm512_storeu_pd(dst + 2 * dst_stride, c2);
    _mm512_storeu_pd(dst + 3 * dst_stride, c3);
    _mm512_storeu_pd(dst + 4 * dst_stride, c4);
    _mm512_storeu_pd(dst + 5 * dst_stride, c5);
    _mm512_storeu_pd(dst + 6 * dst_stride, c6);
    _mm512_storeu_pd(dst + 7 * dst_stride, c7);
}
#endif

/**
 * AVX512 8x8 micro-kernel variant with configurable tile sizes.
 * This is a new variant and does not modify transpose_2level_tuned.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx512(const double* __restrict__ A,
                                          double* __restrict__ AT,
                                          int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 8, "L1 tile must be >= 8 for the 8x8 AVX512 micro-kernel");

#if defined(__AVX512F__)
    for (int l2_ti = 0; l2_ti < n; l2_ti += L2_TILE) {
        const int l2_ti_end = std::min(l2_ti + L2_TILE, n);
        for (int l2_tj = 0; l2_tj < n; l2_tj += L2_TILE) {
            const int l2_tj_end = std::min(l2_tj + L2_TILE, n);

            for (int l1_ti = l2_ti; l1_ti < l2_ti_end; l1_ti += L1_TILE) {
                const int l1_ti_end = std::min(l1_ti + L1_TILE, l2_ti_end);
                for (int l1_tj = l2_tj; l1_tj < l2_tj_end; l1_tj += L1_TILE) {
                    const int l1_tj_end = std::min(l1_tj + L1_TILE, l2_tj_end);

                    int ii = l1_ti;
                    for (; ii + 8 <= l1_ti_end; ii += 8) {
                        int jj = l1_tj;
                        for (; jj + 8 <= l1_tj_end; jj += 8) {
                            transpose_8x8_kernel_2level(A + ii * n + jj, AT + jj * n + ii, n, n);
                        }

                        for (; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii    ] = A[(ii    ) * n + jj];
                            AT[jj * n + ii + 1] = A[(ii + 1) * n + jj];
                            AT[jj * n + ii + 2] = A[(ii + 2) * n + jj];
                            AT[jj * n + ii + 3] = A[(ii + 3) * n + jj];
                            AT[jj * n + ii + 4] = A[(ii + 4) * n + jj];
                            AT[jj * n + ii + 5] = A[(ii + 5) * n + jj];
                            AT[jj * n + ii + 6] = A[(ii + 6) * n + jj];
                            AT[jj * n + ii + 7] = A[(ii + 7) * n + jj];
                        }
                    }

                    for (; ii < l1_ti_end; ++ii) {
                        for (int jj = l1_tj; jj < l1_tj_end; ++jj) {
                            AT[jj * n + ii] = A[ii * n + jj];
                        }
                    }
                }
            }
        }
    }
#else
    transpose_2level_tuned<L2_TILE, L1_TILE>(A, AT, n);
#endif
}

#endif // TRANSPOSE_2LEVEL_H
