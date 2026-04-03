#ifndef TRANSPOSE_2LEVEL_H
#define TRANSPOSE_2LEVEL_H

#include <algorithm>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include "../include/compiler_compat.h"
#include "transpose_avx512.h"

// Forward declarations of micro-kernels to avoid "not declared in this scope" errors in template functions
static inline void transpose8x8_f64_nt_avx2_pf(const double *src, std::ptrdiff_t src_ld, double *dst, std::ptrdiff_t dst_ld);
static inline void transpose8x8_f64_nt_avx2_pf_nofence(const double *src, std::ptrdiff_t src_ld, double *dst, std::ptrdiff_t dst_ld);
static inline void transpose8x8_f64_pf_avx2(const double *src, std::ptrdiff_t src_ld, double *dst, std::ptrdiff_t dst_ld);
static inline void transpose8x8_f64_nt_avx2_pf_v2(const double *src, double *dst, ptrdiff_t src_stride, ptrdiff_t dst_stride);
static inline void transpose8x8_f64_nt_avx2_pf_v5(const double *src, std::ptrdiff_t src_ld, double *dst, std::ptrdiff_t dst_ld);
static inline void transpose8x8_f64_nt_avx2_pf_v3(const double *src, std::ptrdiff_t src_ld, double *dst, std::ptrdiff_t dst_ld);

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
 * Hint-heavy scalar variant of the 2-level tiled transpose.
 *
 * This version keeps the same algorithmic structure as
 * transpose_2level_tuned, but adds extra compiler-friendly structure:
 * - aggressively const-qualified loop invariants and row pointers
 * - restrict-qualified source/destination bases
 * - predictable fast-path 4x4 micro-kernel guarded with __builtin_expect
 * - light software prefetching for upcoming source rows and destination columns
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_hinted(const double* __restrict__ A,
                                          double* __restrict__ AT,
                                          int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 4, "L1 tile must be >= 4 for the 4x4 micro-kernel");

    const int n_local = n;
    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(n_local);
    const double* const __restrict__ A_base = A;
    double* const __restrict__ AT_base = AT;

    for (int l2_ti = 0; l2_ti < n_local; l2_ti += L2_TILE) {
        const int l2_ti_end = std::min(l2_ti + L2_TILE, n_local);
        for (int l2_tj = 0; l2_tj < n_local; l2_tj += L2_TILE) {
            const int l2_tj_end = std::min(l2_tj + L2_TILE, n_local);

            for (int l1_ti = l2_ti; l1_ti < l2_ti_end; l1_ti += L1_TILE) {
                const int l1_ti_end = std::min(l1_ti + L1_TILE, l2_ti_end);
                for (int l1_tj = l2_tj; l1_tj < l2_tj_end; l1_tj += L1_TILE) {
                    const int l1_tj_end = std::min(l1_tj + L1_TILE, l2_tj_end);
                    const int has_full_rows = l1_ti_end - l1_ti;
                    const int has_full_cols = l1_tj_end - l1_tj;

                    int ii = l1_ti;
                    if (__builtin_expect(has_full_rows >= 4 && has_full_cols >= 4, 1)) {
                        for (; ii + 4 <= l1_ti_end; ii += 4) {
                            const std::ptrdiff_t row0_idx = static_cast<std::ptrdiff_t>(ii) * stride;
                            const std::ptrdiff_t row1_idx = static_cast<std::ptrdiff_t>(ii + 1) * stride;
                            const std::ptrdiff_t row2_idx = static_cast<std::ptrdiff_t>(ii + 2) * stride;
                            const std::ptrdiff_t row3_idx = static_cast<std::ptrdiff_t>(ii + 3) * stride;
                            const double* const __restrict__ row0 = A_base + row0_idx;
                            const double* const __restrict__ row1 = A_base + row1_idx;
                            const double* const __restrict__ row2 = A_base + row2_idx;
                            const double* const __restrict__ row3 = A_base + row3_idx;

                            int jj = l1_tj;
                            for (; jj + 4 <= l1_tj_end; jj += 4) {
                                const int pf_j = jj + 16;
                                if (pf_j < l1_tj_end) {
                                    __builtin_prefetch(row0 + pf_j, 0, 1);
                                    __builtin_prefetch(row1 + pf_j, 0, 1);
                                    __builtin_prefetch(row2 + pf_j, 0, 1);
                                    __builtin_prefetch(row3 + pf_j, 0, 1);
                                    __builtin_prefetch(AT_base + static_cast<std::ptrdiff_t>(pf_j) * stride + ii, 1, 1);
                                }

                                const double t00 = row0[jj    ];
                                const double t01 = row0[jj + 1];
                                const double t02 = row0[jj + 2];
                                const double t03 = row0[jj + 3];

                                const double t10 = row1[jj    ];
                                const double t11 = row1[jj + 1];
                                const double t12 = row1[jj + 2];
                                const double t13 = row1[jj + 3];

                                const double t20 = row2[jj    ];
                                const double t21 = row2[jj + 1];
                                const double t22 = row2[jj + 2];
                                const double t23 = row2[jj + 3];

                                const double t30 = row3[jj    ];
                                const double t31 = row3[jj + 1];
                                const double t32 = row3[jj + 2];
                                const double t33 = row3[jj + 3];

                                double* const __restrict__ dst0 = AT_base + static_cast<std::ptrdiff_t>(jj    ) * stride + ii;
                                double* const __restrict__ dst1 = AT_base + static_cast<std::ptrdiff_t>(jj + 1) * stride + ii;
                                double* const __restrict__ dst2 = AT_base + static_cast<std::ptrdiff_t>(jj + 2) * stride + ii;
                                double* const __restrict__ dst3 = AT_base + static_cast<std::ptrdiff_t>(jj + 3) * stride + ii;

                                dst0[0] = t00;
                                dst0[1] = t10;
                                dst0[2] = t20;
                                dst0[3] = t30;

                                dst1[0] = t01;
                                dst1[1] = t11;
                                dst1[2] = t21;
                                dst1[3] = t31;

                                dst2[0] = t02;
                                dst2[1] = t12;
                                dst2[2] = t22;
                                dst2[3] = t32;

                                dst3[0] = t03;
                                dst3[1] = t13;
                                dst3[2] = t23;
                                dst3[3] = t33;
                            }

                            for (; jj < l1_tj_end; ++jj) {
                                double* const __restrict__ dst = AT_base + static_cast<std::ptrdiff_t>(jj) * stride + ii;
                                dst[0] = row0[jj];
                                dst[1] = row1[jj];
                                dst[2] = row2[jj];
                                dst[3] = row3[jj];
                            }
                        }
                    }

                    for (; ii < l1_ti_end; ++ii) {
                        const double* const __restrict__ row = A_base + static_cast<std::ptrdiff_t>(ii) * stride;
                        for (int jj = l1_tj; jj < l1_tj_end; ++jj) {
                            AT_base[static_cast<std::ptrdiff_t>(jj) * stride + ii] = row[jj];
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

/**
 * AVX2 8x8 non-temporal + prefetch micro-kernel variant.
 * Uses transpose8x8_f64_nt_avx2_pf as the inner kernel when AVX2 is available.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx2_nt_pf(const double* __restrict__ A,
                                              double* __restrict__ AT,
                                              int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 8, "L1 tile must be >= 8 for the 8x8 micro-kernel");

#if defined(__AVX2__)
    // Simpler condition: only check if n is divisible by 8 for the kernel
    const bool use_stream_kernel = ((n & 7) == 0);

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
                            if (use_stream_kernel) {
                                transpose8x8_f64_nt_avx2_pf(
                                    A + ii * n + jj,
                                    static_cast<std::ptrdiff_t>(n),
                                    AT + jj * n + ii,
                                    static_cast<std::ptrdiff_t>(n));
                            } else {
                                // Use AVX2 4x4 kernel as fallback instead of scalar
                                const __m256d r0 = _mm256_loadu_pd(&A[(ii    ) * n + jj]);
                                const __m256d r1 = _mm256_loadu_pd(&A[(ii + 1) * n + jj]);
                                const __m256d r2 = _mm256_loadu_pd(&A[(ii + 2) * n + jj]);
                                const __m256d r3 = _mm256_loadu_pd(&A[(ii + 3) * n + jj]);
                                const __m256d r4 = _mm256_loadu_pd(&A[(ii + 4) * n + jj]);
                                const __m256d r5 = _mm256_loadu_pd(&A[(ii + 5) * n + jj]);
                                const __m256d r6 = _mm256_loadu_pd(&A[(ii + 6) * n + jj]);
                                const __m256d r7 = _mm256_loadu_pd(&A[(ii + 7) * n + jj]);

                                const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
                                const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
                                const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
                                const __m256d t3 = _mm256_unpackhi_pd(r2, r3);
                                const __m256d t4 = _mm256_unpacklo_pd(r4, r5);
                                const __m256d t5 = _mm256_unpackhi_pd(r4, r5);
                                const __m256d t6 = _mm256_unpacklo_pd(r6, r7);
                                const __m256d t7 = _mm256_unpackhi_pd(r6, r7);

                                const __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                                const __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                                const __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                                const __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                                _mm256_storeu_pd(&AT[(jj    ) * n + ii], c0);
                                _mm256_storeu_pd(&AT[(jj + 1) * n + ii], c1);
                                _mm256_storeu_pd(&AT[(jj + 2) * n + ii], c2);
                                _mm256_storeu_pd(&AT[(jj + 3) * n + ii], c3);
                            }
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

/**
 * A/B kernel A:
 * same as [`transpose_2level_tuned_avx2_nt_pf()`](gemm/transpose_2level.h:262)
 * but removes the per-8x8-tile [`_mm_sfence()`](gemm/transpose_2level.h:478)
 * and performs a single fence at function end.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx2_nt_pf_nofence(const double* __restrict__ A,
                                                      double* __restrict__ AT,
                                                      int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 8, "L1 tile must be >= 8 for the 8x8 micro-kernel");

#if defined(__AVX2__)
    const bool use_stream_kernel = ((n & 7) == 0);

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
                            if (use_stream_kernel) {
                                transpose8x8_f64_nt_avx2_pf_nofence(
                                    A + ii * n + jj,
                                    static_cast<std::ptrdiff_t>(n),
                                    AT + jj * n + ii,
                                    static_cast<std::ptrdiff_t>(n));
                            } else {
                                for (int i = 0; i < 8; ++i) {
                                    for (int j = 0; j < 8; ++j) {
                                        AT[(jj + j) * n + (ii + i)] = A[(ii + i) * n + (jj + j)];
                                    }
                                }
                            }
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
    if (use_stream_kernel) {
        _mm_sfence();
    }
#else
    transpose_2level_tuned<L2_TILE, L1_TILE>(A, AT, n);
#endif
}

/**
 * A/B kernel B:
 * same 8x8 structure as [`transpose_2level_tuned_avx2_nt_pf()`](gemm/transpose_2level.h:262)
 * but uses cached stores instead of non-temporal stores.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx2_pf_store(const double* __restrict__ A,
                                                 double* __restrict__ AT,
                                                 int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 8, "L1 tile must be >= 8 for the 8x8 micro-kernel");

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
                    for (; ii + 8 <= l1_ti_end; ii += 8) {
                        int jj = l1_tj;
                        for (; jj + 8 <= l1_tj_end; jj += 8) {
                            transpose8x8_f64_pf_avx2(
                                A + ii * n + jj,
                                static_cast<std::ptrdiff_t>(n),
                                AT + jj * n + ii,
                                static_cast<std::ptrdiff_t>(n));
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



// Transpose a single 8×8 tile of double-precision values using AVX2.
//
// Parameters:
//   src     - Pointer to the top-left element of the source tile.
//             The source matrix is assumed to be stored in row-major order.
//   src_ld  - Leading dimension (stride) of the source matrix, in elements.
//             This is the distance between consecutive rows in memory
//             (i.e., number of elements between src[i][0] and src[i+1][0]).
//   dst     - Pointer to the top-left element of the destination tile.
//             The destination matrix is also in row-major order.
//   dst_ld  - Leading dimension (stride) of the destination matrix, in elements.
//             This is the distance between consecutive rows in the destination
//             (i.e., number of elements between dst[i][0] and dst[i+1][0]).
//
// Effect:
//   Writes the transpose of the 8×8 block starting at src into dst:
//       dst[j * dst_ld + i] = src[i * src_ld + j], for i,j in [0,7].
static inline void transpose8x8_f64_nt_avx2_pf(
    const double* src,
    std::ptrdiff_t src_ld,
    double* dst,
    std::ptrdiff_t dst_ld)
{
    transpose8x8_f64_nt_avx2_pf_nofence(src, src_ld, dst, dst_ld);
}

static inline void transpose8x8_f64_nt_avx2_pf_nofence(
    const double* src,
    std::ptrdiff_t src_ld,
    double* dst,
    std::ptrdiff_t dst_ld)
{
    _mm_prefetch(reinterpret_cast<const char*>(src + 0 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 1 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 2 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 3 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 4 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 5 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 6 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 7 * src_ld), _MM_HINT_NTA);

    __m256d r0_lo = _mm256_loadu_pd(src + 0 * src_ld + 0);
    __m256d r0_hi = _mm256_loadu_pd(src + 0 * src_ld + 4);
    __m256d r1_lo = _mm256_loadu_pd(src + 1 * src_ld + 0);
    __m256d r1_hi = _mm256_loadu_pd(src + 1 * src_ld + 4);
    __m256d r2_lo = _mm256_loadu_pd(src + 2 * src_ld + 0);
    __m256d r2_hi = _mm256_loadu_pd(src + 2 * src_ld + 4);
    __m256d r3_lo = _mm256_loadu_pd(src + 3 * src_ld + 0);
    __m256d r3_hi = _mm256_loadu_pd(src + 3 * src_ld + 4);
    __m256d r4_lo = _mm256_loadu_pd(src + 4 * src_ld + 0);
    __m256d r4_hi = _mm256_loadu_pd(src + 4 * src_ld + 4);
    __m256d r5_lo = _mm256_loadu_pd(src + 5 * src_ld + 0);
    __m256d r5_hi = _mm256_loadu_pd(src + 5 * src_ld + 4);
    __m256d r6_lo = _mm256_loadu_pd(src + 6 * src_ld + 0);
    __m256d r6_hi = _mm256_loadu_pd(src + 6 * src_ld + 4);
    __m256d r7_lo = _mm256_loadu_pd(src + 7 * src_ld + 0);
    __m256d r7_hi = _mm256_loadu_pd(src + 7 * src_ld + 4);

    __m256d t0 = _mm256_unpacklo_pd(r0_lo, r1_lo);
    __m256d t1 = _mm256_unpackhi_pd(r0_lo, r1_lo);
    __m256d t2 = _mm256_unpacklo_pd(r2_lo, r3_lo);
    __m256d t3 = _mm256_unpackhi_pd(r2_lo, r3_lo);
    __m256d t4 = _mm256_unpacklo_pd(r4_lo, r5_lo);
    __m256d t5 = _mm256_unpackhi_pd(r4_lo, r5_lo);
    __m256d t6 = _mm256_unpacklo_pd(r6_lo, r7_lo);
    __m256d t7 = _mm256_unpackhi_pd(r6_lo, r7_lo);

    __m256d u0 = _mm256_unpacklo_pd(r0_hi, r1_hi);
    __m256d u1 = _mm256_unpackhi_pd(r0_hi, r1_hi);
    __m256d u2 = _mm256_unpacklo_pd(r2_hi, r3_hi);
    __m256d u3 = _mm256_unpackhi_pd(r2_hi, r3_hi);
    __m256d u4 = _mm256_unpacklo_pd(r4_hi, r5_hi);
    __m256d u5 = _mm256_unpackhi_pd(r4_hi, r5_hi);
    __m256d u6 = _mm256_unpacklo_pd(r6_hi, r7_hi);
    __m256d u7 = _mm256_unpackhi_pd(r6_hi, r7_hi);

    __m256d d0_lo = _mm256_permute2f128_pd(t0, t2, 0x20);
    __m256d d1_lo = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d d2_lo = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d d3_lo = _mm256_permute2f128_pd(t1, t3, 0x31);
    __m256d d0_hi = _mm256_permute2f128_pd(t4, t6, 0x20);
    __m256d d1_hi = _mm256_permute2f128_pd(t5, t7, 0x20);
    __m256d d2_hi = _mm256_permute2f128_pd(t4, t6, 0x31);
    __m256d d3_hi = _mm256_permute2f128_pd(t5, t7, 0x31);
    __m256d d4_lo = _mm256_permute2f128_pd(u0, u2, 0x20);
    __m256d d5_lo = _mm256_permute2f128_pd(u1, u3, 0x20);
    __m256d d6_lo = _mm256_permute2f128_pd(u0, u2, 0x31);
    __m256d d7_lo = _mm256_permute2f128_pd(u1, u3, 0x31);
    __m256d d4_hi = _mm256_permute2f128_pd(u4, u6, 0x20);
    __m256d d5_hi = _mm256_permute2f128_pd(u5, u7, 0x20);
    __m256d d6_hi = _mm256_permute2f128_pd(u4, u6, 0x31);
    __m256d d7_hi = _mm256_permute2f128_pd(u5, u7, 0x31);

    _mm256_stream_pd(dst + 0 * dst_ld + 0, d0_lo);
    _mm256_stream_pd(dst + 0 * dst_ld + 4, d0_hi);
    _mm256_stream_pd(dst + 1 * dst_ld + 0, d1_lo);
    _mm256_stream_pd(dst + 1 * dst_ld + 4, d1_hi);
    _mm256_stream_pd(dst + 2 * dst_ld + 0, d2_lo);
    _mm256_stream_pd(dst + 2 * dst_ld + 4, d2_hi);
    _mm256_stream_pd(dst + 3 * dst_ld + 0, d3_lo);
    _mm256_stream_pd(dst + 3 * dst_ld + 4, d3_hi);
    _mm256_stream_pd(dst + 4 * dst_ld + 0, d4_lo);
    _mm256_stream_pd(dst + 4 * dst_ld + 4, d4_hi);
    _mm256_stream_pd(dst + 5 * dst_ld + 0, d5_lo);
    _mm256_stream_pd(dst + 5 * dst_ld + 4, d5_hi);
    _mm256_stream_pd(dst + 6 * dst_ld + 0, d6_lo);
    _mm256_stream_pd(dst + 6 * dst_ld + 4, d6_hi);
    _mm256_stream_pd(dst + 7 * dst_ld + 0, d7_lo);
    _mm256_stream_pd(dst + 7 * dst_ld + 4, d7_hi);
}

static inline void transpose8x8_f64_pf_avx2(
    const double* src,
    std::ptrdiff_t src_ld,
    double* dst,
    std::ptrdiff_t dst_ld)
{
    _mm_prefetch(reinterpret_cast<const char*>(src + 0 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 1 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 2 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 3 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 4 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 5 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 6 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 7 * src_ld), _MM_HINT_NTA);

    __m256d r0_lo = _mm256_loadu_pd(src + 0 * src_ld + 0);
    __m256d r0_hi = _mm256_loadu_pd(src + 0 * src_ld + 4);
    __m256d r1_lo = _mm256_loadu_pd(src + 1 * src_ld + 0);
    __m256d r1_hi = _mm256_loadu_pd(src + 1 * src_ld + 4);
    __m256d r2_lo = _mm256_loadu_pd(src + 2 * src_ld + 0);
    __m256d r2_hi = _mm256_loadu_pd(src + 2 * src_ld + 4);
    __m256d r3_lo = _mm256_loadu_pd(src + 3 * src_ld + 0);
    __m256d r3_hi = _mm256_loadu_pd(src + 3 * src_ld + 4);
    __m256d r4_lo = _mm256_loadu_pd(src + 4 * src_ld + 0);
    __m256d r4_hi = _mm256_loadu_pd(src + 4 * src_ld + 4);
    __m256d r5_lo = _mm256_loadu_pd(src + 5 * src_ld + 0);
    __m256d r5_hi = _mm256_loadu_pd(src + 5 * src_ld + 4);
    __m256d r6_lo = _mm256_loadu_pd(src + 6 * src_ld + 0);
    __m256d r6_hi = _mm256_loadu_pd(src + 6 * src_ld + 4);
    __m256d r7_lo = _mm256_loadu_pd(src + 7 * src_ld + 0);
    __m256d r7_hi = _mm256_loadu_pd(src + 7 * src_ld + 4);

    __m256d t0 = _mm256_unpacklo_pd(r0_lo, r1_lo);
    __m256d t1 = _mm256_unpackhi_pd(r0_lo, r1_lo);
    __m256d t2 = _mm256_unpacklo_pd(r2_lo, r3_lo);
    __m256d t3 = _mm256_unpackhi_pd(r2_lo, r3_lo);
    __m256d t4 = _mm256_unpacklo_pd(r4_lo, r5_lo);
    __m256d t5 = _mm256_unpackhi_pd(r4_lo, r5_lo);
    __m256d t6 = _mm256_unpacklo_pd(r6_lo, r7_lo);
    __m256d t7 = _mm256_unpackhi_pd(r6_lo, r7_lo);

    __m256d u0 = _mm256_unpacklo_pd(r0_hi, r1_hi);
    __m256d u1 = _mm256_unpackhi_pd(r0_hi, r1_hi);
    __m256d u2 = _mm256_unpacklo_pd(r2_hi, r3_hi);
    __m256d u3 = _mm256_unpackhi_pd(r2_hi, r3_hi);
    __m256d u4 = _mm256_unpacklo_pd(r4_hi, r5_hi);
    __m256d u5 = _mm256_unpackhi_pd(r4_hi, r5_hi);
    __m256d u6 = _mm256_unpacklo_pd(r6_hi, r7_hi);
    __m256d u7 = _mm256_unpackhi_pd(r6_hi, r7_hi);

    __m256d d0_lo = _mm256_permute2f128_pd(t0, t2, 0x20);
    __m256d d1_lo = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d d2_lo = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d d3_lo = _mm256_permute2f128_pd(t1, t3, 0x31);
    __m256d d0_hi = _mm256_permute2f128_pd(t4, t6, 0x20);
    __m256d d1_hi = _mm256_permute2f128_pd(t5, t7, 0x20);
    __m256d d2_hi = _mm256_permute2f128_pd(t4, t6, 0x31);
    __m256d d3_hi = _mm256_permute2f128_pd(t5, t7, 0x31);
    __m256d d4_lo = _mm256_permute2f128_pd(u0, u2, 0x20);
    __m256d d5_lo = _mm256_permute2f128_pd(u1, u3, 0x20);
    __m256d d6_lo = _mm256_permute2f128_pd(u0, u2, 0x31);
    __m256d d7_lo = _mm256_permute2f128_pd(u1, u3, 0x31);
    __m256d d4_hi = _mm256_permute2f128_pd(u4, u6, 0x20);
    __m256d d5_hi = _mm256_permute2f128_pd(u5, u7, 0x20);
    __m256d d6_hi = _mm256_permute2f128_pd(u4, u6, 0x31);
    __m256d d7_hi = _mm256_permute2f128_pd(u5, u7, 0x31);

    _mm256_storeu_pd(dst + 0 * dst_ld + 0, d0_lo);
    _mm256_storeu_pd(dst + 0 * dst_ld + 4, d0_hi);
    _mm256_storeu_pd(dst + 1 * dst_ld + 0, d1_lo);
    _mm256_storeu_pd(dst + 1 * dst_ld + 4, d1_hi);
    _mm256_storeu_pd(dst + 2 * dst_ld + 0, d2_lo);
    _mm256_storeu_pd(dst + 2 * dst_ld + 4, d2_hi);
    _mm256_storeu_pd(dst + 3 * dst_ld + 0, d3_lo);
    _mm256_storeu_pd(dst + 3 * dst_ld + 4, d3_hi);
    _mm256_storeu_pd(dst + 4 * dst_ld + 0, d4_lo);
    _mm256_storeu_pd(dst + 4 * dst_ld + 4, d4_hi);
    _mm256_storeu_pd(dst + 5 * dst_ld + 0, d5_lo);
    _mm256_storeu_pd(dst + 5 * dst_ld + 4, d5_hi);
    _mm256_storeu_pd(dst + 6 * dst_ld + 0, d6_lo);
    _mm256_storeu_pd(dst + 6 * dst_ld + 4, d6_hi);
    _mm256_storeu_pd(dst + 7 * dst_ld + 0, d7_lo);
    _mm256_storeu_pd(dst + 7 * dst_ld + 4, d7_hi);
}
/**
 * transpose_8x8_double_avx2
 *
 * Perform in-place transpose on an 8x8 double matrix stored in row-major format,
 * input and output are both row-major, with row stride in bytes.
 *
 * Algorithm overview (corresponds to disassembly Block 22):
 *
 *   Step 0 — prefetchnta
 *     Prefetch source rows for upcoming iterations into L1/NTA buffer, bypassing LLC,
 *     avoiding LLC pollution (matrix data is used one-shot).
 *
 *   Step 1 — Load 8 rows, each with two ymm registers (16 ymm total)
 *     Each row has 8 doubles = 64 bytes = 2 ymm registers.
 *     Load lower half (cols 0-3) and upper half (cols 4-7) separately.
 *
 *   Step 2 — Stage-A: vunpcklpd / vunpckhpd (interleave within 64-bit lanes)
 *     vunpcklpd ymm_lo, ymm_a, ymm_b
 *       Within each 128-bit lane, take a[lo64] and b[lo64]
 *     vunpckhpd ymm_hi, ymm_a, ymm_b
 *       Within each 128-bit lane, take a[hi64] and b[hi64]
 *     Result: corresponding columns of adjacent rows are paired, yet still separated by lane boundaries.
 *
 *   Step 3 — Stage-B: vinsertf128 / vperm2f128 (cross-lane rearrangement)
 *     vinsertf128 ymm_out, ymm_lo, xmm_hi, 0x1
 *       Insert lower 128 bits of ymm_hi into high 128 bits of ymm_lo
 *       → yields the low 4 columns of one row in transposed output
 *     vperm2f128 ymm_out, ymm_lo, ymm_hi, 0x31
 *       Combine high lane of ymm_lo and high lane of ymm_hi
 *       → yields the high 4 columns of the same transposed row
 *     The same operations for int variants use vinserti128 / vperm2i128.
 *
 *   Step 4 — vmovntps / vmovntdq (non-temporal store)
 *     Bypass cache and write directly to DRAM, suitable for large outputs that are not read again soon,
 *     avoids RFO (Read-For-Ownership) overhead and cache pollution.
 *     Note: call _mm_sfence() afterward to ensure visibility.
 *
 * @param src         source matrix base pointer (32-byte aligned)
 * @param dst         destination matrix base pointer (32-byte aligned)
 * @param src_stride  source matrix row stride (bytes), typically 8 * sizeof(double) = 64
 * @param dst_stride  destination matrix row stride (bytes)
 */
inline void transpose8x8_f64_nt_avx2_pf_v2(
    const double* __restrict__ src,
    double* __restrict__ dst,
    ptrdiff_t src_stride,   // bytes
    ptrdiff_t dst_stride    // bytes
) {
    // ---------------------------------------------------------------
    // Step 0: prefetchnta — prefetch the next batch, NTA = Non-Temporal All
    //         Inform CPU that this data is one-shot, place it in near cache levels without polluting LLC
    // ---------------------------------------------------------------
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 0 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 1 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 2 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 3 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 4 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 5 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 6 + 8, _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src) + src_stride * 7 + 8, _MM_HINT_NTA);

    // ---------------------------------------------------------------
    // Helper lambda: get pointer to row by byte offset
    // ---------------------------------------------------------------
    auto row = [&](const double* base, int r) -> const double* {
        return reinterpret_cast<const double*>(
            reinterpret_cast<const char*>(base) + r * src_stride);
        };
    auto dst_row = [&](double* base, int r) -> double* {
        return reinterpret_cast<double*>(
            reinterpret_cast<char*>(base) + r * dst_stride);
        };

    // ---------------------------------------------------------------
    // Step 1: load 8 rows, split into low half (col 0-3) and high half (col 4-7)
    //
    //   row_lo[i] = [src[i][0], src[i][1], src[i][2], src[i][3]]
    //   row_hi[i] = [src[i][4], src[i][5], src[i][6], src[i][7]]
    // ---------------------------------------------------------------

    // Low half (corresponds to assembly vmovups ymm0/ymm8/... series)
    __m256d r0l = _mm256_loadu_pd(row(src, 0));
    __m256d r1l = _mm256_loadu_pd(row(src, 1));
    __m256d r2l = _mm256_loadu_pd(row(src, 2));
    __m256d r3l = _mm256_loadu_pd(row(src, 3));
    __m256d r4l = _mm256_loadu_pd(row(src, 4));
    __m256d r5l = _mm256_loadu_pd(row(src, 5));
    __m256d r6l = _mm256_loadu_pd(row(src, 6));
    __m256d r7l = _mm256_loadu_pd(row(src, 7));

    // High half (vmovdqu series, +0x20 offset = +32 bytes = 4 doubles)
    __m256d r0h = _mm256_loadu_pd(row(src, 0) + 4);
    __m256d r1h = _mm256_loadu_pd(row(src, 1) + 4);
    __m256d r2h = _mm256_loadu_pd(row(src, 2) + 4);
    __m256d r3h = _mm256_loadu_pd(row(src, 3) + 4);
    __m256d r4h = _mm256_loadu_pd(row(src, 4) + 4);
    __m256d r5h = _mm256_loadu_pd(row(src, 5) + 4);
    __m256d r6h = _mm256_loadu_pd(row(src, 6) + 4);
    __m256d r7h = _mm256_loadu_pd(row(src, 7) + 4);

    // ---------------------------------------------------------------
    // Step 2: Stage-A — vunpcklpd / vunpckhpd
    //
    // Example for the low half (corresponds to assembly vunpcklpd ymm14,ymm0,ymm1 etc.):
    //
    //   unpacklo(r0l, r1l):
    //     lane0: [r0[0], r1[0]]   lane1: [r0[2], r1[2]]
    //   unpackhi(r0l, r1l):
    //     lane0: [r0[1], r1[1]]   lane1: [r0[3], r1[3]]
    //
    // Same operation for (r2l,r3l), (r4l,r5l), (r6l,r7l)
    // ---------------------------------------------------------------

    // --- low half ---
    __m256d t01lo = _mm256_unpacklo_pd(r0l, r1l); // [r0c0,r1c0 | r0c2,r1c2]
    __m256d t01hi = _mm256_unpackhi_pd(r0l, r1l); // [r0c1,r1c1 | r0c3,r1c3]
    __m256d t23lo = _mm256_unpacklo_pd(r2l, r3l); // [r2c0,r3c0 | r2c2,r3c2]
    __m256d t23hi = _mm256_unpackhi_pd(r2l, r3l);
    __m256d t45lo = _mm256_unpacklo_pd(r4l, r5l);
    __m256d t45hi = _mm256_unpackhi_pd(r4l, r5l);
    __m256d t67lo = _mm256_unpacklo_pd(r6l, r7l);
    __m256d t67hi = _mm256_unpackhi_pd(r6l, r7l);

    // --- high half (assembly uses vpunpcklqdq/vpunpckhqdq, semantics equal to vunpck*pd) ---
    __m256d t01lo_h = _mm256_unpacklo_pd(r0h, r1h);
    __m256d t01hi_h = _mm256_unpackhi_pd(r0h, r1h);
    __m256d t23lo_h = _mm256_unpacklo_pd(r2h, r3h);
    __m256d t23hi_h = _mm256_unpackhi_pd(r2h, r3h);
    __m256d t45lo_h = _mm256_unpacklo_pd(r4h, r5h);
    __m256d t45hi_h = _mm256_unpackhi_pd(r4h, r5h);
    __m256d t67lo_h = _mm256_unpacklo_pd(r6h, r7h);
    __m256d t67hi_h = _mm256_unpackhi_pd(r6h, r7h);

    // ---------------------------------------------------------------
    // Step 3: Stage-B — vinsertf128(0x1) + vperm2f128(0x31) cross-lane rearrangement
    //
    //   vinsertf128 dst, src_ymm, src_xmm, 0x1
    //     → dst.lo128 = src_ymm.lo128
    //     → dst.hi128 = src_xmm.lo128         (take lower 128 bits of second operand)
    //
    //   vperm2f128 dst, a, b, 0x31
    //     → dst.lo128 = a.hi128
    //     → dst.hi128 = b.hi128
    //
    // Combined result: col 0 complete = [r0c0, r1c0, r2c0, r3c0]
    // ---------------------------------------------------------------

    // Transposed low-half 4 columns row
    // corresponding assembly: vinsertf128 ymm9,ymm14,xmm1,0x1  / vperm2f128 ymm1,ymm14,ymm1, 0x31
    __m256d col0 = _mm256_insertf128_pd(t01lo,
        _mm256_castpd256_pd128(t23lo), 1); // [r0c0,r1c0,r2c0,r3c0]
    __m256d col1 = _mm256_insertf128_pd(t01hi,
        _mm256_castpd256_pd128(t23hi), 1); // [r0c1,r1c1,r2c1,r3c1]
    __m256d col2 = _mm256_permute2f128_pd(t01lo, t23lo, 0x31); // [r0c2,r1c2,r2c2,r3c2]
    __m256d col3 = _mm256_permute2f128_pd(t01hi, t23hi, 0x31); // [r0c3,r1c3,r2c3,r3c3]

    __m256d col4 = _mm256_insertf128_pd(t45lo,
        _mm256_castpd256_pd128(t67lo), 1); // [r4c0,r5c0,r6c0,r7c0]
    __m256d col5 = _mm256_insertf128_pd(t45hi,
        _mm256_castpd256_pd128(t67hi), 1);
    __m256d col6 = _mm256_permute2f128_pd(t45lo, t67lo, 0x31);
    __m256d col7 = _mm256_permute2f128_pd(t45hi, t67hi, 0x31);

    // high half 4 columns (cast to __m256i for integer path, corresponds to assembly vinserti128/vperm2i128)
    __m256i c0h = _mm256_inserti128_si256(
        _mm256_castpd_si256(t01lo_h),
        _mm256_castsi256_si128(_mm256_castpd_si256(t23lo_h)), 1);
    __m256i c1h = _mm256_inserti128_si256(
        _mm256_castpd_si256(t01hi_h),
        _mm256_castsi256_si128(_mm256_castpd_si256(t23hi_h)), 1);

    __m256i c2h = _mm256_permute2x128_si256(
        _mm256_castpd_si256(t01lo_h),
        _mm256_castpd_si256(t23lo_h), 0x31);
    __m256i c3h = _mm256_permute2x128_si256(
        _mm256_castpd_si256(t01hi_h),
        _mm256_castpd_si256(t23hi_h), 0x31);
    __m256i c4h = _mm256_inserti128_si256(
        _mm256_castpd_si256(t45lo_h),
        _mm256_castsi256_si128(_mm256_castpd_si256(t67lo_h)), 1);
    __m256i c5h = _mm256_inserti128_si256(
        _mm256_castpd_si256(t45hi_h),
        _mm256_castsi256_si128(_mm256_castpd_si256(t67hi_h)), 1);
    __m256i c6h = _mm256_permute2x128_si256(
        _mm256_castpd_si256(t45lo_h),
        _mm256_castpd_si256(t67lo_h), 0x31);
    __m256i c7h = _mm256_permute2x128_si256(
        _mm256_castpd_si256(t45hi_h),
        _mm256_castpd_si256(t67hi_h), 0x31);

    // ---------------------------------------------------------------
    // Step 4: vmovntps / vmovntdq — non-temporal store output
    //
    //   Transpose result: output row c = input column c
    //   Each row of 8 doubles = low 4 (col_*) + high 4 (c*h)
    //
    //   corresponding assembly: vmovntps writes low half, vmovntdq writes high half (both 32 bytes)
    //   Destination address = dst_row(dst, c), requires 32-byte alignment
    // ---------------------------------------------------------------
    // Each destination row should be composed by concatenating the same column from first 4 and last 4 source rows.
    // Previously, low/high halves were incorrectly paired as source cols 0-3 / 4-7, causing the result to not be a proper transpose matrix.
    _mm256_stream_pd(dst_row(dst, 0), col0);                          // [r0c0,r1c0,r2c0,r3c0]
    _mm256_stream_pd(dst_row(dst, 0) + 4, col4);                      // [r4c0,r5c0,r6c0,r7c0]
    _mm256_stream_pd(dst_row(dst, 1), col1);                          // [r0c1,r1c1,r2c1,r3c1]
    _mm256_stream_pd(dst_row(dst, 1) + 4, col5);                      // [r4c1,r5c1,r6c1,r7c1]
    _mm256_stream_pd(dst_row(dst, 2), col2);                          // [r0c2,r1c2,r2c2,r3c2]
    _mm256_stream_pd(dst_row(dst, 2) + 4, col6);                      // [r4c2,r5c2,r6c2,r7c2]
    _mm256_stream_pd(dst_row(dst, 3), col3);                          // [r0c3,r1c3,r2c3,r3c3]
    _mm256_stream_pd(dst_row(dst, 3) + 4, col7);                      // [r4c3,r5c3,r6c3,r7c3]

    _mm256_stream_pd(dst_row(dst, 4), _mm256_castsi256_pd(c0h));      // [r0c4,r1c4,r2c4,r3c4]
    _mm256_stream_pd(dst_row(dst, 4) + 4, _mm256_castsi256_pd(c4h));  // [r4c4,r5c4,r6c4,r7c4]
    _mm256_stream_pd(dst_row(dst, 5), _mm256_castsi256_pd(c1h));      // [r0c5,r1c5,r2c5,r3c5]
    _mm256_stream_pd(dst_row(dst, 5) + 4, _mm256_castsi256_pd(c5h));  // [r4c5,r5c5,r6c5,r7c5]
    _mm256_stream_pd(dst_row(dst, 6), _mm256_castsi256_pd(c2h));      // [r0c6,r1c6,r2c6,r3c6]
    _mm256_stream_pd(dst_row(dst, 6) + 4, _mm256_castsi256_pd(c6h));  // [r4c6,r5c6,r6c6,r7c6]
    _mm256_stream_pd(dst_row(dst, 7), _mm256_castsi256_pd(c3h));      // [r0c7,r1c7,r2c7,r3c7]
    _mm256_stream_pd(dst_row(dst, 7) + 4, _mm256_castsi256_pd(c7h));  // [r4c7,r5c7,r6c7,r7c7]

    // non-temporal store requires sfence, otherwise subsequent normal loads may not see the write result
    _mm_sfence();
}
static inline void transpose8x8_f64_nt_avx2_pf_v5(
    const double* src,
    std::ptrdiff_t src_ld,
    double* dst,
    std::ptrdiff_t dst_ld)
{
    _mm_prefetch(reinterpret_cast<const char*>(src + 0 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 1 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 2 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 3 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 4 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 5 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 6 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 7 * src_ld), _MM_HINT_NTA);

    __m256d r0, r1;

    // src rows 0-3 lo → produce dst[0..3][0..3]
    r0 = _mm256_loadu_pd(src + 0 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 1 * src_ld + 0);
    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 2 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 3 * src_ld + 0);
    __m256d t2 = _mm256_unpacklo_pd(r0, r1);
    __m256d t3 = _mm256_unpackhi_pd(r0, r1);

    __m256d d0_lo = _mm256_permute2f128_pd(t0, t2, 0x20); // dst row 0, cols 0-3
    __m256d d1_lo = _mm256_permute2f128_pd(t1, t3, 0x20); // dst row 1, cols 0-3
    __m256d d2_lo = _mm256_permute2f128_pd(t0, t2, 0x31); // dst row 2, cols 0-3
    __m256d d3_lo = _mm256_permute2f128_pd(t1, t3, 0x31); // dst row 3, cols 0-3

    // src rows 4-7 lo → produce dst[0..3][4..7]
    r0 = _mm256_loadu_pd(src + 4 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 5 * src_ld + 0);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 6 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 7 * src_ld + 0);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    __m256d d0_hi = _mm256_permute2f128_pd(t0, t2, 0x20); // dst row 0, cols 4-7
    __m256d d1_hi = _mm256_permute2f128_pd(t1, t3, 0x20); // dst row 1, cols 4-7
    __m256d d2_hi = _mm256_permute2f128_pd(t0, t2, 0x31); // dst row 2, cols 4-7
    __m256d d3_hi = _mm256_permute2f128_pd(t1, t3, 0x31); // dst row 3, cols 4-7

    // dst rows 0-3 are written contiguously in row-major order, WC buffer friendly
    _mm256_stream_pd(dst + 0 * dst_ld + 0, d0_lo);
    _mm256_stream_pd(dst + 0 * dst_ld + 4, d0_hi);
    _mm256_stream_pd(dst + 1 * dst_ld + 0, d1_lo);
    _mm256_stream_pd(dst + 1 * dst_ld + 4, d1_hi);
    _mm256_stream_pd(dst + 2 * dst_ld + 0, d2_lo);
    _mm256_stream_pd(dst + 2 * dst_ld + 4, d2_hi);
    _mm256_stream_pd(dst + 3 * dst_ld + 0, d3_lo);
    _mm256_stream_pd(dst + 3 * dst_ld + 4, d3_hi);

    // src rows 0-3 hi → produce dst[4..7][0..3]
    r0 = _mm256_loadu_pd(src + 0 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 1 * src_ld + 4);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 2 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 3 * src_ld + 4);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    __m256d d4_lo = _mm256_permute2f128_pd(t0, t2, 0x20); // dst row 4, cols 0-3
    __m256d d5_lo = _mm256_permute2f128_pd(t1, t3, 0x20); // dst row 5, cols 0-3
    __m256d d6_lo = _mm256_permute2f128_pd(t0, t2, 0x31); // dst row 6, cols 0-3
    __m256d d7_lo = _mm256_permute2f128_pd(t1, t3, 0x31); // dst row 7, cols 0-3

    // src rows 4-7 hi → produce dst[4..7][4..7]
    r0 = _mm256_loadu_pd(src + 4 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 5 * src_ld + 4);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 6 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 7 * src_ld + 4);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    __m256d d4_hi = _mm256_permute2f128_pd(t0, t2, 0x20); // dst row 4, cols 4-7
    __m256d d5_hi = _mm256_permute2f128_pd(t1, t3, 0x20); // dst row 5, cols 4-7
    __m256d d6_hi = _mm256_permute2f128_pd(t0, t2, 0x31); // dst row 6, cols 4-7
    __m256d d7_hi = _mm256_permute2f128_pd(t1, t3, 0x31); // dst row 7, cols 4-7

    // dst rows 4-7 are written contiguously in rows
    _mm256_stream_pd(dst + 4 * dst_ld + 0, d4_lo);
    _mm256_stream_pd(dst + 4 * dst_ld + 4, d4_hi);
    _mm256_stream_pd(dst + 5 * dst_ld + 0, d5_lo);
    _mm256_stream_pd(dst + 5 * dst_ld + 4, d5_hi);
    _mm256_stream_pd(dst + 6 * dst_ld + 0, d6_lo);
    _mm256_stream_pd(dst + 6 * dst_ld + 4, d6_hi);
    _mm256_stream_pd(dst + 7 * dst_ld + 0, d7_lo);
    _mm256_stream_pd(dst + 7 * dst_ld + 4, d7_hi);

    _mm_sfence();
}
static inline void transpose8x8_f64_nt_avx2_pf_v3(
    const double* src,
    std::ptrdiff_t src_ld,
    double* dst,
    std::ptrdiff_t dst_ld)
{
    _mm_prefetch(reinterpret_cast<const char*>(src + 0 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 1 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 2 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 3 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 4 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 5 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 6 * src_ld), _MM_HINT_NTA);
    _mm_prefetch(reinterpret_cast<const char*>(src + 7 * src_ld), _MM_HINT_NTA);

    __m256d r0, r1, t0, t1, t2, t3;

    // ── Block A: src cols 0-3, src rows 0-3 → dst rows 0-3 lo ──────────────
    r0 = _mm256_loadu_pd(src + 0 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 1 * src_ld + 0);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 2 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 3 * src_ld + 0);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    _mm256_stream_pd(dst + 0 * dst_ld + 0, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_stream_pd(dst + 2 * dst_ld + 0, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_stream_pd(dst + 1 * dst_ld + 0, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_stream_pd(dst + 3 * dst_ld + 0, _mm256_permute2f128_pd(t1, t3, 0x31));

    // ── Block B: src cols 0-3, src rows 4-7 → dst rows 4-7 lo ──────────────
    // t0~t3 are all reused
    r0 = _mm256_loadu_pd(src + 4 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 5 * src_ld + 0);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 6 * src_ld + 0);
    r1 = _mm256_loadu_pd(src + 7 * src_ld + 0);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    _mm256_stream_pd(dst + 4 * dst_ld + 0, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_stream_pd(dst + 6 * dst_ld + 0, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_stream_pd(dst + 5 * dst_ld + 0, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_stream_pd(dst + 7 * dst_ld + 0, _mm256_permute2f128_pd(t1, t3, 0x31));

    // ── Block C: src cols 4-7, src rows 0-3 → dst rows 0-3 hi ──────────────
    // t0~t3 are reused again
    r0 = _mm256_loadu_pd(src + 0 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 1 * src_ld + 4);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 2 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 3 * src_ld + 4);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    _mm256_stream_pd(dst + 0 * dst_ld + 4, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_stream_pd(dst + 2 * dst_ld + 4, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_stream_pd(dst + 1 * dst_ld + 4, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_stream_pd(dst + 3 * dst_ld + 4, _mm256_permute2f128_pd(t1, t3, 0x31));

    // ── Block D: src cols 4-7, src rows 4-7 → dst rows 4-7 hi ──────────────
    r0 = _mm256_loadu_pd(src + 4 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 5 * src_ld + 4);
    t0 = _mm256_unpacklo_pd(r0, r1);
    t1 = _mm256_unpackhi_pd(r0, r1);

    r0 = _mm256_loadu_pd(src + 6 * src_ld + 4);
    r1 = _mm256_loadu_pd(src + 7 * src_ld + 4);
    t2 = _mm256_unpacklo_pd(r0, r1);
    t3 = _mm256_unpackhi_pd(r0, r1);

    _mm256_stream_pd(dst + 4 * dst_ld + 4, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_stream_pd(dst + 6 * dst_ld + 4, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_stream_pd(dst + 5 * dst_ld + 4, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_stream_pd(dst + 7 * dst_ld + 4, _mm256_permute2f128_pd(t1, t3, 0x31));

    _mm_sfence();
}

/**
 * AVX2 8x8 non-temporal + prefetch micro-kernel variant v2.
 * Uses transpose8x8_f64_nt_avx2_pf_v2 as the inner kernel when AVX2 is available.
 */
template<int L2_TILE = 256, int L1_TILE = 32>
inline void transpose_2level_tuned_avx2_nt_pf_v2(const double* __restrict__ A,
                                                 double* __restrict__ AT,
                                                 int n) noexcept {
    static_assert(L2_TILE >= L1_TILE, "L2 tile must be >= L1 tile");
    static_assert(L1_TILE >= 8, "L1 tile must be >= 8 for the 8x8 micro-kernel");

#if defined(__AVX2__)
    // Simpler condition: only check if n is divisible by 8 for the kernel
    const bool use_stream_kernel = ((n & 7) == 0);
    const std::ptrdiff_t stride_bytes = static_cast<std::ptrdiff_t>(n) * sizeof(double);

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
                            if (use_stream_kernel) {
                                transpose8x8_f64_nt_avx2_pf_v2(
                                    A + ii * n + jj,
                                    AT + jj * n + ii,
                                    stride_bytes,
                                    stride_bytes);
                            } else {
                                // Use AVX2 4x4 kernel as fallback instead of scalar
                                const __m256d r0 = _mm256_loadu_pd(&A[(ii    ) * n + jj]);
                                const __m256d r1 = _mm256_loadu_pd(&A[(ii + 1) * n + jj]);
                                const __m256d r2 = _mm256_loadu_pd(&A[(ii + 2) * n + jj]);
                                const __m256d r3 = _mm256_loadu_pd(&A[(ii + 3) * n + jj]);
                                const __m256d r4 = _mm256_loadu_pd(&A[(ii + 4) * n + jj]);
                                const __m256d r5 = _mm256_loadu_pd(&A[(ii + 5) * n + jj]);
                                const __m256d r6 = _mm256_loadu_pd(&A[(ii + 6) * n + jj]);
                                const __m256d r7 = _mm256_loadu_pd(&A[(ii + 7) * n + jj]);

                                const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
                                const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
                                const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
                                const __m256d t3 = _mm256_unpackhi_pd(r2, r3);
                                const __m256d t4 = _mm256_unpacklo_pd(r4, r5);
                                const __m256d t5 = _mm256_unpackhi_pd(r4, r5);
                                const __m256d t6 = _mm256_unpacklo_pd(r6, r7);
                                const __m256d t7 = _mm256_unpackhi_pd(r6, r7);

                                const __m256d t8  = _mm256_unpacklo_pd(r4, r5);
                                const __m256d t9  = _mm256_unpackhi_pd(r4, r5);
                                const __m256d t10 = _mm256_unpacklo_pd(r6, r7);
                                const __m256d t11 = _mm256_unpackhi_pd(r6, r7);

                                const __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                                const __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                                const __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                                const __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);
                                const __m256d c4 = _mm256_permute2f128_pd(t8, t10, 0x20);
                                const __m256d c5 = _mm256_permute2f128_pd(t9, t11, 0x20);
                                const __m256d c6 = _mm256_permute2f128_pd(t8, t10, 0x31);
                                const __m256d c7 = _mm256_permute2f128_pd(t9, t11, 0x31);

                                _mm256_storeu_pd(&AT[(jj    ) * n + ii], c0);
                                _mm256_storeu_pd(&AT[(jj + 1) * n + ii], c1);
                                _mm256_storeu_pd(&AT[(jj + 2) * n + ii], c2);
                                _mm256_storeu_pd(&AT[(jj + 3) * n + ii], c3);
                                _mm256_storeu_pd(&AT[(jj + 4) * n + ii], c4);
                                _mm256_storeu_pd(&AT[(jj + 5) * n + ii], c5);
                                _mm256_storeu_pd(&AT[(jj + 6) * n + ii], c6);
                                _mm256_storeu_pd(&AT[(jj + 7) * n + ii], c7);
                            }
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
                            transpose_8x8_kernel(A + ii * n + jj, AT + jj * n + ii, n, n);
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




