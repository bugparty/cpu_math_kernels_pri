#ifndef TRANSPOSE_PREFETCH_H
#define TRANSPOSE_PREFETCH_H

#include <algorithm>
#include "../include/compiler_compat.h"

/**
 * Tiled transpose with explicit software prefetch.
 *
 * Strategy:
 *   - Source (A) is read row-major (sequential), but we still prefetch
 *     PFETCH_SRC rows ahead so the data is warm in L2 before it hits the
 *     inner loop.
 *   - Destination (AT) is written column-major (stride-n), so every write
 *     to AT[j*n + i] is a potential cache miss.  We prefetch PFETCH_DST
 *     columns ahead into L2 with write-intent.
 *   - The inner j-loop is manually unrolled by 4 to amortise the prefetch
 *     instruction overhead.
 *
 * Tile size 64x64 matches the existing variants in common.h.
 */
inline void transpose_prefetch(const double* __restrict__ A,
                               double* __restrict__ AT,
                               int n) noexcept {
    constexpr int TILE       = 64;
    constexpr int PFETCH_SRC = 8;   // prefetch source rows this far ahead
    constexpr int PFETCH_DST = 4;   // prefetch destination cols this far ahead (also unroll factor)

    // ---- full tiles --------------------------------------------------------
    for (int ti = 0; ti + TILE <= n; ti += TILE) {
        for (int tj = 0; tj + TILE <= n; tj += TILE) {
            for (int i = ti; i < ti + TILE; ++i) {
                // Prefetch a future source row into L2 (read, temporal hint L2)
                if (i + PFETCH_SRC < ti + TILE) {
                    __builtin_prefetch(&A[(i + PFETCH_SRC) * n + tj], 0, 2);
                }

                for (int j = tj; j < tj + TILE; j += PFETCH_DST) {
                    // Prefetch future destination column positions (write, L2)
                    if (j + PFETCH_DST < tj + TILE) {
                        __builtin_prefetch(&AT[(j + PFETCH_DST) * n + i], 1, 1);
                    }
                    // 4-way unrolled transpose copy
                    AT[(j    ) * n + i] = A[i * n + (j    )];
                    AT[(j + 1) * n + i] = A[i * n + (j + 1)];
                    AT[(j + 2) * n + i] = A[i * n + (j + 2)];
                    AT[(j + 3) * n + i] = A[i * n + (j + 3)];
                }
            }
        }
    }

    // ---- boundary (scalar) -------------------------------------------------
    for (int ti = 0; ti < n; ti += TILE) {
        for (int tj = 0; tj < n; tj += TILE) {
            if (ti + TILE <= n && tj + TILE <= n) continue; // already done
            const int ie = std::min(ti + TILE, n);
            const int je = std::min(tj + TILE, n);
            for (int i = ti; i < ie; ++i)
                for (int j = tj; j < je; ++j)
                    AT[j * n + i] = A[i * n + j];
        }
    }
}

/**
 * Variant 2: two-level prefetch (L1 + L2) with larger unroll.
 *
 * Uses a short-distance L1 prefetch (2 rows/cols ahead) and a longer-
 * distance L2 prefetch (8 rows/cols ahead) to keep both cache levels fed.
 * Inner loop unrolled by 8.
 */
inline void transpose_prefetch_v2(const double* __restrict__ A,
                                  double* __restrict__ AT,
                                  int n) noexcept {
    constexpr int TILE        = 64;
    constexpr int PF_SRC_L2   = 8;
    constexpr int PF_SRC_L1   = 2;
    constexpr int PF_DST_L2   = 8;
    constexpr int PF_DST_L1   = 2;
    constexpr int UNROLL      = 8;

    // ---- full tiles --------------------------------------------------------
    for (int ti = 0; ti + TILE <= n; ti += TILE) {
        for (int tj = 0; tj + TILE <= n; tj += TILE) {
            for (int i = ti; i < ti + TILE; ++i) {
                // L2 prefetch: source row PF_SRC_L2 rows ahead
                if (i + PF_SRC_L2 < ti + TILE) {
                    __builtin_prefetch(&A[(i + PF_SRC_L2) * n + tj], 0, 2);
                    __builtin_prefetch(&A[(i + PF_SRC_L2) * n + tj + 32], 0, 2);
                }
                // L1 prefetch: source row PF_SRC_L1 rows ahead
                if (i + PF_SRC_L1 < ti + TILE) {
                    __builtin_prefetch(&A[(i + PF_SRC_L1) * n + tj], 0, 3);
                    __builtin_prefetch(&A[(i + PF_SRC_L1) * n + tj + 32], 0, 3);
                }

                for (int j = tj; j < tj + TILE; j += UNROLL) {
                    // L2 prefetch destination columns ahead
                    if (j + PF_DST_L2 < tj + TILE) {
                        __builtin_prefetch(&AT[(j + PF_DST_L2) * n + i], 1, 1);
                    }
                    // L1 prefetch destination columns (closer)
                    if (j + PF_DST_L1 < tj + TILE) {
                        __builtin_prefetch(&AT[(j + PF_DST_L1) * n + i], 1, 3);
                    }

                    AT[(j    ) * n + i] = A[i * n + (j    )];
                    AT[(j + 1) * n + i] = A[i * n + (j + 1)];
                    AT[(j + 2) * n + i] = A[i * n + (j + 2)];
                    AT[(j + 3) * n + i] = A[i * n + (j + 3)];
                    AT[(j + 4) * n + i] = A[i * n + (j + 4)];
                    AT[(j + 5) * n + i] = A[i * n + (j + 5)];
                    AT[(j + 6) * n + i] = A[i * n + (j + 6)];
                    AT[(j + 7) * n + i] = A[i * n + (j + 7)];
                }
            }
        }
    }

    // ---- boundary ----------------------------------------------------------
    for (int ti = 0; ti < n; ti += TILE) {
        for (int tj = 0; tj < n; tj += TILE) {
            if (ti + TILE <= n && tj + TILE <= n) continue;
            const int ie = std::min(ti + TILE, n);
            const int je = std::min(tj + TILE, n);
            for (int i = ti; i < ie; ++i)
                for (int j = tj; j < je; ++j)
                    AT[j * n + i] = A[i * n + j];
        }
    }
}

/**
 * Variant 3: batch-prefetch at tile start.
 *
 * Before processing each tile, prefetch the first several rows of the
 * source tile and all destination column heads.  This "primes the pump"
 * so that the first iterations of the inner loops don't stall.  Ongoing
 * prefetching within the loop keeps the pipeline full.
 */
inline void transpose_prefetch_v3(const double* __restrict__ A,
                                  double* __restrict__ AT,
                                  int n) noexcept {
    constexpr int TILE       = 64;
    constexpr int PFETCH_DST = 4;

    // ---- full tiles --------------------------------------------------------
    for (int ti = 0; ti + TILE <= n; ti += TILE) {
        for (int tj = 0; tj + TILE <= n; tj += TILE) {
            // Prime: prefetch first 8 source rows of this tile into L2
            for (int p = 0; p < 8; ++p) {
                __builtin_prefetch(&A[(ti + p) * n + tj],      0, 2);
                __builtin_prefetch(&A[(ti + p) * n + tj + 32], 0, 2);
            }
            // Prime: prefetch first 8 destination column heads
            for (int p = 0; p < 8; ++p) {
                __builtin_prefetch(&AT[(tj + p) * n + ti], 1, 1);
            }

            for (int i = ti; i < ti + TILE; ++i) {
                // Rolling source prefetch
                if (i + 8 < ti + TILE) {
                    __builtin_prefetch(&A[(i + 8) * n + tj],      0, 2);
                    __builtin_prefetch(&A[(i + 8) * n + tj + 32], 0, 2);
                }

                for (int j = tj; j < tj + TILE; j += PFETCH_DST) {
                    // Rolling destination prefetch
                    if (j + PFETCH_DST < tj + TILE) {
                        __builtin_prefetch(&AT[(j + PFETCH_DST) * n + i], 1, 1);
                    }

                    AT[(j    ) * n + i] = A[i * n + (j    )];
                    AT[(j + 1) * n + i] = A[i * n + (j + 1)];
                    AT[(j + 2) * n + i] = A[i * n + (j + 2)];
                    AT[(j + 3) * n + i] = A[i * n + (j + 3)];
                }
            }
        }
    }

    // ---- boundary ----------------------------------------------------------
    for (int ti = 0; ti < n; ti += TILE) {
        for (int tj = 0; tj < n; tj += TILE) {
            if (ti + TILE <= n && tj + TILE <= n) continue;
            const int ie = std::min(ti + TILE, n);
            const int je = std::min(tj + TILE, n);
            for (int i = ti; i < ie; ++i)
                for (int j = tj; j < je; ++j)
                    AT[j * n + i] = A[i * n + j];
        }
    }
}

#endif // TRANSPOSE_PREFETCH_H
