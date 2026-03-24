#ifndef TRANSPOSE_NONTEMPORAL_H
#define TRANSPOSE_NONTEMPORAL_H

#include <algorithm>
#include <immintrin.h>

// Non-temporal (streaming) store transpose.
// Bypasses cache on writes to avoid write-allocate overhead,
// since each output location is written once and not read again.

inline void transpose_nontemporal(const double* __restrict__ A,
                                  double* __restrict__ AT,
                                  int n) noexcept {
#ifdef __AVX512F__
    constexpr int TILE = 64;
    constexpr int MICRO = 8; // 8 doubles = 64 bytes = one cache line

    // Precompute gather indices: row offsets 0, n, 2n, ..., 7n (in elements)
    const __m512i vindex = _mm512_set_epi64(
        (long long)7 * n, (long long)6 * n, (long long)5 * n, (long long)4 * n,
        (long long)3 * n, (long long)2 * n, (long long)1 * n, 0LL
    );

    // Process full TILE x TILE blocks
    for (int ti = 0; ti + TILE <= n; ti += TILE) {
        for (int tj = 0; tj + TILE <= n; tj += TILE) {
            for (int ii = ti; ii < ti + TILE; ii += MICRO) {
                for (int j = tj; j < tj + TILE; ++j) {
                    // Gather 8 elements from column j, rows ii..ii+7
                    __m512d col = _mm512_i64gather_pd(vindex, &A[ii * n + j], sizeof(double));
                    // Stream-write to AT[j*n + ii .. ii+7] (contiguous, 64-byte aligned)
                    _mm512_stream_pd(&AT[j * n + ii], col);
                }
            }
        }
    }

    // Fence: ensure all NT stores are globally visible before any
    // subsequent loads could observe stale data
    _mm_sfence();

    // Boundary: rows that didn't fit into full TILE blocks
    {
        const int ti_start = (n / TILE) * TILE;
        for (int i = ti_start; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                AT[j * n + i] = A[i * n + j];
            }
        }
    }
    // Boundary: columns that didn't fit, but only for the rows already handled
    {
        const int tj_start = (n / TILE) * TILE;
        const int ti_end = (n / TILE) * TILE;
        for (int i = 0; i < ti_end; ++i) {
            for (int j = tj_start; j < n; ++j) {
                AT[j * n + i] = A[i * n + j];
            }
        }
    }

#elif defined(__AVX2__)
    constexpr int TILE = 64;
    constexpr int MICRO = 4; // 4 doubles = 32 bytes

    const __m256i vindex = _mm256_set_epi64x(
        (long long)3 * n, (long long)2 * n, (long long)1 * n, 0LL
    );

    for (int ti = 0; ti + TILE <= n; ti += TILE) {
        for (int tj = 0; tj + TILE <= n; tj += TILE) {
            for (int ii = ti; ii < ti + TILE; ii += MICRO) {
                for (int j = tj; j < tj + TILE; ++j) {
                    __m256d col = _mm256_i64gather_pd(&A[ii * n + j], vindex, sizeof(double));
                    _mm256_stream_pd(&AT[j * n + ii], col);
                }
            }
        }
    }

    _mm_sfence();

    // Boundary rows
    {
        const int ti_start = (n / TILE) * TILE;
        for (int i = ti_start; i < n; ++i)
            for (int j = 0; j < n; ++j)
                AT[j * n + i] = A[i * n + j];
    }
    // Boundary columns (full-tile rows only)
    {
        const int tj_start = (n / TILE) * TILE;
        const int ti_end = (n / TILE) * TILE;
        for (int i = 0; i < ti_end; ++i)
            for (int j = tj_start; j < n; ++j)
                AT[j * n + i] = A[i * n + j];
    }

#else
    // Scalar fallback with tiling
    constexpr int TILE = 64;

    for (int ti = 0; ti + TILE <= n; ti += TILE)
        for (int tj = 0; tj + TILE <= n; tj += TILE)
            for (int i = ti; i < ti + TILE; ++i)
                for (int j = tj; j < tj + TILE; ++j)
                    AT[j * n + i] = A[i * n + j];

    // Boundary rows
    {
        const int ti_start = (n / TILE) * TILE;
        for (int i = ti_start; i < n; ++i)
            for (int j = 0; j < n; ++j)
                AT[j * n + i] = A[i * n + j];
    }
    // Boundary columns
    {
        const int tj_start = (n / TILE) * TILE;
        const int ti_end = (n / TILE) * TILE;
        for (int i = 0; i < ti_end; ++i)
            for (int j = tj_start; j < n; ++j)
                AT[j * n + i] = A[i * n + j];
    }
#endif
}

#endif // TRANSPOSE_NONTEMPORAL_H
