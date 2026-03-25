#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include "transpose_2level.h"

constexpr int TILE_SIZE = 64;

/**
 * High-performance tiled transpose with template support
 * Divides matrix into 64x64 tiles to maximize cache efficiency
 */
template<typename T>
inline void transpose_tiled(const T *A, T *A_T, int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TILE_SIZE <= n; ti += TILE_SIZE) {
        for (int tj = 0; tj + TILE_SIZE <= n; tj += TILE_SIZE) {
            // Full 64x64 tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            for (int i = const_ti; i < const_ti + TILE_SIZE; ++i) {
                for (int j = const_tj; j < const_tj + TILE_SIZE; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TILE_SIZE) {
        for (int tj = 0; tj < n; tj += TILE_SIZE) {
            // Skip if already processed as full tile
            if (ti + TILE_SIZE <= n && tj + TILE_SIZE <= n) continue;
            
            const int ti_end = std::min(ti + TILE_SIZE, n);
            const int tj_end = std::min(tj + TILE_SIZE, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}

// Specialized for double
inline void transpose_tiled(const double *A, double *A_T, int n) noexcept {
    transpose_2level_tuned_avx2_nt_pf_nofence<256, 32>(A, A_T, n);
}

/**
 * Naive transpose - simple and inefficient
 * For reference and benchmarking purposes
 */
template<typename T>
inline void transpose_naive(const T *A, T *A_T, int n) noexcept {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_T[j*n + i] = A[i*n + j];
        }
    }
}

// Specialized for double
inline void transpose_naive(const double *A, double *A_T, int n) noexcept {
    transpose_naive<double>(A, A_T, n);
}

#endif // COMMON_H
