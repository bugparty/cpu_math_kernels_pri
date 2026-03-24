#ifndef COMMON_H
#define COMMON_H

#include <algorithm>

constexpr int TILE_SIZE = 64;

#if defined(_MSC_VER)
#define PRAGMA_UNROLL _Pragma("loop( hint_unroll(8) )")
#elif defined(__clang__)
#define PRAGMA_UNROLL _Pragma("clang loop unroll(enable)")
#elif defined(__GNUC__)
#define PRAGMA_UNROLL _Pragma("GCC unroll 8")
#else
#define PRAGMA_UNROLL
#endif

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
    transpose_tiled<double>(A, A_T, n);
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

/**
 * High-performance tiled transpose with template support
 * Divides matrix into TileSize x TileSize tiles to maximize cache efficiency
 * Template parameter allows tuning tile size for different cache hierarchies
 */
template<typename T, int TileSize = 64, int UnrollFactor = 8>
inline void transpose_tiled_v2(const T *A, T *A_T, const int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            // Full TileSize x TileSize tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; i+=UnrollFactor) {
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                    A_T[j*n + i + 1] = A[(i + 1)*n + j];
                    A_T[j*n + i + 2] = A[(i + 2)*n + j];
                    A_T[j*n + i + 3] = A[(i + 3)*n + j];
                    A_T[j*n + i + 4] = A[(i + 4)*n + j];
                    A_T[j*n + i + 5 ] = A[(i + 5)*n + j];
                    A_T[j*n + i + 6] = A[(i + 6)*n + j];
                    A_T[j*n + i + 7] = A[(i + 7)*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            // Skip if already processed as full tile
            if (ti + TileSize <= n && tj + TileSize <= n) continue;
            
            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}

// Convenience specialization for double with default TileSize=64
inline void transpose_tiled_v2(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v2<double, 64>(A, A_T, n);
}


template<typename T, int TileSize = 64, int UnrollFactor = 8>
inline void transpose_tiled_v3(const T *A, T *A_T, const int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            // Full TileSize x TileSize tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; i+=UnrollFactor) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                    A_T[j*n + i + 1] = A[(i + 1)*n + j];
                    A_T[j*n + i + 2] = A[(i + 2)*n + j];
                    A_T[j*n + i + 3] = A[(i + 3)*n + j];
                    A_T[j*n + i + 4] = A[(i + 4)*n + j];
                    A_T[j*n + i + 5 ] = A[(i + 5)*n + j];
                    A_T[j*n + i + 6] = A[(i + 6)*n + j];
                    A_T[j*n + i + 7] = A[(i + 7)*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            // Skip if already processed as full tile
            if (ti + TileSize <= n && tj + TileSize <= n) continue;
            
            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}
// Convenience specialization for double with default TileSize=64
inline void transpose_tiled_v3(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v3<double, 64>(A, A_T, n);
}


template<typename T, int TileSize = 64>
inline void transpose_tiled_v4(const T *A, T *A_T, const int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            // Full TileSize x TileSize tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            PRAGMA_UNROLL
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            // Skip if already processed as full tile
            if (ti + TileSize <= n && tj + TileSize <= n) continue;
            
            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}
inline void transpose_tiled_v4(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v4<double, 64>(A, A_T, n);
}


template<typename T, int TileSize = 64>
inline void transpose_tiled_v5(const T *A, T *A_T, const int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            // Full TileSize x TileSize tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            PRAGMA_UNROLL
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            // Skip if already processed as full tile
            if (ti + TileSize <= n && tj + TileSize <= n) continue;
            
            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}
inline void transpose_tiled_v5(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v5<double, 64>(A, A_T, n);
}
template<typename T, int TileSize = 64>
inline void transpose_tiled_v6(const T *A, T *A_T, const int n) noexcept {
    // Process full tiles (no boundary checks)
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            // Full TileSize x TileSize tile - no bounds checking
            auto const const_ti = ti;
            auto const const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    // Process boundary tiles (remainder)
    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            // Skip if already processed as full tile
            if (ti + TileSize <= n && tj + TileSize <= n) continue;
            
            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            
            // Partial tile at boundary
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}
inline void transpose_tiled_v6(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v6<double, 64>(A, A_T, n);
}
#undef PRAGMA_UNROLL

#endif // COMMON_H
