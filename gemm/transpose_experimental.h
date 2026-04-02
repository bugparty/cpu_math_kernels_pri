#ifndef TRANSPOSE_EXPERIMENTAL_H
#define TRANSPOSE_EXPERIMENTAL_H

#include <algorithm>

#if defined(_MSC_VER)
#define PRAGMA_UNROLL _Pragma("loop( hint_unroll(8) )")
#elif defined(__clang__)
#define PRAGMA_UNROLL _Pragma("clang loop unroll(enable)")
#elif defined(__GNUC__)
#define PRAGMA_UNROLL _Pragma("GCC unroll 8")
#else
#define PRAGMA_UNROLL
#endif

template<typename T, int TileSize = 64, int UnrollFactor = 8>
inline void transpose_tiled_v2(const T *A, T *A_T, const int n) noexcept {
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            const auto const_ti = ti;
            const auto const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; i += UnrollFactor) {
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                    A_T[j*n + i + 1] = A[(i + 1)*n + j];
                    A_T[j*n + i + 2] = A[(i + 2)*n + j];
                    A_T[j*n + i + 3] = A[(i + 3)*n + j];
                    A_T[j*n + i + 4] = A[(i + 4)*n + j];
                    A_T[j*n + i + 5] = A[(i + 5)*n + j];
                    A_T[j*n + i + 6] = A[(i + 6)*n + j];
                    A_T[j*n + i + 7] = A[(i + 7)*n + j];
                }
            }
        }
    }

    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            if (ti + TileSize <= n && tj + TileSize <= n) continue;

            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}

inline void transpose_tiled_v2(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v2<double, 64>(A, A_T, n);
}

template<typename T, int TileSize = 64, int UnrollFactor = 8>
inline void transpose_tiled_v3(const T *A, T *A_T, const int n) noexcept {
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            const auto const_ti = ti;
            const auto const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; i += UnrollFactor) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                    A_T[j*n + i + 1] = A[(i + 1)*n + j];
                    A_T[j*n + i + 2] = A[(i + 2)*n + j];
                    A_T[j*n + i + 3] = A[(i + 3)*n + j];
                    A_T[j*n + i + 4] = A[(i + 4)*n + j];
                    A_T[j*n + i + 5] = A[(i + 5)*n + j];
                    A_T[j*n + i + 6] = A[(i + 6)*n + j];
                    A_T[j*n + i + 7] = A[(i + 7)*n + j];
                }
            }
        }
    }

    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            if (ti + TileSize <= n && tj + TileSize <= n) continue;

            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
            for (int i = ti; i < ti_end; ++i) {
                for (int j = tj; j < tj_end; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }
}

inline void transpose_tiled_v3(const double *A, double *A_T, const int n) noexcept {
    transpose_tiled_v3<double, 64>(A, A_T, n);
}

template<typename T, int TileSize = 64>
inline void transpose_tiled_v4(const T *A, T *A_T, const int n) noexcept {
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            const auto const_ti = ti;
            const auto const_tj = tj;
            PRAGMA_UNROLL
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            if (ti + TileSize <= n && tj + TileSize <= n) continue;

            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
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
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            const auto const_ti = ti;
            const auto const_tj = tj;
            PRAGMA_UNROLL
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            if (ti + TileSize <= n && tj + TileSize <= n) continue;

            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
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
    for (int ti = 0; ti + TileSize <= n; ti += TileSize) {
        for (int tj = 0; tj + TileSize <= n; tj += TileSize) {
            const auto const_ti = ti;
            const auto const_tj = tj;
            for (int i = const_ti; i < const_ti + TileSize; ++i) {
                PRAGMA_UNROLL
                for (int j = const_tj; j < const_tj + TileSize; ++j) {
                    A_T[j*n + i] = A[i*n + j];
                }
            }
        }
    }

    for (int ti = 0; ti < n; ti += TileSize) {
        for (int tj = 0; tj < n; tj += TileSize) {
            if (ti + TileSize <= n && tj + TileSize <= n) continue;

            const int ti_end = std::min(ti + TileSize, n);
            const int tj_end = std::min(tj + TileSize, n);
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

#endif // TRANSPOSE_EXPERIMENTAL_H
