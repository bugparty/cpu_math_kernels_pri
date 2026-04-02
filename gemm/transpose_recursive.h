#ifndef TRANSPOSE_RECURSIVE_H
#define TRANSPOSE_RECURSIVE_H

#include "../include/compiler_compat.h"

namespace {  // anonymous namespace to avoid ODR issues

static constexpr int RECURSIVE_BASE = 32;

static void transpose_recursive_impl(const double* __restrict__ A, double* __restrict__ AT,
                                      int n, int ri, int cj, int rlen, int clen) noexcept {
    if (rlen <= RECURSIVE_BASE && clen <= RECURSIVE_BASE) {
        // Base case: small block, transpose with 4-wide unrolled inner loop
        for (int i = ri; i < ri + rlen; ++i) {
            const double* src_row = A + i * n;
            int j = cj;
            const int cj_end = cj + clen;
            const int cj_end4 = cj + (clen & ~3);  // round down to multiple of 4

            // Process 4 columns at a time
            for (; j < cj_end4; j += 4) {
                AT[(j + 0) * n + i] = src_row[j + 0];
                AT[(j + 1) * n + i] = src_row[j + 1];
                AT[(j + 2) * n + i] = src_row[j + 2];
                AT[(j + 3) * n + i] = src_row[j + 3];
            }
            // Remainder
            for (; j < cj_end; ++j) {
                AT[j * n + i] = src_row[j];
            }
        }
        return;
    }

    if (rlen >= clen) {
        int half = rlen / 2;
        transpose_recursive_impl(A, AT, n, ri, cj, half, clen);
        transpose_recursive_impl(A, AT, n, ri + half, cj, rlen - half, clen);
    } else {
        int half = clen / 2;
        transpose_recursive_impl(A, AT, n, ri, cj, rlen, half);
        transpose_recursive_impl(A, AT, n, ri, cj + half, rlen, clen - half);
    }
}

} // anonymous namespace

inline void transpose_recursive(const double* A, double* AT, int n) noexcept {
    if (__builtin_expect(n <= 0, 0)) return;
    transpose_recursive_impl(A, AT, n, 0, 0, n, n);
}

#endif // TRANSPOSE_RECURSIVE_H
