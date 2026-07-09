#pragma once

#include "softmax.h"

namespace ml_kernels {

inline __m256 exp256_ps_v3(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    __m256 x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));

    __m256i n_int = _mm256_cvtps_epi32(x_log2e);
    __m256 n = _mm256_cvtepi32_ps(n_int);

    // ⚡ Thunderbolt: Single FMA for range reduction
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);

    // Horner's scheme
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(1.0f / 2.0f);
    __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);

    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c1);

    __m256i exp_shift = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
    __m256i exp_shifted = _mm256_slli_epi32(exp_shift, 23);
    __m256 exp2n = _mm256_castsi256_ps(exp_shifted);

    return _mm256_mul_ps(p, exp2n);
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with single-FMA exp256 and aggressive 8x unroll
// Target: AVX2 (Haswell+)
// Reason: Combining constants for ln(2) into a single FMA instruction reduces instruction count and register pressure,
// allowing aggressive 8x (64 elements) unrolling across all phases (Max, Exp/Sum, Normalize) to fully utilize YMM registers
// and shift bottlenecks to L1/L2 bandwidth.
// Expected gain: Better throughput over v5.
inline void softmax_v6(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    std::size_t i = 0;

    // 1. Find max (8x unrolled - 64 elements)
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;
    __m256 max4 = max_v, max5 = max_v, max6 = max_v, max7 = max_v;

    for (; i + 63 < n; i += 64) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
        max4 = _mm256_max_ps(max4, _mm256_loadu_ps(input + i + 32));
        max5 = _mm256_max_ps(max5, _mm256_loadu_ps(input + i + 40));
        max6 = _mm256_max_ps(max6, _mm256_loadu_ps(input + i + 48));
        max7 = _mm256_max_ps(max7, _mm256_loadu_ps(input + i + 56));
    }
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max4 = _mm256_max_ps(max4, max5);
    max6 = _mm256_max_ps(max6, max7);
    max0 = _mm256_max_ps(max0, max2);
    max4 = _mm256_max_ps(max4, max6);
    max0 = _mm256_max_ps(max0, max4);

    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = reduce_max(max0);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum (8x unrolled)
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    for (; i + 63 < n; i += 64) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);
        __m256 x4 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 32), max_vec);
        __m256 x5 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 40), max_vec);
        __m256 x6 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 48), max_vec);
        __m256 x7 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 56), max_vec);

        __m256 e0 = exp256_ps_v3(x0);
        __m256 e1 = exp256_ps_v3(x1);
        __m256 e2 = exp256_ps_v3(x2);
        __m256 e3 = exp256_ps_v3(x3);
        __m256 e4 = exp256_ps_v3(x4);
        __m256 e5 = exp256_ps_v3(x5);
        __m256 e6 = exp256_ps_v3(x6);
        __m256 e7 = exp256_ps_v3(x7);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);
        _mm256_storeu_ps(output + i + 32, e4);
        _mm256_storeu_ps(output + i + 40, e5);
        _mm256_storeu_ps(output + i + 48, e6);
        _mm256_storeu_ps(output + i + 56, e7);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
        sum4 = _mm256_add_ps(sum4, e4);
        sum5 = _mm256_add_ps(sum5, e5);
        sum6 = _mm256_add_ps(sum6, e6);
        sum7 = _mm256_add_ps(sum7, e7);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum4 = _mm256_add_ps(sum4, sum5);
    sum6 = _mm256_add_ps(sum6, sum7);
    sum0 = _mm256_add_ps(sum0, sum2);
    sum4 = _mm256_add_ps(sum4, sum6);
    sum0 = _mm256_add_ps(sum0, sum4);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps_v3(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output + i, e);
        sum0 = _mm256_add_ps(sum0, e);
    }

    float sum_val = reduce_sum(sum0);
    for (; i < n; ++i) {
        float e = std::exp(input[i] - max_val);
        output[i] = e;
        sum_val += e;
    }

    if (sum_val == 0.0f) return;

    // 3. Normalize (8x unrolled)
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 63 < n; i += 64) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);
        __m256 o4 = _mm256_loadu_ps(output + i + 32);
        __m256 o5 = _mm256_loadu_ps(output + i + 40);
        __m256 o6 = _mm256_loadu_ps(output + i + 48);
        __m256 o7 = _mm256_loadu_ps(output + i + 56);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);
        __m256 m4 = _mm256_mul_ps(o4, inv_sum_v);
        __m256 m5 = _mm256_mul_ps(o5, inv_sum_v);
        __m256 m6 = _mm256_mul_ps(o6, inv_sum_v);
        __m256 m7 = _mm256_mul_ps(o7, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
        _mm256_storeu_ps(output + i + 32, m4);
        _mm256_storeu_ps(output + i + 40, m5);
        _mm256_storeu_ps(output + i + 48, m6);
        _mm256_storeu_ps(output + i + 56, m7);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace ml_kernels
