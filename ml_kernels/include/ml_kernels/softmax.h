#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace ml_kernels {

inline void exp256_ps_4x(__m256& x0, __m256& x1, __m256& x2, __m256& x3) {
    auto clamp = _mm256_set1_ps(-87.3f);
    x0 = _mm256_max_ps(x0, clamp);
    x1 = _mm256_max_ps(x1, clamp);
    x2 = _mm256_max_ps(x2, clamp);
    x3 = _mm256_max_ps(x3, clamp);

    auto log2e = _mm256_set1_ps(1.4426950408889634f);
    auto x0_log2e = _mm256_mul_ps(x0, log2e);
    auto x1_log2e = _mm256_mul_ps(x1, log2e);
    auto x2_log2e = _mm256_mul_ps(x2, log2e);
    auto x3_log2e = _mm256_mul_ps(x3, log2e);

    auto n0 = _mm256_round_ps(x0_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto n1 = _mm256_round_ps(x1_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto n2 = _mm256_round_ps(x2_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto n3 = _mm256_round_ps(x3_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    auto ln2_hi = _mm256_set1_ps(0.693145751953125f);
    auto r0 = _mm256_sub_ps(x0, _mm256_mul_ps(n0, ln2_hi));
    auto r1 = _mm256_sub_ps(x1, _mm256_mul_ps(n1, ln2_hi));
    auto r2 = _mm256_sub_ps(x2, _mm256_mul_ps(n2, ln2_hi));
    auto r3 = _mm256_sub_ps(x3, _mm256_mul_ps(n3, ln2_hi));

    auto ln2_lo = _mm256_set1_ps(1.428606765330187e-06f);
    r0 = _mm256_sub_ps(r0, _mm256_mul_ps(n0, ln2_lo));
    r1 = _mm256_sub_ps(r1, _mm256_mul_ps(n1, ln2_lo));
    r2 = _mm256_sub_ps(r2, _mm256_mul_ps(n2, ln2_lo));
    r3 = _mm256_sub_ps(r3, _mm256_mul_ps(n3, ln2_lo));

    auto c1 = _mm256_set1_ps(1.0f);
    auto c2 = _mm256_set1_ps(1.0f / 2.0f);
    auto c3 = _mm256_set1_ps(1.0f / 6.0f);
    auto c4 = _mm256_set1_ps(1.0f / 24.0f);
    auto c5 = _mm256_set1_ps(1.0f / 120.0f);

    auto p0 = c5;
    auto p1 = c5;
    auto p2 = c5;
    auto p3 = c5;

    p0 = _mm256_fmadd_ps(p0, r0, c4);
    p1 = _mm256_fmadd_ps(p1, r1, c4);
    p2 = _mm256_fmadd_ps(p2, r2, c4);
    p3 = _mm256_fmadd_ps(p3, r3, c4);

    p0 = _mm256_fmadd_ps(p0, r0, c3);
    p1 = _mm256_fmadd_ps(p1, r1, c3);
    p2 = _mm256_fmadd_ps(p2, r2, c3);
    p3 = _mm256_fmadd_ps(p3, r3, c3);

    p0 = _mm256_fmadd_ps(p0, r0, c2);
    p1 = _mm256_fmadd_ps(p1, r1, c2);
    p2 = _mm256_fmadd_ps(p2, r2, c2);
    p3 = _mm256_fmadd_ps(p3, r3, c2);

    p0 = _mm256_fmadd_ps(p0, r0, c1);
    p1 = _mm256_fmadd_ps(p1, r1, c1);
    p2 = _mm256_fmadd_ps(p2, r2, c1);
    p3 = _mm256_fmadd_ps(p3, r3, c1);

    p0 = _mm256_fmadd_ps(p0, r0, c1);
    p1 = _mm256_fmadd_ps(p1, r1, c1);
    p2 = _mm256_fmadd_ps(p2, r2, c1);
    p3 = _mm256_fmadd_ps(p3, r3, c1);

    auto exp_offset = _mm256_set1_epi32(127);

    auto exp2n_0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n0), exp_offset), 23));
    auto exp2n_1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n1), exp_offset), 23));
    auto exp2n_2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n2), exp_offset), 23));
    auto exp2n_3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n3), exp_offset), 23));

    x0 = _mm256_mul_ps(p0, exp2n_0);
    x1 = _mm256_mul_ps(p1, exp2n_1);
    x2 = _mm256_mul_ps(p2, exp2n_2);
    x3 = _mm256_mul_ps(p3, exp2n_3);
}

inline __m256 exp256_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    __m256 x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 n = _mm256_round_ps(x_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693145751953125f)));
    r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(1.428606765330187e-06f)));

    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(1.0f / 2.0f);
    __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);

    __m256 p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c1);

    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256i exp_shift = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
    __m256i exp_shifted = _mm256_slli_epi32(exp_shift, 23);
    __m256 exp2n = _mm256_castsi256_ps(exp_shifted);

    return _mm256_mul_ps(p, exp2n);
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax
// Target: AVX2 (Haswell+)
// Reason: Replaces scalar pass with fully vectorized max, exp, and inverse-sum normalization. Unrolls loop 4x to hide FMA latency in exp calculation.
// Expected gain: ~4-5x throughput on large inputs by avoiding scalar math and div latency, and up to ~20-30% beyond that from 4x loop unrolling.
inline void softmax_v4(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(-INFINITY);
    __m256 max_v0 = _mm256_set1_ps(-INFINITY);
    __m256 max_v1 = _mm256_set1_ps(-INFINITY);
    __m256 max_v2 = _mm256_set1_ps(-INFINITY);
    __m256 max_v3 = _mm256_set1_ps(-INFINITY);

    for (; i + 31 < n; i += 32) {
        max_v0 = _mm256_max_ps(max_v0, _mm256_loadu_ps(input + i));
        max_v1 = _mm256_max_ps(max_v1, _mm256_loadu_ps(input + i + 8));
        max_v2 = _mm256_max_ps(max_v2, _mm256_loadu_ps(input + i + 16));
        max_v3 = _mm256_max_ps(max_v3, _mm256_loadu_ps(input + i + 24));
    }
    max_v0 = _mm256_max_ps(max_v0, max_v1);
    max_v2 = _mm256_max_ps(max_v2, max_v3);
    max_v = _mm256_max_ps(max_v0, max_v2);

    for (; i + 7 < n; i += 8) {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(input + i));
    }
    float max_arr[8];
    _mm256_storeu_ps(max_arr, max_v);
    float max_val = max_arr[0];
    for (int j = 1; j < 8; ++j) max_val = std::max(max_val, max_arr[j]);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum_v0 = _mm256_setzero_ps();
    __m256 sum_v1 = _mm256_setzero_ps();
    __m256 sum_v2 = _mm256_setzero_ps();
    __m256 sum_v3 = _mm256_setzero_ps();

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        x0 = _mm256_sub_ps(x0, max_vec);
        x1 = _mm256_sub_ps(x1, max_vec);
        x2 = _mm256_sub_ps(x2, max_vec);
        x3 = _mm256_sub_ps(x3, max_vec);

        exp256_ps_4x(x0, x1, x2, x3);

        _mm256_storeu_ps(output + i, x0);
        _mm256_storeu_ps(output + i + 8, x1);
        _mm256_storeu_ps(output + i + 16, x2);
        _mm256_storeu_ps(output + i + 24, x3);

        sum_v0 = _mm256_add_ps(sum_v0, x0);
        sum_v1 = _mm256_add_ps(sum_v1, x1);
        sum_v2 = _mm256_add_ps(sum_v2, x2);
        sum_v3 = _mm256_add_ps(sum_v3, x3);
    }

    sum_v0 = _mm256_add_ps(sum_v0, sum_v1);
    sum_v2 = _mm256_add_ps(sum_v2, sum_v3);
    __m256 sum_v = _mm256_add_ps(sum_v0, sum_v2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output + i, e);
        sum_v = _mm256_add_ps(sum_v, e);
    }
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_v);
    float sum_val = 0.0f;
    for (int j = 0; j < 8; ++j) sum_val += sum_arr[j];
    for (; i < n; ++i) {
        float e = std::exp(input[i] - max_val);
        output[i] = e;
        sum_val += e;
    }

    if (sum_val == 0.0f) return;

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        o0 = _mm256_mul_ps(o0, inv_sum_v);
        o1 = _mm256_mul_ps(o1, inv_sum_v);
        o2 = _mm256_mul_ps(o2, inv_sum_v);
        o3 = _mm256_mul_ps(o3, inv_sum_v);

        _mm256_storeu_ps(output + i, o0);
        _mm256_storeu_ps(output + i + 8, o1);
        _mm256_storeu_ps(output + i + 16, o2);
        _mm256_storeu_ps(output + i + 24, o3);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace ml_kernels
