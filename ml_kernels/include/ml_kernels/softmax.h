#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <limits>

namespace ml_kernels {


inline __m256 exp256_ps_estrin(__m256 x) {
    // Range reduction: exp(x) = 2^(x * log2(e)) = 2^(n + f)
    // Clamp x to avoid underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));

    __m256 x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 n = _mm256_round_ps(x_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - n * ln(2). Split ln(2) for precision
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693145751953125f)));
    r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(1.428606765330187e-06f)));

    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(1.0f / 2.0f);
    __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);

    // ⚡ Thunderbolt: break dependency chain using Estrin's scheme instead of Horner's method
    __m256 p01 = _mm256_fmadd_ps(c1, r, c1);
    __m256 p23 = _mm256_fmadd_ps(c3, r, c2);
    __m256 p45 = _mm256_fmadd_ps(c5, r, c4);

    __m256 r2 = _mm256_mul_ps(r, r);

    __m256 p03 = _mm256_fmadd_ps(p23, r2, p01);

    __m256 r4 = _mm256_mul_ps(r2, r2);
    __m256 p = _mm256_fmadd_ps(p45, r4, p03);

    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256i exp_shift = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
    __m256i exp_shifted = _mm256_slli_epi32(exp_shift, 23);
    __m256 exp2n = _mm256_castsi256_ps(exp_shifted);

    return _mm256_mul_ps(p, exp2n);
}
inline __m256 exp256_ps(__m256 x) {
    // Range reduction: exp(x) = 2^(x * log2(e)) = 2^(n + f)
    // Clamp x to avoid underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));

    __m256 x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 n = _mm256_round_ps(x_log2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - n * ln(2). Split ln(2) for precision
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
// Reason: Replaces scalar pass with fully vectorized max, exp, and inverse-sum normalization.
// Expected gain: ~4-5x throughput on large inputs by avoiding scalar math and div latency.
inline void softmax_v2(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(-INFINITY);
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
    __m256 sum_v = _mm256_setzero_ps();
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
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with 4x unrolling and instruction interleaving
// Target: AVX2 (Haswell+)
// Reason: Explicit interleaving of loads/subs and exp evaluations breaks FMA latency chains, giving the out-of-order scheduler 4 independent streams.
// Expected gain: ~5-10% throughput improvement over standard 4x unroll by hiding exp256_ps latency.
inline float reduce_max(__m256 v) {
    __m256 t1 = _mm256_permute2f128_ps(v, v, 1);
    v = _mm256_max_ps(v, t1);
    __m256 t2 = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm256_max_ps(v, t2);
    __m256 t3 = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_max_ps(v, t3);
    return _mm256_cvtss_f32(v);
}

inline float reduce_sum(__m256 v) {
    __m256 t1 = _mm256_permute2f128_ps(v, v, 1);
    v = _mm256_add_ps(v, t1);
    __m256 t2 = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm256_add_ps(v, t2);
    __m256 t3 = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_add_ps(v, t3);
    return _mm256_cvtss_f32(v);
}

inline void softmax_v3(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = reduce_max(max0);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);

        __m256 e0 = exp256_ps(x0);
        __m256 e1 = exp256_ps(x1);
        __m256 e2 = exp256_ps(x2);
        __m256 e3 = exp256_ps(x3);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps(_mm256_sub_ps(x, max_vec));
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

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with Estrin's scheme for exp256
// Target: AVX2 (Haswell+)
// Reason: Uses Estrin's scheme in exp256_ps_estrin instead of Horner's method, improving ILP for exp computation
// Expected gain: ~10% over softmax_v3 due to reduced exp256 latency.
inline void softmax_v4(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = reduce_max(max0);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);

        __m256 e0 = exp256_ps_estrin(x0);
        __m256 e1 = exp256_ps_estrin(x1);
        __m256 e2 = exp256_ps_estrin(x2);
        __m256 e3 = exp256_ps_estrin(x3);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps_estrin(_mm256_sub_ps(x, max_vec));
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

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

inline __m256 exp256_ps_v2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    __m256 x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));

    // cvtps_epi32 defaults to round-to-nearest in AVX2, avoiding round_ps
    __m256i n_int = _mm256_cvtps_epi32(x_log2e);
    __m256 n = _mm256_cvtepi32_ps(n_int);

    // Use fnmadd to do r = x - n*ln2
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-06f), r);

    // Horner's scheme instead of Estrin
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

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with FMA-optimized exp256
// Target: AVX2 (Haswell+)
// Reason: Avoids `round_ps` by leveraging `cvtps_epi32` rounding mode, and replaces Estrin's scheme with Horner's.
// When unrolled 4x, the independent Horner chains interleave perfectly, saturating execution ports and hiding latency better than Estrin, leading to higher throughput.
// Expected gain: ~15-25% over softmax_v4.
inline void softmax_v5(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = reduce_max(max0);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);

        __m256 e0 = exp256_ps_v2(x0);
        __m256 e1 = exp256_ps_v2(x1);
        __m256 e2 = exp256_ps_v2(x2);
        __m256 e3 = exp256_ps_v2(x3);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps_v2(_mm256_sub_ps(x, max_vec));
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

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}


// ⚡ Thunderbolt: Explicitly Interleaved AVX2 Softmax
// Target: AVX2 (Haswell+)
// Reason: Manual instruction interleaving of 4x unrolled exp256 breaks FMA latency chains
// Expected gain: ~10% throughput over softmax_v5
inline void softmax_v6(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = reduce_max(max0);
    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(1.0f / 2.0f);
    __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
    __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    __m256 ln2_lo = _mm256_set1_ps(1.428606765330187e-06f);
    __m256 min_val = _mm256_set1_ps(-87.3f);
    __m256i shift127 = _mm256_set1_epi32(127);

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);

        x0 = _mm256_max_ps(x0, min_val);
        x1 = _mm256_max_ps(x1, min_val);
        x2 = _mm256_max_ps(x2, min_val);
        x3 = _mm256_max_ps(x3, min_val);

        __m256 x0_log2e = _mm256_mul_ps(x0, log2e);
        __m256 x1_log2e = _mm256_mul_ps(x1, log2e);
        __m256 x2_log2e = _mm256_mul_ps(x2, log2e);
        __m256 x3_log2e = _mm256_mul_ps(x3, log2e);

        __m256i n0_int = _mm256_cvtps_epi32(x0_log2e);
        __m256i n1_int = _mm256_cvtps_epi32(x1_log2e);
        __m256i n2_int = _mm256_cvtps_epi32(x2_log2e);
        __m256i n3_int = _mm256_cvtps_epi32(x3_log2e);

        __m256 n0 = _mm256_cvtepi32_ps(n0_int);
        __m256 n1 = _mm256_cvtepi32_ps(n1_int);
        __m256 n2 = _mm256_cvtepi32_ps(n2_int);
        __m256 n3 = _mm256_cvtepi32_ps(n3_int);

        __m256 r0 = _mm256_fnmadd_ps(n0, ln2_hi, x0);
        __m256 r1 = _mm256_fnmadd_ps(n1, ln2_hi, x1);
        __m256 r2 = _mm256_fnmadd_ps(n2, ln2_hi, x2);
        __m256 r3 = _mm256_fnmadd_ps(n3, ln2_hi, x3);

        r0 = _mm256_fnmadd_ps(n0, ln2_lo, r0);
        r1 = _mm256_fnmadd_ps(n1, ln2_lo, r1);
        r2 = _mm256_fnmadd_ps(n2, ln2_lo, r2);
        r3 = _mm256_fnmadd_ps(n3, ln2_lo, r3);

        __m256 p0 = _mm256_fmadd_ps(c5, r0, c4);
        __m256 p1 = _mm256_fmadd_ps(c5, r1, c4);
        __m256 p2 = _mm256_fmadd_ps(c5, r2, c4);
        __m256 p3 = _mm256_fmadd_ps(c5, r3, c4);

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

        __m256i exp_shift0 = _mm256_add_epi32(n0_int, shift127);
        __m256i exp_shift1 = _mm256_add_epi32(n1_int, shift127);
        __m256i exp_shift2 = _mm256_add_epi32(n2_int, shift127);
        __m256i exp_shift3 = _mm256_add_epi32(n3_int, shift127);

        __m256i exp_shifted0 = _mm256_slli_epi32(exp_shift0, 23);
        __m256i exp_shifted1 = _mm256_slli_epi32(exp_shift1, 23);
        __m256i exp_shifted2 = _mm256_slli_epi32(exp_shift2, 23);
        __m256i exp_shifted3 = _mm256_slli_epi32(exp_shift3, 23);

        __m256 exp2n0 = _mm256_castsi256_ps(exp_shifted0);
        __m256 exp2n1 = _mm256_castsi256_ps(exp_shifted1);
        __m256 exp2n2 = _mm256_castsi256_ps(exp_shifted2);
        __m256 exp2n3 = _mm256_castsi256_ps(exp_shifted3);

        __m256 e0 = _mm256_mul_ps(p0, exp2n0);
        __m256 e1 = _mm256_mul_ps(p1, exp2n1);
        __m256 e2 = _mm256_mul_ps(p2, exp2n2);
        __m256 e3 = _mm256_mul_ps(p3, exp2n3);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps_v2(_mm256_sub_ps(x, max_vec));
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

    // 3. Normalize
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}
} // namespace ml_kernels
