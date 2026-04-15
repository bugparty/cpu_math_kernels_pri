#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <limits>

namespace ml_kernels {

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

// ⚡ Thunderbolt: AVX2 4x Unrolled Softmax with In-Register Reductions
// Target: AVX2 (Haswell+)
// Reason: Hides FMA and instruction latency by unrolling core loops 4x (32 elements). Replaces memory-bound array extraction with fast horizontal in-register tree reductions.
// Expected gain: ~1.5-2x over v2 due to increased instruction-level parallelism and elimination of store/load latency during reductions.
inline void softmax_v3(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max0 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max1 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max2 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max3 = _mm256_set1_ps(std::numeric_limits<float>::lowest());

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i + 0));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    __m256 max_v = _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));
    for (; i + 7 < n; i += 8) {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(input + i));
    }

    // Horizontal max reduction
    __m256 t1 = _mm256_permute2f128_ps(max_v, max_v, 1);
    __m256 m1 = _mm256_max_ps(max_v, t1);
    __m256 t2 = _mm256_shuffle_ps(m1, m1, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 m2 = _mm256_max_ps(m1, t2);
    __m256 t3 = _mm256_shuffle_ps(m2, m2, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 m3 = _mm256_max_ps(m2, t3);
    float max_val = _mm_cvtss_f32(_mm256_castps256_ps128(m3));

    for (; i < n; ++i) max_val = std::max(max_val, input[i]);

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum
    i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i + 0);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        __m256 e0 = exp256_ps(_mm256_sub_ps(x0, max_vec));
        __m256 e1 = exp256_ps(_mm256_sub_ps(x1, max_vec));
        __m256 e2 = exp256_ps(_mm256_sub_ps(x2, max_vec));
        __m256 e3 = exp256_ps(_mm256_sub_ps(x3, max_vec));

        _mm256_storeu_ps(output + i + 0, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }

    __m256 sum_v = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output + i, e);
        sum_v = _mm256_add_ps(sum_v, e);
    }

    // Horizontal sum reduction
    __m256 s1 = _mm256_add_ps(sum_v, _mm256_permute2f128_ps(sum_v, sum_v, 1));
    __m256 s2 = _mm256_add_ps(s1, _mm256_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2)));
    __m256 s3 = _mm256_add_ps(s2, _mm256_shuffle_ps(s2, s2, _MM_SHUFFLE(2, 3, 0, 1)));
    float sum_val = _mm_cvtss_f32(_mm256_castps256_ps128(s3));

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
        __m256 o0 = _mm256_loadu_ps(output + i + 0);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);

        _mm256_storeu_ps(output + i + 0, _mm256_mul_ps(o0, inv_sum_v));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(o1, inv_sum_v));
        _mm256_storeu_ps(output + i + 16, _mm256_mul_ps(o2, inv_sum_v));
        _mm256_storeu_ps(output + i + 24, _mm256_mul_ps(o3, inv_sum_v));
    }

    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace ml_kernels
