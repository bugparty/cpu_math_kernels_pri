#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>

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

inline float hmax256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_max_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
    vlow = _mm_max_ps(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
    vlow = _mm_add_ps(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with 4x Unrolling and Register Reductions
// Target: AVX2 (Haswell+)
// Reason: Hides FMA and instruction latency via 4x unrolling, and replaces memory-based reductions with fast in-register tree reductions.
// Expected gain: ~2-3x throughput on large inputs over non-unrolled scalar reduction version.
inline void softmax_v2(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max (4x unrolled)
    std::size_t i = 0;
    __m256 max_v0 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max_v1 = max_v0;
    __m256 max_v2 = max_v0;
    __m256 max_v3 = max_v0;

    for (; i + 31 < n; i += 32) {
        max_v0 = _mm256_max_ps(max_v0, _mm256_loadu_ps(input + i));
        max_v1 = _mm256_max_ps(max_v1, _mm256_loadu_ps(input + i + 8));
        max_v2 = _mm256_max_ps(max_v2, _mm256_loadu_ps(input + i + 16));
        max_v3 = _mm256_max_ps(max_v3, _mm256_loadu_ps(input + i + 24));
    }

    // Fold 4 accumulators
    max_v0 = _mm256_max_ps(max_v0, max_v1);
    max_v2 = _mm256_max_ps(max_v2, max_v3);
    max_v0 = _mm256_max_ps(max_v0, max_v2);

    // Remainder loop for max
    for (; i + 7 < n; i += 8) {
        max_v0 = _mm256_max_ps(max_v0, _mm256_loadu_ps(input + i));
    }

    float max_val = hmax256_ps(max_v0);
    for (; i < n; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    __m256 max_vec = _mm256_set1_ps(max_val);

    // 2. Compute exp and sum (4x unrolled)
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

        __m256 e0 = exp256_ps(_mm256_sub_ps(x0, max_vec));
        __m256 e1 = exp256_ps(_mm256_sub_ps(x1, max_vec));
        __m256 e2 = exp256_ps(_mm256_sub_ps(x2, max_vec));
        __m256 e3 = exp256_ps(_mm256_sub_ps(x3, max_vec));

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum_v0 = _mm256_add_ps(sum_v0, e0);
        sum_v1 = _mm256_add_ps(sum_v1, e1);
        sum_v2 = _mm256_add_ps(sum_v2, e2);
        sum_v3 = _mm256_add_ps(sum_v3, e3);
    }

    // Fold 4 accumulators
    sum_v0 = _mm256_add_ps(sum_v0, sum_v1);
    sum_v2 = _mm256_add_ps(sum_v2, sum_v3);
    sum_v0 = _mm256_add_ps(sum_v0, sum_v2);

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp256_ps(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output + i, e);
        sum_v0 = _mm256_add_ps(sum_v0, e);
    }

    float sum_val = hsum256_ps(sum_v0);
    for (; i < n; ++i) {
        float e = std::exp(input[i] - max_val);
        output[i] = e;
        sum_val += e;
    }

    if (sum_val == 0.0f) return;

    // 3. Normalize (4x unrolled)
    float inv_sum = 1.0f / sum_val;
    __m256 inv_sum_v = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 31 < n; i += 32) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(_mm256_loadu_ps(output + i + 8), inv_sum_v));
        _mm256_storeu_ps(output + i + 16, _mm256_mul_ps(_mm256_loadu_ps(output + i + 16), inv_sum_v));
        _mm256_storeu_ps(output + i + 24, _mm256_mul_ps(_mm256_loadu_ps(output + i + 24), inv_sum_v));
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace ml_kernels
