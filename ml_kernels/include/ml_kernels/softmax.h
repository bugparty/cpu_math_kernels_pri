#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace ml_kernels {

// Taylor approximation for exp(x), x <= 0
// Using a 5th degree polynomial for good accuracy in [-87.3, 0]
inline __m256 exp_ps_taylor_avx2(__m256 x) {
    // clamp x to [-87.3, 0]
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(0.0f));

    __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 y = _mm256_mul_ps(x, log2e);

    // round to nearest integer
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 n_float = _mm256_cvtepi32_ps(n);

    // fractional part
    __m256 f = _mm256_sub_ps(y, n_float);

    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    __m256 c3 = _mm256_set1_ps(0.0555041086648215f);
    __m256 c4 = _mm256_set1_ps(0.0096181291076284f);
    __m256 c5 = _mm256_set1_ps(0.0013338312066657f);

    __m256 poly = _mm256_fmadd_ps(f, c5, c4);
    poly = _mm256_fmadd_ps(f, poly, c3);
    poly = _mm256_fmadd_ps(f, poly, c2);
    poly = _mm256_fmadd_ps(f, poly, c1);
    poly = _mm256_fmadd_ps(f, poly, c0);

    __m256i bias = _mm256_set1_epi32(127);
    __m256i n_biased = _mm256_add_epi32(n, bias);
    __m256i exp2n = _mm256_slli_epi32(n_biased, 23);

    __m256 result = _mm256_mul_ps(_mm256_castsi256_ps(exp2n), poly);

    return result;
}

inline void softmax_v2(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    float current_max = input[0];
    std::size_t i = 0;

    if (n >= 8) {
        __m256 vmax = _mm256_loadu_ps(input);
        for (i = 8; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        __m128 vmax128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(1, 0, 3, 2)));
        vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, _MM_SHUFFLE(0, 0, 0, 1)));
        current_max = _mm_cvtss_f32(vmax128);
    }

    for (; i < n; ++i) {
        if (input[i] > current_max) {
            current_max = input[i];
        }
    }

    // 2. Exp and Sum
    float sum = 0.0f;
    __m256 vmax_b = _mm256_set1_ps(current_max);

    i = 0;
    if (n >= 8) {
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            v = _mm256_sub_ps(v, vmax_b);
            __m256 vexp = exp_ps_taylor_avx2(v);
            _mm256_storeu_ps(output + i, vexp);
            vsum = _mm256_add_ps(vsum, vexp);
        }
        __m128 vsum128 = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
        vsum128 = _mm_add_ps(vsum128, _mm_shuffle_ps(vsum128, vsum128, _MM_SHUFFLE(1, 0, 3, 2)));
        vsum128 = _mm_add_ps(vsum128, _mm_shuffle_ps(vsum128, vsum128, _MM_SHUFFLE(0, 0, 0, 1)));
        sum += _mm_cvtss_f32(vsum128);
    }

    for (; i < n; ++i) {
        output[i] = std::exp(input[i] - current_max);
        sum += output[i];
    }

    if (sum == 0.0f) return;

    // 3. Normalize
    float inv_sum = 1.0f / sum;
    __m256 vinv_sum = _mm256_set1_ps(inv_sum);
    i = 0;
    if (n >= 8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(output + i);
            v = _mm256_mul_ps(v, vinv_sum);
            _mm256_storeu_ps(output + i, v);
        }
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

inline float hmax256_ps(__m256 x) {
    __m256 y = _mm256_permute2f128_ps(x, x, 1);
    __m256 m1 = _mm256_max_ps(x, y);
    y = _mm256_shuffle_ps(m1, m1, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 m2 = _mm256_max_ps(m1, y);
    y = _mm256_shuffle_ps(m2, m2, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 m3 = _mm256_max_ps(m2, y);
    return _mm_cvtss_f32(_mm256_castps256_ps128(m3));
}

inline float hsum256_ps(__m256 x) {
    __m256 y = _mm256_permute2f128_ps(x, x, 1);
    __m256 s1 = _mm256_add_ps(x, y);
    y = _mm256_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 s2 = _mm256_add_ps(s1, y);
    y = _mm256_shuffle_ps(s2, s2, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 s3 = _mm256_add_ps(s2, y);
    return _mm_cvtss_f32(_mm256_castps256_ps128(s3));
}

// ⚡ Thunderbolt: AVX2 Vectorized Softmax with 4-way unrolling and tree reduction
// Target: AVX2 (Haswell+)
// Reason: Replaces scalar extraction with in-register tree reduction and unrolls the main loops 4x to hide latency and increase ILP.
// Expected gain: ~25-50% throughput improvement over softmax_v2 due to better FMA latency hiding and fewer scalar/SIMD transitions.
inline void softmax_v3(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    std::size_t i = 0;
    __m256 max0 = _mm256_set1_ps(-INFINITY);
    __m256 max1 = _mm256_set1_ps(-INFINITY);
    __m256 max2 = _mm256_set1_ps(-INFINITY);
    __m256 max3 = _mm256_set1_ps(-INFINITY);

    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }
    max0 = _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));

    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }
    float max_val = hmax256_ps(max0);
    for (; i < n; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    __m256 max_vec = _mm256_set1_ps(max_val);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        __m256 e0 = exp_ps_taylor_avx2(_mm256_sub_ps(x0, max_vec));
        __m256 e1 = exp_ps_taylor_avx2(_mm256_sub_ps(x1, max_vec));
        __m256 e2 = exp_ps_taylor_avx2(_mm256_sub_ps(x2, max_vec));
        __m256 e3 = exp_ps_taylor_avx2(_mm256_sub_ps(x3, max_vec));

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = exp_ps_taylor_avx2(_mm256_sub_ps(x, max_vec));
        _mm256_storeu_ps(output + i, e);
        sum0 = _mm256_add_ps(sum0, e);
    }
    float sum_val = hsum256_ps(sum0);
    for (; i < n; ++i) {
        float e = std::exp(input[i] - max_val);
        output[i] = e;
        sum_val += e;
    }

    if (sum_val == 0.0f) return;

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
