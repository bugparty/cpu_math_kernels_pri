#pragma once

#include <cstddef>
#include <algorithm>
#include <limits>
#include <immintrin.h>

namespace ml_kernels {

// ⚡ Thunderbolt: AVX2 Vectorized Max Reduction
// Target: AVX2 (Haswell+)
// Reason: Replacing scalar max reduction with a 4-way unrolled AVX2 implementation
// hides the 4-cycle FMA/max latency by breaking the loop-carried dependency chain.
// Expected gain: ~Nx throughput on large arrays, transitioning from latency-bound to memory-bound.
inline float max_v2(const float *input, std::size_t n) {
    if (n == 0) return 0.0f;

    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;

    const float lowest = std::numeric_limits<float>::lowest();
    __m256 max0 = _mm256_set1_ps(lowest);
    __m256 max1 = _mm256_set1_ps(lowest);
    __m256 max2 = _mm256_set1_ps(lowest);
    __m256 max3 = _mm256_set1_ps(lowest);

    for (; i < groups; i += kStride) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);
        __m256 v2 = _mm256_loadu_ps(input + i + 16);
        __m256 v3 = _mm256_loadu_ps(input + i + 24);

        max0 = _mm256_max_ps(max0, v0);
        max1 = _mm256_max_ps(max1, v1);
        max2 = _mm256_max_ps(max2, v2);
        max3 = _mm256_max_ps(max3, v3);
    }

    // Combine unrolled accumulators
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);

    // Remaining elements that form full vectors of 8
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }

    // Horizontal max of the final vector using in-register tree reduction
    __m256 t0 = _mm256_permute2f128_ps(max0, max0, 1);
    max0 = _mm256_max_ps(max0, t0);
    __m256 t1 = _mm256_shuffle_ps(max0, max0, _MM_SHUFFLE(1, 0, 3, 2));
    max0 = _mm256_max_ps(max0, t1);
    __m256 t2 = _mm256_shuffle_ps(max0, max0, _MM_SHUFFLE(2, 3, 0, 1));
    max0 = _mm256_max_ps(max0, t2);

    float max_val = _mm_cvtss_f32(_mm256_castps256_ps128(max0));

    // Scalar fallback for remainder
    for (; i < n; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    return max_val;
}

} // namespace ml_kernels
