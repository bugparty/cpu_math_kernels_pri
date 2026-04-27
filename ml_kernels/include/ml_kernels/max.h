#pragma once

#include <cstddef>
#include <limits>
#include <immintrin.h>
#include <algorithm>

namespace ml_kernels {

// ⚡ Thunderbolt: AVX2 Vectorized Max Reduction
// Target: AVX2 (Haswell+)
// Reason: The naive scalar max reduction (max_naive) is bottlenecked by a loop-carried dependency and low ILP.
// Vectorizing it with AVX2 and unrolling 4x allows 32 elements to be processed per iteration across multiple execution ports.
// The final reduction is done efficiently in-register using shuffles, avoiding a scalar extraction loop.
// Expected gain: ~4-5x throughput vs max_naive.
inline float max_v2(const float *input, std::size_t n) {
    if (n == 0) return 0.0f;

    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;

    // Unroll 4x for 32 elements per iteration
    for (; i + 31 < n; i += 32) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(input + i + 8));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(input + i + 16));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(input + i + 24));
    }

    // Reduce the 4 vectors into 1
    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);

    // Remainder loop for multiples of 8 elements
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }

    // In-register horizontal reduction
    __m128 lo = _mm256_castps256_ps128(max0);
    __m128 hi = _mm256_extractf128_ps(max0, 1);
    lo = _mm_max_ps(lo, hi);

    __m128 shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_max_ps(lo, shuf);
    shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_max_ps(lo, shuf);

    float max_val = _mm_cvtss_f32(lo);

    // Scalar epilogue
    for (; i < n; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    return max_val;
}

} // namespace ml_kernels

// ⚡ Thunderbolt: AVX2 Vectorized Max Reduction (8x unroll)
// Target: AVX2 (Haswell+)
// Reason: `_mm256_max_ps` has a 4-cycle latency and 0.5-cycle throughput on most modern Intel uarchs.
// A 4x unroll issues 4 instructions (2 cycles) but must wait 2 more cycles for the dependency chain to resolve.
// Unrolling 8x maintains 8 independent accumulators, issuing 8 instructions over 4 cycles, perfectly matching
// the instruction latency and fully saturating the execution ports, transitioning from latency-bound to throughput-bound.
// Expected gain: ~1.5x-2.0x throughput over 4x unroll (max_v2) on large arrays.
namespace ml_kernels {
inline float max_v3(const float *input, std::size_t n) {
    if (n == 0) return 0.0f;

    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;
    __m256 max4 = max_v, max5 = max_v, max6 = max_v, max7 = max_v;

    // Unroll 8x for 64 elements per iteration
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

    // Reduce the 8 vectors into 1
    max0 = _mm256_max_ps(max0, max4);
    max1 = _mm256_max_ps(max1, max5);
    max2 = _mm256_max_ps(max2, max6);
    max3 = _mm256_max_ps(max3, max7);

    max0 = _mm256_max_ps(max0, max1);
    max2 = _mm256_max_ps(max2, max3);
    max0 = _mm256_max_ps(max0, max2);

    // Remainder loop for multiples of 8 elements
    for (; i + 7 < n; i += 8) {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(input + i));
    }

    // In-register horizontal reduction
    __m128 lo = _mm256_castps256_ps128(max0);
    __m128 hi = _mm256_extractf128_ps(max0, 1);
    lo = _mm_max_ps(lo, hi);

    __m128 shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_max_ps(lo, shuf);
    shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_max_ps(lo, shuf);

    float max_val = _mm_cvtss_f32(lo);

    // Scalar epilogue
    for (; i < n; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    return max_val;
}
} // namespace ml_kernels
