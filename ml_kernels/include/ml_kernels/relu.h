#pragma once

#include <cstddef>

#include "compiler_compat.h"
#include "immintrin.h"
namespace ml_kernels {

inline void relu_v2(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        auto i0 = _mm256_loadu_ps(input+i);
        auto i1 = _mm256_loadu_ps(input+i+8);
        auto i2 = _mm256_loadu_ps(input+i+16);
        auto i3 = _mm256_loadu_ps(input+i+24);

        i0 = _mm256_max_ps(i0, zeros);
        i1 = _mm256_max_ps(i1, zeros);
        i2 = _mm256_max_ps(i2, zeros);
        i3 = _mm256_max_ps(i3, zeros);

        _mm256_storeu_ps(output+i, i0);
        _mm256_storeu_ps(output+i+8,i1);
        _mm256_storeu_ps(output+i+16,i2);
        _mm256_storeu_ps(output+i+24,i3);
    }

    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

inline void relu_v3(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 64;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        auto i0 = _mm256_loadu_ps(input+i);
        auto i1 = _mm256_loadu_ps(input+i+8);
        auto i2 = _mm256_loadu_ps(input+i+16);
        auto i3 = _mm256_loadu_ps(input+i+24);
        auto i4 = _mm256_loadu_ps(input+i+32);
        auto i5 = _mm256_loadu_ps(input+i+40);
        auto i6 = _mm256_loadu_ps(input+i+48);
        auto i7 = _mm256_loadu_ps(input+i+56);

        i0 = _mm256_max_ps(i0, zeros);
        i1 = _mm256_max_ps(i1, zeros);
        i2 = _mm256_max_ps(i2, zeros);
        i3 = _mm256_max_ps(i3, zeros);
        i4 = _mm256_max_ps(i4, zeros);
        i5 = _mm256_max_ps(i5, zeros);
        i6 = _mm256_max_ps(i6, zeros);
        i7 = _mm256_max_ps(i7, zeros);

        _mm256_storeu_ps(output+i, i0);
        _mm256_storeu_ps(output+i+8,i1);
        _mm256_storeu_ps(output+i+16,i2);
        _mm256_storeu_ps(output+i+24,i3);
        _mm256_storeu_ps(output+i+32,i4);
        _mm256_storeu_ps(output+i+40,i5);
        _mm256_storeu_ps(output+i+48,i6);
        _mm256_storeu_ps(output+i+56,i7);
    }

    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

inline void relu_v2_1(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {

        auto i0 = _mm256_loadu_ps(input+i);
        auto i1 = _mm256_loadu_ps(input+i+8);
        auto i2 = _mm256_loadu_ps(input+i+16);
        auto i3 = _mm256_loadu_ps(input+i+24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_storeu_ps(output+i, i0);
        _mm256_storeu_ps(output+i+8,i1);
        _mm256_storeu_ps(output+i+16,i2);
        _mm256_storeu_ps(output+i+24,i3);
    }

    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

inline void relu_4block_stream(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {

        auto i0 = _mm256_loadu_ps(input+i);
        auto i1 = _mm256_loadu_ps(input+i+8);
        auto i2 = _mm256_loadu_ps(input+i+16);
        auto i3 = _mm256_loadu_ps(input+i+24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_stream_ps(output+i, i0);
        _mm256_stream_ps(output+i+8,i1);
        _mm256_stream_ps(output+i+16,i2);
        _mm256_stream_ps(output+i+24,i3);
    }
    _mm_sfence();
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}
} // namespace ml_kernels
