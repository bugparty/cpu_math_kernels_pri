#pragma once

#include <cstddef>

#include "ml_kernels/kernel_common.h"
#include "immintrin.h"
#include "xmmintrin.h"
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

inline void relu_v2_2(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+24), _MM_HINT_T0);

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

inline void relu_v2_3(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+24), _MM_HINT_T0);

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

inline void relu_v2_4(const float *input, float *output, std::size_t n) {
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

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));
        i4 = _mm256_max_ps(i4, _mm256_set1_ps(0.0f));
        i5 = _mm256_max_ps(i5, _mm256_set1_ps(0.0f));
        i6 = _mm256_max_ps(i6, _mm256_set1_ps(0.0f));
        i7 = _mm256_max_ps(i7, _mm256_set1_ps(0.0f));

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

inline void relu_v2_5(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_T0);

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
inline void relu_v2_6(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T1);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_T1);

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
inline void relu_v2_7(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_T2);

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
inline void relu_v2_8(const float *input, float *output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(input+i+16), _MM_HINT_NTA);

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
inline void relu_4block_stream_unroll(const float* input, float* output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
#pragma unroll(2)
    for (; i < groups; i += kStride) {

        auto i0 = _mm256_loadu_ps(input + i);
        auto i1 = _mm256_loadu_ps(input + i + 8);
        auto i2 = _mm256_loadu_ps(input + i + 16);
        auto i3 = _mm256_loadu_ps(input + i + 24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_stream_ps(output + i, i0);
        _mm256_stream_ps(output + i + 8, i1);
        _mm256_stream_ps(output + i + 16, i2);
        _mm256_stream_ps(output + i + 24, i3);
    }
    _mm_sfence();
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}
inline void relu_4block_stream_nofence(const float *input, float *output, std::size_t n) {
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
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}
inline void relu_4block_stream_nofence2(const float* input, float* output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 16), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char*>(input + i + 32), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char*>(input + i + 48), _MM_HINT_T0);
        auto i0 = _mm256_loadu_ps(input + i);
        auto i1 = _mm256_loadu_ps(input + i + 8);
        auto i2 = _mm256_loadu_ps(input + i + 16);
        auto i3 = _mm256_loadu_ps(input + i + 24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_stream_ps(output + i, i0);
        _mm256_stream_ps(output + i + 8, i1);
        _mm256_stream_ps(output + i + 16, i2);
        _mm256_stream_ps(output + i + 24, i3);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}
inline void relu_4block_stream_nofence3(const float* input, float* output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 48), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 80), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 96), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char*>(input + i + 112), _MM_HINT_T0);
        auto i0 = _mm256_loadu_ps(input + i);
        auto i1 = _mm256_loadu_ps(input + i + 8);
        auto i2 = _mm256_loadu_ps(input + i + 16);
        auto i3 = _mm256_loadu_ps(input + i + 24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_stream_ps(output + i, i0);
        _mm256_stream_ps(output + i + 8, i1);
        _mm256_stream_ps(output + i + 16, i2);
        _mm256_stream_ps(output + i + 24, i3);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}

inline void relu_4block_stream_nofence4(const float* input, float* output, std::size_t n) {
    std::size_t i = 0;
    constexpr std::size_t kStride = 32;
    const std::size_t groups = n - n % kStride;
    auto const  zeros = _mm256_set1_ps(0.0f);
    for (; i < groups; i += kStride) {
        _mm_prefetch(reinterpret_cast<const char*>(input), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 16), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 32), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 48), _MM_HINT_NTA);
        auto i0 = _mm256_loadu_ps(input + i);
        auto i1 = _mm256_loadu_ps(input + i + 8);
        auto i2 = _mm256_loadu_ps(input + i + 16);
        auto i3 = _mm256_loadu_ps(input + i + 24);

        i0 = _mm256_max_ps(i0, _mm256_set1_ps(0.0f));
        i1 = _mm256_max_ps(i1, _mm256_set1_ps(0.0f));
        i2 = _mm256_max_ps(i2, _mm256_set1_ps(0.0f));
        i3 = _mm256_max_ps(i3, _mm256_set1_ps(0.0f));

        _mm256_stream_ps(output + i, i0);
        _mm256_stream_ps(output + i + 8, i1);
        _mm256_stream_ps(output + i + 16, i2);
        _mm256_stream_ps(output + i + 24, i3);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}

// ⚡ Thunderbolt: AVX2 Vectorized ReLU with 8x Unrolling and Streaming Stores
// Target: AVX2 (Haswell+)
// Reason: ReLU is a pure memory-bound kernel (read 1 float, max, write 1 float). For large vectors
// that exceed L3 cache, writing data with standard stores incurs a Read-For-Ownership (RFO)
// cache miss penalty. Streaming stores (`_mm256_stream_ps`) bypass the cache and write directly to main memory,
// eliminating this penalty. Unrolling the loop 8x helps maintain multiple independent memory streams
// to saturate the store buffers, maximizing throughput.
// Expected gain: Measurable throughput improvement over 4-block streaming versions on out-of-cache large arrays.
inline void relu_v4(const float* input, float* output, std::size_t n) {
    std::size_t i = 0;

    // Prologue: process scalar elements until output pointer is 32-byte aligned
    while (i < n && reinterpret_cast<uintptr_t>(output + i) % 32 != 0) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        ++i;
    }

    constexpr std::size_t kStride = 64;
    const std::size_t remaining = n - i;
    const std::size_t groups = remaining - remaining % kStride;
    const std::size_t limit = i + groups;

    auto const zeros = _mm256_setzero_ps();

    for (; i < limit; i += kStride) {
        auto i0 = _mm256_loadu_ps(input + i);
        auto i1 = _mm256_loadu_ps(input + i + 8);
        auto i2 = _mm256_loadu_ps(input + i + 16);
        auto i3 = _mm256_loadu_ps(input + i + 24);
        auto i4 = _mm256_loadu_ps(input + i + 32);
        auto i5 = _mm256_loadu_ps(input + i + 40);
        auto i6 = _mm256_loadu_ps(input + i + 48);
        auto i7 = _mm256_loadu_ps(input + i + 56);

        i0 = _mm256_max_ps(i0, zeros);
        i1 = _mm256_max_ps(i1, zeros);
        i2 = _mm256_max_ps(i2, zeros);
        i3 = _mm256_max_ps(i3, zeros);
        i4 = _mm256_max_ps(i4, zeros);
        i5 = _mm256_max_ps(i5, zeros);
        i6 = _mm256_max_ps(i6, zeros);
        i7 = _mm256_max_ps(i7, zeros);

        _mm256_stream_ps(output + i, i0);
        _mm256_stream_ps(output + i + 8, i1);
        _mm256_stream_ps(output + i + 16, i2);
        _mm256_stream_ps(output + i + 24, i3);
        _mm256_stream_ps(output + i + 32, i4);
        _mm256_stream_ps(output + i + 40, i5);
        _mm256_stream_ps(output + i + 48, i6);
        _mm256_stream_ps(output + i + 56, i7);
    }
    _mm_sfence(); // Ensure non-temporal stores are visible
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
    _mm256_zeroupper(); // Clean up YMM state
}



} // namespace ml_kernels
