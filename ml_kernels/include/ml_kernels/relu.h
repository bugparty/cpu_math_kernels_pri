#pragma once

#include <cstddef>

#include "compiler_compat.h"
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

        _mm256_storeu_ps(output + i, i0);
        _mm256_storeu_ps(output + i + 8, i1);
        _mm256_storeu_ps(output + i + 16, i2);
        _mm256_storeu_ps(output + i + 24, i3);
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

        _mm256_storeu_ps(output + i, i0);
        _mm256_storeu_ps(output + i + 8, i1);
        _mm256_storeu_ps(output + i + 16, i2);
        _mm256_storeu_ps(output + i + 24, i3);
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

        _mm256_storeu_ps(output + i, i0);
        _mm256_storeu_ps(output + i + 8, i1);
        _mm256_storeu_ps(output + i + 16, i2);
        _mm256_storeu_ps(output + i + 24, i3);
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

        _mm256_storeu_ps(output + i, i0);
        _mm256_storeu_ps(output + i + 8, i1);
        _mm256_storeu_ps(output + i + 16, i2);
        _mm256_storeu_ps(output + i + 24, i3);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

}

// ⚡ Thunderbolt: AVX2 Vectorized ReLU (8x unroll + unaligned stores + masked epilogue)
// Target: AVX2 (Haswell+)
// Reason: Previous versions relied on scalar epilogues which are slow.
// Unrolling 8x maximizes instruction throughput. Used unaligned stores instead of streaming stores to avoid GP fault since caller alignment is not guaranteed.
// Using AVX2 masked loads/stores for the remainder elements completely eliminates the scalar epilogue, maintaining vector throughput until the very end.
// Expected gain: Better throughput on non-multiple-of-64 array sizes and reduced L1 cache pollution.
inline void relu_v4(const float *input, float *output, std::size_t n) {
    // Assert alignment not guaranteed, so we use storeu_ps
    // assert((uintptr_t)output % 32 == 0); // Not assuming caller aligns, so unaligned store used.

    if (n == 0) return;
    std::size_t i = 0;
    constexpr std::size_t kStride = 64;
    const std::size_t groups = n - n % kStride;
    auto const zeros = _mm256_set1_ps(0.0f);

    for (; i < groups; i += kStride) {
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

        _mm256_storeu_ps(output + i, i0);
        _mm256_storeu_ps(output + i + 8, i1);
        _mm256_storeu_ps(output + i + 16, i2);
        _mm256_storeu_ps(output + i + 24, i3);
        _mm256_storeu_ps(output + i + 32, i4);
        _mm256_storeu_ps(output + i + 40, i5);
        _mm256_storeu_ps(output + i + 48, i6);
        _mm256_storeu_ps(output + i + 56, i7);
    }

    // Remaining elements using masked vector operations
    if (i < n) {
        // Remainder loop for 8-element blocks
        for (; i + 7 < n; i += 8) {
            auto i0 = _mm256_loadu_ps(input + i);
            i0 = _mm256_max_ps(i0, zeros);
            // Can't stream unaligned easily, use regular store for remainder
            _mm256_storeu_ps(output + i, i0);
        }

        // Final remainder < 8 elements
        if (i < n) {
            std::size_t rem = n - i;
            // Generate mask for remaining elements
            // E.g. rem = 3 -> mask = 0b00000111
            int mask_int = (1 << rem) - 1;
            __m256i mask = _mm256_setr_epi32(
                (mask_int & 1) ? -1 : 0,
                (mask_int & 2) ? -1 : 0,
                (mask_int & 4) ? -1 : 0,
                (mask_int & 8) ? -1 : 0,
                (mask_int & 16) ? -1 : 0,
                (mask_int & 32) ? -1 : 0,
                (mask_int & 64) ? -1 : 0,
                (mask_int & 128) ? -1 : 0
            );

            auto i0 = _mm256_maskload_ps(input + i, mask);
            i0 = _mm256_max_ps(i0, zeros);
            _mm256_maskstore_ps(output + i, mask, i0);
        }
    }

}

} // namespace ml_kernels
