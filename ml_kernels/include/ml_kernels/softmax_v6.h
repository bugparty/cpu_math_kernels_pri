// ⚡ Thunderbolt: AVX2 Vectorized Softmax with 8x Unrolling
// Target: AVX2 (Haswell+)
// Reason: Aggressive 8x unrolling across all map-reduce phases perfectly hides instruction latency
// and fully utilizes all 16 YMM registers, transitioning from latency-bound to throughput-bound.
// Expected gain: ~5% throughput over softmax_v5 for large arrays.
inline void softmax_v6(const float *input, float *output, std::size_t n) {
    if (n == 0) return;

    // 1. Find max
    std::size_t i = 0;
    __m256 max_v = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    __m256 max0 = max_v, max1 = max_v, max2 = max_v, max3 = max_v;
    __m256 max4 = max_v, max5 = max_v, max6 = max_v, max7 = max_v;

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
    max0 = _mm256_max_ps(max0, max4);
    max1 = _mm256_max_ps(max1, max5);
    max2 = _mm256_max_ps(max2, max6);
    max3 = _mm256_max_ps(max3, max7);

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
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    for (; i + 63 < n; i += 64) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(input + i), max_vec);
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 8), max_vec);
        __m256 x2 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 16), max_vec);
        __m256 x3 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 24), max_vec);
        __m256 x4 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 32), max_vec);
        __m256 x5 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 40), max_vec);
        __m256 x6 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 48), max_vec);
        __m256 x7 = _mm256_sub_ps(_mm256_loadu_ps(input + i + 56), max_vec);

        __m256 e0 = exp256_ps_v2(x0);
        __m256 e1 = exp256_ps_v2(x1);
        __m256 e2 = exp256_ps_v2(x2);
        __m256 e3 = exp256_ps_v2(x3);
        __m256 e4 = exp256_ps_v2(x4);
        __m256 e5 = exp256_ps_v2(x5);
        __m256 e6 = exp256_ps_v2(x6);
        __m256 e7 = exp256_ps_v2(x7);

        _mm256_storeu_ps(output + i, e0);
        _mm256_storeu_ps(output + i + 8, e1);
        _mm256_storeu_ps(output + i + 16, e2);
        _mm256_storeu_ps(output + i + 24, e3);
        _mm256_storeu_ps(output + i + 32, e4);
        _mm256_storeu_ps(output + i + 40, e5);
        _mm256_storeu_ps(output + i + 48, e6);
        _mm256_storeu_ps(output + i + 56, e7);

        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
        sum4 = _mm256_add_ps(sum4, e4);
        sum5 = _mm256_add_ps(sum5, e5);
        sum6 = _mm256_add_ps(sum6, e6);
        sum7 = _mm256_add_ps(sum7, e7);
    }
    sum0 = _mm256_add_ps(sum0, sum4);
    sum1 = _mm256_add_ps(sum1, sum5);
    sum2 = _mm256_add_ps(sum2, sum6);
    sum3 = _mm256_add_ps(sum3, sum7);

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
    for (; i + 63 < n; i += 64) {
        __m256 o0 = _mm256_loadu_ps(output + i);
        __m256 o1 = _mm256_loadu_ps(output + i + 8);
        __m256 o2 = _mm256_loadu_ps(output + i + 16);
        __m256 o3 = _mm256_loadu_ps(output + i + 24);
        __m256 o4 = _mm256_loadu_ps(output + i + 32);
        __m256 o5 = _mm256_loadu_ps(output + i + 40);
        __m256 o6 = _mm256_loadu_ps(output + i + 48);
        __m256 o7 = _mm256_loadu_ps(output + i + 56);

        __m256 m0 = _mm256_mul_ps(o0, inv_sum_v);
        __m256 m1 = _mm256_mul_ps(o1, inv_sum_v);
        __m256 m2 = _mm256_mul_ps(o2, inv_sum_v);
        __m256 m3 = _mm256_mul_ps(o3, inv_sum_v);
        __m256 m4 = _mm256_mul_ps(o4, inv_sum_v);
        __m256 m5 = _mm256_mul_ps(o5, inv_sum_v);
        __m256 m6 = _mm256_mul_ps(o6, inv_sum_v);
        __m256 m7 = _mm256_mul_ps(o7, inv_sum_v);

        _mm256_storeu_ps(output + i, m0);
        _mm256_storeu_ps(output + i + 8, m1);
        _mm256_storeu_ps(output + i + 16, m2);
        _mm256_storeu_ps(output + i + 24, m3);
        _mm256_storeu_ps(output + i + 32, m4);
        _mm256_storeu_ps(output + i + 40, m5);
        _mm256_storeu_ps(output + i + 48, m6);
        _mm256_storeu_ps(output + i + 56, m7);
    }
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), inv_sum_v));
    }
    for (; i < n; ++i) {
        output[i] *= inv_sum;
    }
}
