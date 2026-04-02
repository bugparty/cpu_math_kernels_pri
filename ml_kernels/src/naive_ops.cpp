#include "ml_kernels/naive_ops.h"

#include <cmath>

namespace ml_kernels {

void relu_naive(const float *input, float *output, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

float max_naive(const float *input, std::size_t n) {
    if (n == 0) {
        return 0.0f;
    }

    float current_max = input[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (input[i] > current_max) {
            current_max = input[i];
        }
    }
    return current_max;
}

void softmax_naive(const float *input, float *output, std::size_t n) {
    if (n == 0) {
        return;
    }

    const float max_value = max_naive(input, n);

    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }

    if (sum == 0.0f) {
        return;
    }

    for (std::size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

} // namespace ml_kernels
