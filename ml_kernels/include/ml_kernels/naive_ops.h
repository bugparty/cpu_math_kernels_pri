#pragma once

#include <cstddef>

namespace ml_kernels {

void relu_naive(const float *input, float *output, std::size_t n);

float max_naive(const float *input, std::size_t n);

void softmax_naive(const float *input, float *output, std::size_t n);

} // namespace ml_kernels
