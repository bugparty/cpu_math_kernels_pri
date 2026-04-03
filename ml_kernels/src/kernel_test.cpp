#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>

#include "ml_kernels/naive_ops.h"

void test_softmax_naive_basic() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    ml_kernels::softmax_naive(input.data(), output.data(), input.size());

    float max_val = 3.0f;
    float sum_exp = std::exp(1.0f - max_val) + std::exp(2.0f - max_val) + std::exp(3.0f - max_val);

    float expected_0 = std::exp(1.0f - max_val) / sum_exp;
    float expected_1 = std::exp(2.0f - max_val) / sum_exp;
    float expected_2 = std::exp(3.0f - max_val) / sum_exp;

    float sum = 0.0f;
    for (float val : output) {
        sum += val;
    }

    assert(std::abs(sum - 1.0f) < 1e-5f);
    assert(std::abs(output[0] - expected_0) < 1e-5f);
    assert(std::abs(output[1] - expected_1) < 1e-5f);
    assert(std::abs(output[2] - expected_2) < 1e-5f);
}

void test_softmax_naive_empty() {
    std::vector<float> input;
    std::vector<float> output;

    // Should not crash
    ml_kernels::softmax_naive(input.data(), output.data(), 0);
}

void test_softmax_naive_negative() {
    std::vector<float> input = {-1000.0f, -1000.0f, -1000.0f};
    std::vector<float> output(3, 0.0f);

    ml_kernels::softmax_naive(input.data(), output.data(), input.size());

    float sum = 0.0f;
    for (float val : output) {
        sum += val;
    }

    // Since they are equal, each should be 1/3
    assert(std::abs(sum - 1.0f) < 1e-5f);
    assert(std::abs(output[0] - (1.0f / 3.0f)) < 1e-5f);
    assert(std::abs(output[1] - (1.0f / 3.0f)) < 1e-5f);
    assert(std::abs(output[2] - (1.0f / 3.0f)) < 1e-5f);
}

int main() {
    test_softmax_naive_basic();
    test_softmax_naive_empty();
    test_softmax_naive_negative();
    std::cout << "All softmax_naive tests passed!" << std::endl;
    return 0;
}
