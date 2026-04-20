#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "ml_kernels/naive_ops.h"
#include "ml_kernels/naive_ops.h"
#include "ml_kernels/softmax.h"

void test_max_naive() {
    // Happy path
    {
        std::vector<float> input = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
        float result = ml_kernels::max_naive(input.data(), input.size());
        assert(result == 5.0f);
    }

    // Negative values
    {
        std::vector<float> input = {-5.0f, -2.0f, -8.0f};
        float result = ml_kernels::max_naive(input.data(), input.size());
        assert(result == -2.0f);
    }

    // Single element
    {
        std::vector<float> input = {42.0f};
        float result = ml_kernels::max_naive(input.data(), input.size());
        assert(result == 42.0f);
    }

    // Empty array
    {
        float result = ml_kernels::max_naive(nullptr, 0);
        assert(result == 0.0f);
    }

    std::cout << "test_max_naive passed!" << std::endl;
}

void test_relu_naive() {
    std::cout << "Running test_relu_naive..." << std::endl;

    // Test 1: Mixed positive, negative, and zero values
    {
        std::vector<float> input = {-1.0f, 0.0f, 2.5f, -3.14f, 5.0f};
        std::vector<float> expected = {0.0f, 0.0f, 2.5f, 0.0f, 5.0f};
        std::vector<float> output(input.size(), -1.0f); // Initialize with dummy values

        ml_kernels::relu_naive(input.data(), output.data(), input.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(output[i] - expected[i]) < 1e-6f);
        }
    }

    // Test 2: All negative values
    {
        std::vector<float> input = {-1.0f, -0.5f, -100.0f};
        std::vector<float> expected = {0.0f, 0.0f, 0.0f};
        std::vector<float> output(input.size(), -1.0f);

        ml_kernels::relu_naive(input.data(), output.data(), input.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(output[i] - expected[i]) < 1e-6f);
        }
    }

    // Test 3: All positive values
    {
        std::vector<float> input = {1.0f, 0.5f, 100.0f};
        std::vector<float> expected = {1.0f, 0.5f, 100.0f};
        std::vector<float> output(input.size(), -1.0f);

        ml_kernels::relu_naive(input.data(), output.data(), input.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(output[i] - expected[i]) < 1e-6f);
        }
    }

    // Test 4: Empty input
    {
        std::vector<float> input = {};
        std::vector<float> output = {};

        // Should not crash
        ml_kernels::relu_naive(input.data(), output.data(), 0);
    }

    std::cout << "test_relu_naive passed!" << std::endl;
}

void test_softmax_v3() {
    std::cout << "Running test_softmax_v3..." << std::endl;
    std::vector<float> input = {
        -2.0f, -0.5f, 1.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, 100.0f, -100.0f, -100.0f,
        5.0f, -5.0f, 2.0f, -2.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
        -1.0f, -2.0f, -3.0f, -4.0f,
        10.0f, 9.0f, 8.0f, 7.0f,
        -0.1f, -0.2f, -0.3f, -0.4f,
        1.1f, 2.2f, 3.3f, 4.4f,
        -1.1f, -2.2f, -3.3f, -4.4f
    };
    std::vector<float> output_naive(input.size());
    std::vector<float> output_v3(input.size());
    std::vector<float> output_v3_estrin(input.size());
    std::vector<float> output_v2_estrin(input.size());

    ml_kernels::softmax_naive(input.data(), output_naive.data(), input.size());
    ml_kernels::softmax_v3(input.data(), output_v3.data(), input.size());
    ml_kernels::softmax_v3_estrin(input.data(), output_v3_estrin.data(), input.size());
    ml_kernels::softmax_v2_estrin(input.data(), output_v2_estrin.data(), input.size());

    float sum = 0.0f;
    float sum_v3_estrin = 0.0f;
    float sum_v2_estrin = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        assert(std::fabs(output_naive[i] - output_v3[i]) < 1e-4f);
        assert(std::fabs(output_naive[i] - output_v3_estrin[i]) < 1e-4f);
        assert(std::fabs(output_naive[i] - output_v2_estrin[i]) < 1e-4f);
        sum += output_v3[i];
        sum_v3_estrin += output_v3_estrin[i];
        sum_v2_estrin += output_v2_estrin[i];
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);
    assert(std::fabs(sum_v3_estrin - 1.0f) < 1e-4f);
    assert(std::fabs(sum_v2_estrin - 1.0f) < 1e-4f);

    std::cout << "test_softmax_v3 passed!" << std::endl;
}

int main() {
    test_relu_naive();
    test_max_naive();
    test_softmax_v3();
    std::cout << "All tests passed successfully!" << std::endl;
}
