#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "ml_kernels/naive_ops.h"
#include "ml_kernels/max.h"

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

void test_max_v2() {
    // Happy path
    {
        std::vector<float> input = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
        float result = ml_kernels::max_v2(input.data(), input.size());
        assert(result == 5.0f);
    }

    // Negative values
    {
        std::vector<float> input = {-5.0f, -2.0f, -8.0f};
        float result = ml_kernels::max_v2(input.data(), input.size());
        assert(result == -2.0f);
    }

    // Single element
    {
        std::vector<float> input = {42.0f};
        float result = ml_kernels::max_v2(input.data(), input.size());
        assert(result == 42.0f);
    }

    // Empty array
    {
        float result = ml_kernels::max_v2(nullptr, 0);
        assert(result == 0.0f);
    }

    // Array exactly matching AVX stride
    {
        std::vector<float> input(64, 0.0f);
        input[32] = 42.0f;
        float result = ml_kernels::max_v2(input.data(), input.size());
        assert(result == 42.0f);
    }

    // Array larger than stride with odd size
    {
        std::vector<float> input(100, 0.0f);
        input[99] = 100.0f;
        float result = ml_kernels::max_v2(input.data(), input.size());
        assert(result == 100.0f);
    }

    std::cout << "test_max_v2 passed!" << std::endl;
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

int main() {
    test_relu_naive();
    test_max_naive();
    test_max_v2();
    std::cout << "All tests passed successfully!" << std::endl;
}