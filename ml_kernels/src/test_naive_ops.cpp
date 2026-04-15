#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "ml_kernels/naive_ops.h"
#include "ml_kernels/softmax.h"

void test_softmax() {
    std::cout << "Running test_softmax..." << std::endl;

    auto run_test_case = [](const std::vector<float>& input) {
        std::vector<float> expected(input.size());
        std::vector<float> out_v2(input.size(), -1.0f);
        std::vector<float> out_v3(input.size(), -1.0f);

        ml_kernels::softmax_naive(input.data(), expected.data(), input.size());
        ml_kernels::softmax_v2(input.data(), out_v2.data(), input.size());
        ml_kernels::softmax_v3(input.data(), out_v3.data(), input.size());

        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::fabs(out_v2[i] - expected[i]) < 1e-5f);
            assert(std::fabs(out_v3[i] - expected[i]) < 1e-5f);
        }
    };

    // Test 1: Standard
    run_test_case({-1.0f, 0.0f, 1.0f, 2.0f, 3.0f});

    // Test 2: Negative
    run_test_case({-5.0f, -10.0f, -1.0f});

    // Test 3: Large Values
    run_test_case({100.0f, 101.0f, 99.0f});

    // Test 4: Small Size (less than unroll factor)
    run_test_case({1.0f, 2.0f});

    // Test 5: Exact Multiple of 32
    std::vector<float> large_input(64);
    for (int i = 0; i < 64; ++i) large_input[i] = static_cast<float>(i % 10);
    run_test_case(large_input);

    // Test 6: Not a multiple of 32
    std::vector<float> large_input2(70);
    for (int i = 0; i < 70; ++i) large_input2[i] = static_cast<float>(i % 10);
    run_test_case(large_input2);

    std::cout << "test_softmax passed!" << std::endl;
}

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

int main() {
    test_relu_naive();
    test_max_naive();
    test_softmax();
    std::cout << "All tests passed successfully!" << std::endl;
}