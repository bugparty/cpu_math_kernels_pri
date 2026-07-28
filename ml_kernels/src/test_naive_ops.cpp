#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "ml_kernels/naive_ops.h"
#include "ml_kernels/relu.h"
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


void test_relu_8block_stream_unroll() {
    std::cout << "Running test_relu_8block_stream_unroll..." << std::endl;

    // We need at least 72 elements to trigger both the 64-element main loop and the 8-element remainder loop
    std::vector<float> input = {
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        -1.0f, 0.0f, 2.5f, -3.14f, 5.0f, -1.0f, 0.0f, 2.5f, -3.14f, 5.0f,
        1.0f, 1.0f
    };

    std::vector<float> expected(input.size());
    ml_kernels::relu_naive(input.data(), expected.data(), input.size());

    // Ensure memory is aligned for streaming stores if required (though stream_ps handles unaligned well, better safe than sorry, but std::vector alignment is often good enough for our tests, we will just allocate a bit larger and align manually or just use standard vector)
    // Actually, `_mm256_stream_ps` requires 32-byte alignment. std::vector is usually 16 or 32 aligned, but to be strictly safe, let's just use `posix_memalign` if we can, or just try vector. The previous tests don't use aligned_alloc for tests. Let's see if we can just align a buffer.

    float* aligned_out;
    if (posix_memalign((void**)&aligned_out, 32, input.size() * sizeof(float)) != 0) return;
    float* aligned_in;
    if (posix_memalign((void**)&aligned_in, 32, input.size() * sizeof(float)) != 0) { free(aligned_out); return; }
    for(size_t i=0; i<input.size(); ++i) aligned_in[i] = input[i];

    ml_kernels::relu_8block_stream_unroll(aligned_in, aligned_out, input.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::fabs(aligned_out[i] - expected[i]) < 1e-6f);
    }

    free(aligned_out);
    free(aligned_in);

    std::cout << "test_relu_8block_stream_unroll passed!" << std::endl;
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

    ml_kernels::softmax_naive(input.data(), output_naive.data(), input.size());
    ml_kernels::softmax_v3(input.data(), output_v3.data(), input.size());

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        assert(std::fabs(output_naive[i] - output_v3[i]) < 1e-4f);
        sum += output_v3[i];
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);

    std::cout << "test_softmax_v3 passed!" << std::endl;
}

void test_softmax_v4() {
    std::cout << "Running test_softmax_v4..." << std::endl;
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
    std::vector<float> output_v4(input.size());

    ml_kernels::softmax_naive(input.data(), output_naive.data(), input.size());
    ml_kernels::softmax_v4(input.data(), output_v4.data(), input.size());

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        assert(std::fabs(output_naive[i] - output_v4[i]) < 1e-4f);
        sum += output_v4[i];
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);

    std::cout << "test_softmax_v4 passed!" << std::endl;
}

void test_softmax_v5() {
    std::cout << "Running test_softmax_v5..." << std::endl;
    std::vector<float> input = {
        -2.0f, -0.5f, 1.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, 100.0f, -100.0f, -100.0f,
        5.0f, -5.0f, 2.0f, -2.0f,
        1.1f, 1.2f, 1.3f, 1.4f,
        -1.1f, -1.2f, -1.3f, -1.4f,
        10.0f, 20.0f, 30.0f, 40.0f,
        -10.0f, -20.0f, -30.0f, -40.0f
    };

    std::vector<float> output_naive(input.size(), 0.0f);
    std::vector<float> output_v5(input.size(), 0.0f);

    ml_kernels::softmax_naive(input.data(), output_naive.data(), input.size());
    ml_kernels::softmax_v5(input.data(), output_v5.data(), input.size());

    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        assert(std::fabs(output_naive[i] - output_v5[i]) < 1e-4f);
        sum += output_v5[i];
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);

    std::cout << "test_softmax_v5 passed!" << std::endl;
}

int main() {
    test_relu_naive();
    test_relu_8block_stream_unroll();
    test_max_naive();
    test_softmax_v3();
    test_softmax_v4();
    test_softmax_v5();
    std::cout << "All tests passed successfully!" << std::endl;
}