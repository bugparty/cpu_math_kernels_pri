#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

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

void test_softmax_v2() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f};
    std::vector<float> output_ref(input.size());
    std::vector<float> output_avx(input.size());

    ml_kernels::softmax_naive(input.data(), output_ref.data(), input.size());
    ml_kernels::softmax_v2(input.data(), output_avx.data(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        float diff = std::abs(output_ref[i] - output_avx[i]);
        assert(diff < 1e-4f);
    }

    std::cout << "test_softmax_v2 passed!" << std::endl;
}

int main() {
    test_max_naive();
    test_softmax_v2();
    return 0;
}
