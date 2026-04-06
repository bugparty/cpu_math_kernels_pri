#include <cassert>
#include <iostream>
#include <vector>

#include "ml_kernels/naive_ops.h"

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

int main() {
    test_max_naive();
    return 0;
}
