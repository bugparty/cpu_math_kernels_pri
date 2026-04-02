#include <iomanip>
#include <iostream>
#include <vector>

#include "ml_kernels/kernel_common.h"
#include "ml_kernels/naive_ops.h"

int main() {
    const auto &spec = ml_kernels::kSmokeSpec;
    const std::vector<float> input{-2.0f, -0.5f, 1.0f, 3.0f};
    std::vector<float> relu_output(input.size(), 0.0f);
    std::vector<float> softmax_output(input.size(), 0.0f);

    ml_kernels::relu_naive(input.data(), relu_output.data(), input.size());
    ml_kernels::softmax_naive(input.data(), softmax_output.data(), input.size());
    const float max_value = ml_kernels::max_naive(input.data(), input.size());

    std::cout << "ml_kernels workspace ready\n";
    std::cout << "kernel=" << spec.name
              << " tile=(" << spec.tile_m
              << "," << spec.tile_n
              << "," << spec.tile_k << ")\n";

    std::cout << "input: ";
    for (float value : input) {
        std::cout << value << " ";
    }
    std::cout << "\n";

    std::cout << "relu_naive: ";
    for (float value : relu_output) {
        std::cout << value << " ";
    }
    std::cout << "\n";

    std::cout << "max_naive: " << max_value << "\n";

    std::cout << "softmax_naive: ";
    std::cout << std::fixed << std::setprecision(6);
    for (float value : softmax_output) {
        std::cout << value << " ";
    }
    std::cout << "\n";
    return 0;
}
