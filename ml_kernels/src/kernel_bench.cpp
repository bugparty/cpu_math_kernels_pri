#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ml_kernels/naive_ops.h"

namespace {

using Clock = std::chrono::steady_clock;

double bench_relu(const std::vector<float> &input, std::vector<float> &output, int iters) {
    const auto start = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        ml_kernels::relu_naive(input.data(), output.data(), input.size());
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

double bench_softmax(const std::vector<float> &input, std::vector<float> &output, int iters) {
    const auto start = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        ml_kernels::softmax_naive(input.data(), output.data(), input.size());
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

double bench_max(const std::vector<float> &input, int iters, float &sink) {
    const auto start = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        sink = ml_kernels::max_naive(input.data(), input.size());
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

} // namespace

int main(int argc, char **argv) {
    std::string op = "all";
    std::size_t size = 1 << 20;
    int iters = 200;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--op" && i + 1 < argc) {
            op = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            size = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::max(1, std::atoi(argv[++i]));
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--op all|relu|max|softmax] [--size N] [--iters N]\n";
            return 1;
        }
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<float> input(size);
    std::vector<float> output(size, 0.0f);
    for (float &value : input) {
        value = dist(rng);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "size=" << size << " iters=" << iters << "\n";

    if (op == "all" || op == "relu") {
        const double avg_ms = bench_relu(input, output, iters);
        std::cout << "relu_naive avg_ms=" << avg_ms << "\n";
    }

    if (op == "all" || op == "max") {
        float sink = 0.0f;
        const double avg_ms = bench_max(input, iters, sink);
        std::cout << "max_naive avg_ms=" << avg_ms << " result=" << sink << "\n";
    }

    if (op == "all" || op == "softmax") {
        const double avg_ms = bench_softmax(input, output, iters);
        std::cout << "softmax_naive avg_ms=" << avg_ms << " first=" << output.front() << "\n";
    }

    return 0;
}
