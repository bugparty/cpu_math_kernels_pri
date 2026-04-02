#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#ifdef __linux__
#include <sched.h>
#endif

#include "aligned_buffer.h"
#include "benchmark.h"
#include "ml_kernels/naive_ops.h"
#include "ml_kernels/relu.h"

namespace {

#ifdef __linux__
void bind_default_benchmark_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(10, &set);
    CPU_SET(11, &set);
    CPU_SET(12, &set);
    CPU_SET(13, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::cerr << "Failed to bind benchmark to CPUs 10,11,12,13: "
                  << std::strerror(errno) << std::endl;
        std::exit(2);
    }

    std::cout << "cpu_affinity=10,11,12,13" << std::endl;
}
#else
void bind_default_benchmark_cpus() {}
#endif

template <const char *Name, void (*Kernel)(const float *, float *, std::size_t)>
class ReLUBenchmarkBase : public BenchmarkBase {
public:
    const char *name() const override { return Name; }

    void setup(int n) override {
        input_.resize(n);
        output_.assign(n, 0.0f);
        output_ref_.resize(n);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (float &value : input_) {
            value = dist(rng);
        }
        for (int i = 0; i < n; ++i) {
            output_ref_[i] = input_[i] > 0.0f ? input_[i] : 0.0f;
        }
    }

    void run() override {
        Kernel(input_.data(), output_.data(), input_.size());
    }

    bool verify() override {
        run();
        constexpr float tol = 1e-6f;
        for (std::size_t i = 0; i < output_.size(); ++i) {
            if (std::fabs(output_[i] - output_ref_[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    void teardown() override {
        input_.clear();
        output_.clear();
        output_ref_.clear();
    }

    double bytes_accessed(int n) const override {
        return 2.0 * n * sizeof(float);
    }

    double flops(int n) const override { return static_cast<double>(n); }

private:
    AlignedBuffer<float> input_;
    AlignedBuffer<float> output_;
    AlignedBuffer<float> output_ref_;
};

inline constexpr char kReLUNaiveName[] = "relu_naive";
inline constexpr char kReLUV2Name[] = "relu_v2";
inline constexpr char kReLUV3Name[] = "relu_v3";
inline constexpr char kReLUV2_1Name[] = "relu_v2_1";
inline constexpr char kReLU4BlockStreamName[] = "relu_4block_stream";
inline constexpr char kReLU4BlockStreamNofenceName[] = "relu_4block_stream_nofence";
inline constexpr char kReLUV2_2Name[] = "relu_v2_2";
inline constexpr char kReLUV2_3Name[] = "relu_v2_3";
inline constexpr char kReLUV2_4Name[] = "relu_v2_4";
inline constexpr char kReLUV2_5Name[] = "relu_v2_5";
inline constexpr char kReLUV2_6Name[] = "relu_v2_6";
inline constexpr char kReLUV2_7Name[] = "relu_v2_7";
inline constexpr char kReLUV2_8Name[] = "relu_v2_8";

using ReLUBenchmark = ReLUBenchmarkBase<kReLUNaiveName, ml_kernels::relu_naive>;
using ReLUV2Benchmark = ReLUBenchmarkBase<kReLUV2Name, ml_kernels::relu_v2>;
using ReLUV3Benchmark = ReLUBenchmarkBase<kReLUV3Name, ml_kernels::relu_v3>;
using ReLUV2_1Benchmark = ReLUBenchmarkBase<kReLUV2_1Name, ml_kernels::relu_v2_1>;
using ReLU4BlockStreamBenchmark = ReLUBenchmarkBase<kReLU4BlockStreamName, ml_kernels::relu_4block_stream>;
using ReLU4BlockStreamNofenceBenchmark = ReLUBenchmarkBase<kReLU4BlockStreamNofenceName, ml_kernels::relu_4block_stream_nofence>;
using ReLUV2_2Benchmark = ReLUBenchmarkBase<kReLUV2_2Name, ml_kernels::relu_v2_2>;
using ReLUV2_3Benchmark = ReLUBenchmarkBase<kReLUV2_3Name, ml_kernels::relu_v2_3>;
using ReLUV2_4Benchmark = ReLUBenchmarkBase<kReLUV2_4Name, ml_kernels::relu_v2_4>;
using ReLUV2_5Benchmark = ReLUBenchmarkBase<kReLUV2_5Name, ml_kernels::relu_v2_5>;
using ReLUV2_6Benchmark = ReLUBenchmarkBase<kReLUV2_6Name, ml_kernels::relu_v2_6>;
using ReLUV2_7Benchmark = ReLUBenchmarkBase<kReLUV2_7Name, ml_kernels::relu_v2_7>;
using ReLUV2_8Benchmark = ReLUBenchmarkBase<kReLUV2_8Name, ml_kernels::relu_v2_8>;

class MaxBenchmark : public BenchmarkBase {
public:
    const char *name() const override { return "max_naive"; }

    void setup(int n) override {
        input_.resize(n);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (float &value : input_) {
            value = dist(rng);
        }
        result_ref_ = input_.size() == 0
            ? 0.0f
            : *std::max_element(input_.begin(), input_.end());
        result_ = 0.0f;
    }

    void run() override {
        result_ = ml_kernels::max_naive(input_.data(), input_.size());
    }

    bool verify() override {
        run();
        return std::fabs(result_ - result_ref_) <= 1e-6f;
    }

    void teardown() override {
        input_.clear();
        result_ = 0.0f;
        result_ref_ = 0.0f;
    }

    double bytes_accessed(int n) const override { return n * sizeof(float); }

    double flops(int n) const override { return static_cast<double>(n); }

private:
    AlignedBuffer<float> input_;
    float result_ = 0.0f;
    float result_ref_ = 0.0f;
};

class SoftmaxBenchmark : public BenchmarkBase {
public:
    const char *name() const override { return "softmax_naive"; }

    void setup(int n) override {
        input_.resize(n);
        output_.assign(n, 0.0f);
        output_ref_.assign(n, 0.0f);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (float &value : input_) {
            value = dist(rng);
        }

        if (input_.size() == 0) {
            return;
        }

        const float max_value = *std::max_element(input_.begin(), input_.end());
        float sum = 0.0f;
        for (std::size_t i = 0; i < input_.size(); ++i) {
            output_ref_[i] = std::exp(input_[i] - max_value);
            sum += output_ref_[i];
        }
        for (float &value : output_ref_) {
            value /= sum;
        }
    }

    void run() override {
        ml_kernels::softmax_naive(input_.data(), output_.data(), input_.size());
    }

    bool verify() override {
        run();
        constexpr float tol = 1e-5f;
        for (std::size_t i = 0; i < output_.size(); ++i) {
            if (std::fabs(output_[i] - output_ref_[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    void teardown() override {
        input_.clear();
        output_.clear();
        output_ref_.clear();
    }

    double bytes_accessed(int n) const override {
        return 2.0 * n * sizeof(float);
    }

    double flops(int n) const override { return 4.0 * n; }

private:
    AlignedBuffer<float> input_;
    AlignedBuffer<float> output_;
    AlignedBuffer<float> output_ref_;
};

REGISTER_BENCHMARK(ReLUBenchmark);
REGISTER_BENCHMARK(ReLUV2Benchmark);
REGISTER_BENCHMARK(ReLUV3Benchmark);
REGISTER_BENCHMARK(ReLUV2_1Benchmark);
REGISTER_BENCHMARK(ReLU4BlockStreamBenchmark);
REGISTER_BENCHMARK(ReLU4BlockStreamNofenceBenchmark);
REGISTER_BENCHMARK(ReLUV2_2Benchmark);
REGISTER_BENCHMARK(ReLUV2_3Benchmark);
REGISTER_BENCHMARK(ReLUV2_4Benchmark);
REGISTER_BENCHMARK(ReLUV2_5Benchmark);
REGISTER_BENCHMARK(ReLUV2_6Benchmark);
REGISTER_BENCHMARK(ReLUV2_7Benchmark);
REGISTER_BENCHMARK(ReLUV2_8Benchmark);
    // REGISTER_BENCHMARK(MaxBenchmark);
    // REGISTER_BENCHMARK(SoftmaxBenchmark);
std::vector<int> parse_sizes(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int value = std::atoi(tok.c_str());
        if (value > 0) {
            out.push_back(value);
        }
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::string filter;
    std::string sizes_str = "16384,65536,262144,1048576";
    int iters = 20000;
    int warmup = 20;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--filter" || arg == "-f") && i + 1 < argc) {
            filter = argv[++i];
        } else if ((arg == "--sizes" || arg == "-s") && i + 1 < argc) {
            sizes_str = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::max(0, std::atoi(argv[++i]));
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--filter NAME] [--sizes 1024,4096,...]"
                      << " [--iters N] [--warmup N]" << std::endl;
            return 1;
        }
    }

    bind_default_benchmark_cpus();

    const std::vector<int> sizes = parse_sizes(sizes_str);
    if (sizes.empty()) {
        std::cerr << "No valid sizes specified." << std::endl;
        return 1;
    }

    std::cout << "iters=" << iters
              << ", warmup=" << warmup
              << ", sizes=" << sizes_str;
    if (!filter.empty()) {
        std::cout << ", filter=" << filter;
    }
    std::cout << std::endl << std::endl;

    for (int n : sizes) {
        std::cout << "=== N=" << n << " ===" << std::endl;
        print_table_header();

        for (auto *bench : BenchmarkRegistry::instance().all()) {
            if (!filter.empty() && filter != bench->name()) {
                continue;
            }
            if (n > bench->max_n()) {
                print_skip_row(bench->name());
                continue;
            }

            bench->setup(n);
            const BenchmarkResult result = run_benchmark(bench, warmup, iters);
            const bool ok = bench->verify();
            const double gflops = result.avg_ms > 0.0
                ? bench->flops(n) / (result.avg_ms / 1000.0) / 1e9
                : 0.0;
            print_table_row(bench->name(), result, gflops, ok);
            bench->teardown();
        }
        std::cout << std::endl;
    }

    return 0;
}
