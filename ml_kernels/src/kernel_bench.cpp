#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <cstdlib>

#ifdef __linux__
#include <sched.h>
#endif

#include "aligned_buffer.h"
#include "benchmark.h"
#include "ml_kernels/naive_ops.h"
#include "ml_kernels/relu.h"
#include "ml_kernels/softmax.h"

namespace {

static bool g_use_pool = true;

#ifdef __linux__
void bind_default_benchmark_cpus() {
    const char* disable_binding = std::getenv("DISABLE_CPU_BINDING");
    if (disable_binding && std::string(disable_binding) == "1") {
        std::cout << "CPU binding disabled by DISABLE_CPU_BINDING environment variable." << std::endl;
        return;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(10, &set);
    CPU_SET(11, &set);
    CPU_SET(12, &set);
    CPU_SET(13, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::cerr << "Warning: Failed to bind benchmark to CPUs 10,11,12,13: "
                  << std::strerror(errno) << ". Proceeding without CPU binding." << std::endl;
    } else {
        std::cout << "cpu_affinity=10,11,12,13" << std::endl;
    }
}
#else
void bind_default_benchmark_cpus() {}
#endif

template <const char *Name, void (*Kernel)(const float *, float *, std::size_t)>
class ReLUBenchmarkBase : public BenchmarkBase {
public:
    const char *name() const override { return Name; }

    void setup(int n) override {
        size_t bytes_per_iteration = 2ULL * n * sizeof(float);
        size_t target_pool_bytes = 100ULL * 1024 * 1024;
        pool_size_ = g_use_pool ? std::max<std::size_t>(1, target_pool_bytes / bytes_per_iteration) : 1;

        inputs_.resize(pool_size_);
        outputs_.resize(pool_size_);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (std::size_t i = 0; i < pool_size_; ++i) {
            inputs_[i].resize(n);
            outputs_[i].assign(n, 0.0f);
            for (float &value : inputs_[i]) {
                value = dist(rng);
            }
        }

        output_ref_.resize(n);
        for (int i = 0; i < n; ++i) {
            output_ref_[i] = inputs_[0][i] > 0.0f ? inputs_[0][i] : 0.0f;
        }
        current_idx_ = 0;
    }

    void run() override {
        Kernel(inputs_[current_idx_].data(), outputs_[current_idx_].data(), inputs_[0].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }

    bool verify() override {
        current_idx_ = 0;
        run();
        constexpr float tol = 1e-6f;
        for (std::size_t i = 0; i < outputs_[0].size(); ++i) {
            if (std::fabs(outputs_[0][i] - output_ref_[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    void teardown() override {
        inputs_.clear();
        outputs_.clear();
        output_ref_.clear();
    }

    double bytes_accessed(int n) const override {
        return 2.0 * n * sizeof(float);
    }

    double flops(int n) const override { return static_cast<double>(n); }

protected:
    std::vector<AlignedBuffer<float>> inputs_;
    std::vector<AlignedBuffer<float>> outputs_;
    AlignedBuffer<float> output_ref_;
    std::size_t pool_size_ = 1;
    std::size_t current_idx_ = 0;
};

#define REGISTER_RELU_BENCHMARK(KernelFunc) \
    inline constexpr char k##KernelFunc##Name[] = #KernelFunc; \
    using KernelFunc##Benchmark = ReLUBenchmarkBase<k##KernelFunc##Name, ml_kernels::KernelFunc>; \
    REGISTER_BENCHMARK(KernelFunc##Benchmark)

REGISTER_RELU_BENCHMARK(relu_naive);
REGISTER_RELU_BENCHMARK(relu_v2);
REGISTER_RELU_BENCHMARK(relu_v3);
REGISTER_RELU_BENCHMARK(relu_v2_1);
REGISTER_RELU_BENCHMARK(relu_4block_stream);
REGISTER_RELU_BENCHMARK(relu_4block_stream_unroll);
REGISTER_RELU_BENCHMARK(relu_4block_stream_nofence);
REGISTER_RELU_BENCHMARK(relu_4block_stream_nofence2);
REGISTER_RELU_BENCHMARK(relu_4block_stream_nofence3);
REGISTER_RELU_BENCHMARK(relu_4block_stream_nofence4);

REGISTER_RELU_BENCHMARK(relu_v2_2);
REGISTER_RELU_BENCHMARK(relu_v2_3);
REGISTER_RELU_BENCHMARK(relu_v2_4);
REGISTER_RELU_BENCHMARK(relu_v2_5);
REGISTER_RELU_BENCHMARK(relu_v2_6);
REGISTER_RELU_BENCHMARK(relu_v2_7);
REGISTER_RELU_BENCHMARK(relu_v2_8);

class MaxBenchmark : public BenchmarkBase {
public:
    const char *name() const override { return "max_naive"; }

    void setup(int n) override {
        size_t bytes_per_iteration = n * sizeof(float);
        size_t target_pool_bytes = 100ULL * 1024 * 1024;
        pool_size_ = g_use_pool ? std::max<std::size_t>(1, target_pool_bytes / bytes_per_iteration) : 1;

        inputs_.resize(pool_size_);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (std::size_t i = 0; i < pool_size_; ++i) {
            inputs_[i].resize(n);
            for (float &value : inputs_[i]) {
                value = dist(rng);
            }
        }

        result_ref_ = inputs_[0].size() == 0
            ? 0.0f
            : *std::max_element(inputs_[0].begin(), inputs_[0].end());
        result_ = 0.0f;
        current_idx_ = 0;
    }

    void run() override {
        result_ = ml_kernels::max_naive(inputs_[current_idx_].data(), inputs_[current_idx_].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }

    bool verify() override {
        current_idx_ = 0;
        run();
        return std::fabs(result_ - result_ref_) <= 1e-6f;
    }

    void teardown() override {
        inputs_.clear();
        result_ = 0.0f;
        result_ref_ = 0.0f;
    }

    double bytes_accessed(int n) const override { return n * sizeof(float); }

    double flops(int n) const override { return static_cast<double>(n); }

protected:
    std::vector<AlignedBuffer<float>> inputs_;
    float result_ = 0.0f;
    float result_ref_ = 0.0f;
    std::size_t pool_size_ = 1;
    std::size_t current_idx_ = 0;
};

class SoftmaxBenchmark : public BenchmarkBase {
public:
    const char *name() const override { return "softmax_naive"; }

    void setup(int n) override {
        size_t bytes_per_iteration = 3ULL * n * sizeof(float);
        size_t target_pool_bytes = 100ULL * 1024 * 1024;
        pool_size_ = g_use_pool ? std::max<std::size_t>(1, target_pool_bytes / bytes_per_iteration) : 1;

        inputs_.resize(pool_size_);
        outputs_.resize(pool_size_);
        output_ref_.assign(n, 0.0f);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (std::size_t i = 0; i < pool_size_; ++i) {
            inputs_[i].resize(n);
            outputs_[i].assign(n, 0.0f);
            for (float &value : inputs_[i]) {
                value = dist(rng);
            }
        }

        if (n == 0) {
            return;
        }

        const float max_value = *std::max_element(inputs_[0].begin(), inputs_[0].end());
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            output_ref_[i] = std::exp(inputs_[0][i] - max_value);
            sum += output_ref_[i];
        }
        for (float &value : output_ref_) {
            value /= sum;
        }
        current_idx_ = 0;
    }

    void run() override {
        ml_kernels::softmax_naive(inputs_[current_idx_].data(), outputs_[current_idx_].data(), inputs_[0].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }

    bool verify() override {
        current_idx_ = 0;
        run();
        constexpr float tol = 1e-5f;
        for (std::size_t i = 0; i < outputs_[0].size(); ++i) {
            if (std::fabs(outputs_[0][i] - output_ref_[i]) > tol) {
                return false;
            }
        }
        return true;
    }

    void teardown() override {
        inputs_.clear();
        outputs_.clear();
        output_ref_.clear();
    }

    double bytes_accessed(int n) const override {
        return 2.0 * n * sizeof(float);
    }

    double flops(int n) const override { return 4.0 * n; }

protected:
    std::vector<AlignedBuffer<float>> inputs_;
    std::vector<AlignedBuffer<float>> outputs_;
    AlignedBuffer<float> output_ref_;
    std::size_t pool_size_ = 1;
    std::size_t current_idx_ = 0;
};

REGISTER_BENCHMARK(MaxBenchmark);
REGISTER_BENCHMARK(SoftmaxBenchmark);

class SoftmaxV2Benchmark : public SoftmaxBenchmark {
public:
    const char *name() const override { return "softmax_v2"; }
    void run() override {
        ml_kernels::softmax_v2(inputs_[current_idx_].data(), outputs_[current_idx_].data(), inputs_[0].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }
};
REGISTER_BENCHMARK(SoftmaxV2Benchmark);
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

class SoftmaxV3Benchmark : public SoftmaxBenchmark {
public:
    const char *name() const override { return "softmax_v3"; }

    void run() override {
        ml_kernels::softmax_v3(inputs_[current_idx_].data(), outputs_[current_idx_].data(), inputs_[0].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }
};
REGISTER_BENCHMARK(SoftmaxV3Benchmark);

class SoftmaxV4Benchmark : public SoftmaxBenchmark {
public:
    const char *name() const override { return "softmax_v4"; }

    void run() override {
        ml_kernels::softmax_v4(inputs_[current_idx_].data(), outputs_[current_idx_].data(), inputs_[0].size());
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }
};
REGISTER_BENCHMARK(SoftmaxV4Benchmark);

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
        for (bool use_pool : {true, false}) {
            g_use_pool = use_pool;
            std::cout << "=== N=" << n << (use_pool ? " (Pool Mode)" : " (Fixed Memory)") << " ===" << std::endl;
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
    }

    return 0;
}
