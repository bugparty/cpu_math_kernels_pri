#pragma once
//
// CPU Auto-Registration Benchmark Framework
// Inspired by LeetGPUPractise/utility.cuh, adapted for CPU (no CUDA)
//

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

// ---------------------------------------------------------------------------
// BenchmarkBase — each benchmark module implements this interface
// ---------------------------------------------------------------------------
struct BenchmarkBase {
    virtual ~BenchmarkBase() = default;

    // Human-readable name for this benchmark variant
    virtual const char *name() const = 0;

    // Allocate & initialize data for problem size n
    virtual void setup(int n) = 0;

    // Run the kernel (no launch config needed for CPU)
    virtual void run() = 0;

    // Verify correctness. Called after run(). Return true if correct.
    virtual bool verify() = 0;

    // Free memory
    virtual void teardown() = 0;

    // For bandwidth reporting: total bytes read + written per call
    virtual double bytes_accessed(int n) const { return 0.0; }

    // For FLOP/s reporting: total floating-point operations per call
    virtual double flops(int n) const { return 0.0; }

    // Max problem size n this benchmark supports (override to skip slow kernels at large sizes)
    virtual int max_n() const { return INT_MAX; }
};

// ---------------------------------------------------------------------------
// BenchmarkRegistry — global singleton that collects all registered benchmarks
// ---------------------------------------------------------------------------
struct BenchmarkRegistry {
    static BenchmarkRegistry &instance() {
        static BenchmarkRegistry reg;
        return reg;
    }

    void add(BenchmarkBase *bench) { benchmarks_.push_back(bench); }

    const std::vector<BenchmarkBase *> &all() const { return benchmarks_; }

private:
    BenchmarkRegistry() = default;
    std::vector<BenchmarkBase *> benchmarks_;
};

// ---------------------------------------------------------------------------
// REGISTER_BENCHMARK macro — place at file scope in each benchmark module
//
// Usage:
//   struct MyBench : BenchmarkBase { ... };
//   REGISTER_BENCHMARK(MyBench)
// ---------------------------------------------------------------------------
#define REGISTER_BENCHMARK(BenchClass)                                         \
    static BenchClass _bench_instance_##BenchClass;                            \
    static int _bench_reg_##BenchClass = [] {                                  \
        BenchmarkRegistry::instance().add(&_bench_instance_##BenchClass);      \
        return 0;                                                               \
    }()

// ---------------------------------------------------------------------------
// Generic benchmark runner
// ---------------------------------------------------------------------------
inline BenchmarkResult run_benchmark(BenchmarkBase *bench, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i)
        bench->run();

    std::vector<double> timings;
    timings.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        bench->run();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timings.push_back(ms);
    }

    auto mm = std::minmax_element(timings.begin(), timings.end());
    double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    return {avg, *mm.first, *mm.second};
}

// ---------------------------------------------------------------------------
// Formatted table output helpers
// ---------------------------------------------------------------------------
inline void print_table_header() {
    std::cout << std::left << std::setw(20) << "benchmark"
              << std::right
              << std::setw(12) << "avg_ms"
              << std::setw(12) << "min_ms"
              << std::setw(12) << "max_ms"
              << std::setw(14) << "GFLOP/s"
              << std::setw(10) << "verify"
              << "\n"
              << std::string(80, '-') << "\n";
}

inline void print_table_row(const char *name, const BenchmarkResult &r,
                            double gflops, bool ok) {
    std::cout << std::left << std::setw(20) << name
              << std::right << std::fixed
              << std::setprecision(3)
              << std::setw(12) << r.avg_ms
              << std::setw(12) << r.min_ms
              << std::setw(12) << r.max_ms
              << std::setprecision(2)
              << std::setw(14) << gflops
              << std::setw(10) << (ok ? "PASS" : "FAIL")
              << "\n";
}

inline void print_skip_row(const char *name) {
    std::cout << std::left << std::setw(20) << name
              << std::right
              << std::setw(12) << "-"
              << std::setw(12) << "-"
              << std::setw(12) << "-"
              << std::setw(14) << "-"
              << std::setw(10) << "SKIP"
              << "\n";
}
