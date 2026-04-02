#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../benchmark.h"
#include "common.h"
#include "transpose_experimental.h"
#include "transpose_avx512.h"
#include "transpose_recursive.h"
#include "transpose_prefetch.h"
#include "transpose_nontemporal.h"

#ifndef GEMM_HAVE_MKL
#define GEMM_HAVE_MKL 0
#endif

#if GEMM_HAVE_MKL
#include <mkl.h>
#endif

// ---------------------------------------------------------------------------
// TransposeBenchmark — wraps a transpose kernel function
// ---------------------------------------------------------------------------
class TransposeBenchmark : public BenchmarkBase {
public:
    using KernelFn = void (*)(const double *, double *, int);

    TransposeBenchmark(const char *name, KernelFn fn, int max_n_limit = INT_MAX)
        : name_(name), kernel_(fn), max_n_limit_(max_n_limit) {}

    const char *name() const override { return name_.c_str(); }
    int max_n() const override { return max_n_limit_; }

    void setup(int n) override {
        n_ = n;
        const int BLOCK_SIZE = 64;
        int n_padded = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
        size_t bytes = (size_t)n_padded * n_padded * sizeof(double);
        
        A_       = static_cast<double *>(_mm_malloc(bytes, 64));
        A_T_     = static_cast<double *>(_mm_malloc(bytes, 64));
        A_T_ref_ = static_cast<double *>(_mm_malloc(bytes, 64));

        // Fill A with random doubles
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        size_t elems = (size_t)n * n;
        for (size_t i = 0; i < elems; ++i) {
            A_[i] = dist(rng);
        }

        // Precompute reference transpose (using naive method)
        for (size_t i = 0; i < (size_t)n; ++i) {
            for (size_t j = 0; j < (size_t)n; ++j) {
                A_T_ref_[j*n + i] = A_[i*n + j];
            }
        }
    }

    void run() override {
        kernel_(A_, A_T_, n_);
    }

    bool verify() override {
        // Compare against reference transpose
        size_t elems = (size_t)n_ * n_;
        constexpr double tol = 1e-10;
        for (size_t i = 0; i < elems; ++i) {
            if (std::abs(A_T_[i] - A_T_ref_[i]) > tol)
                return false;
        }
        return true;
    }

    double flops(int n) const override {
        // Transpose: n*n reads + n*n writes = 2*n*n operations
        return 2.0 * n * n;
    }

    double bytes_accessed(int n) const override {
        // n*n reads (A) + n*n writes (A_T) = 2*n*n doubles
        return 2.0 * (double)n * n * sizeof(double);
    }

    void teardown() override {
        _mm_free(A_);        A_        = nullptr;
        _mm_free(A_T_);      A_T_      = nullptr;
        _mm_free(A_T_ref_);  A_T_ref_  = nullptr;
        n_ = 0;
    }

private:
    std::string name_;
    KernelFn    kernel_;
    int         max_n_limit_;
    int         n_        = 0;
    double     *A_        = nullptr;
    double     *A_T_      = nullptr;
    double     *A_T_ref_  = nullptr;
};

// ---------------------------------------------------------------------------
// Registration macros
// ---------------------------------------------------------------------------
#define REGISTER_TRANSPOSE(namestr, fn)                                        \
    static TransposeBenchmark _tbench_##fn(namestr, fn);                       \
    static int _treg_##fn = [] {                                               \
        BenchmarkRegistry::instance().add(&_tbench_##fn);                      \
        return 0;                                                               \
    }()

#define REGISTER_TRANSPOSE_SLOW(namestr, fn, limit)                            \
    static TransposeBenchmark _tbench_##fn(namestr, fn, limit);                \
    static int _treg_##fn = [] {                                               \
        BenchmarkRegistry::instance().add(&_tbench_##fn);                      \
        return 0;                                                               \
    }()

// ---------------------------------------------------------------------------
// Register all transpose kernels
// ---------------------------------------------------------------------------

// Wrapper functions with correct signature
static void transpose_naive_wrapper(const double *A, double *A_T, int n) {
    transpose_naive(A, A_T, n);
}

static void transpose_tiled_wrapper(const double *A, double *A_T, int n) {
    transpose_tiled(A, A_T, n);
}

#if GEMM_HAVE_MKL
static void transpose_mkl_domatcopy_wrapper(const double *A, double *A_T, int n) {
    mkl_domatcopy('R', 'T', n, n, 1.0, A, n, A_T, n);
}
#endif

// Register benchmarks
// Wrapper for transpose_2level_tuned (template -> function pointer)
static void transpose_2level_tuned_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_128_32_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned<128, 32>(A, AT, n);
}

static void transpose_2level_tuned_256_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned<256, 64>(A, AT, n);
}

static void transpose_2level_tuned_512_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned<512, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx2_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_hinted_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_hinted<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_128_32_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2<128, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_256_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2<256, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx2_512_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2<512, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_128_32_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf<128, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_256_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf<256, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_512_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf<512, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_v2_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf_v2<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_nofence_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf_nofence<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_pf_store_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_pf_store<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx2_nt_pf_ab_tile128_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx2_nt_pf<128, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx512_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx512<256, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx512_128_32_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx512<128, 32>(A, AT, n);
}

static void transpose_2level_tuned_avx512_256_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx512<256, 64>(A, AT, n);
}

static void transpose_2level_tuned_avx512_512_64_wrapper(const double *A, double *AT, int n) {
    transpose_2level_tuned_avx512<512, 64>(A, AT, n);
}

REGISTER_TRANSPOSE_SLOW("transpose_naive", transpose_naive_wrapper, 4096);
REGISTER_TRANSPOSE("transpose_tiled", transpose_tiled_wrapper);
REGISTER_TRANSPOSE("transpose_tiled_v2", transpose_tiled_v2);
REGISTER_TRANSPOSE("transpose_tiled_v3", transpose_tiled_v3);
//REGISTER_TRANSPOSE("transpose_tiled_v4", transpose_tiled_v4);
//REGISTER_TRANSPOSE("transpose_tiled_v5", transpose_tiled_v5);
//REGISTER_TRANSPOSE("transpose_tiled_v6", transpose_tiled_v6);
#if GEMM_HAVE_MKL
REGISTER_TRANSPOSE("mkl_domatcopy", transpose_mkl_domatcopy_wrapper);
#endif
// --- New kernels from optimization team ---
//REGISTER_TRANSPOSE("avx512_8x8",       transpose_avx512);
//REGISTER_TRANSPOSE("recursive",        transpose_recursive);
//REGISTER_TRANSPOSE("prefetch",         transpose_prefetch);
//REGISTER_TRANSPOSE("prefetch_v2",      transpose_prefetch_v2);
//REGISTER_TRANSPOSE("prefetch_v3",      transpose_prefetch_v3);
//REGISTER_TRANSPOSE("nontemporal",      transpose_nontemporal);
REGISTER_TRANSPOSE("2level_256_32",    transpose_2level);
REGISTER_TRANSPOSE("2level_tuned",     transpose_2level_tuned_wrapper);
REGISTER_TRANSPOSE("2level_tuned_hinted", transpose_2level_tuned_hinted_wrapper);
REGISTER_TRANSPOSE("2level_tuned_128_32", transpose_2level_tuned_128_32_wrapper);
REGISTER_TRANSPOSE("2level_tuned_256_64", transpose_2level_tuned_256_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_512_64", transpose_2level_tuned_512_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2", transpose_2level_tuned_avx2_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_128_32", transpose_2level_tuned_avx2_128_32_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_256_64", transpose_2level_tuned_avx2_256_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_512_64", transpose_2level_tuned_avx2_512_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf", transpose_2level_tuned_avx2_nt_pf_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_128_32", transpose_2level_tuned_avx2_nt_pf_128_32_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_256_64", transpose_2level_tuned_avx2_nt_pf_256_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_512_64", transpose_2level_tuned_avx2_nt_pf_512_64_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_v2", transpose_2level_tuned_avx2_nt_pf_v2_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_nofence", transpose_2level_tuned_avx2_nt_pf_nofence_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_pf_store", transpose_2level_tuned_avx2_pf_store_wrapper);
REGISTER_TRANSPOSE("2level_tuned_avx2_nt_pf_ab_tile128", transpose_2level_tuned_avx2_nt_pf_ab_tile128_wrapper);

//REGISTER_TRANSPOSE("2level_tuned_avx512", transpose_2level_tuned_avx512_wrapper);
//REGISTER_TRANSPOSE("2level_tuned_avx512_128_32", transpose_2level_tuned_avx512_128_32_wrapper);
//REGISTER_TRANSPOSE("2level_tuned_avx512_256_64", transpose_2level_tuned_avx512_256_64_wrapper);
//REGISTER_TRANSPOSE("2level_tuned_avx512_512_64", transpose_2level_tuned_avx512_512_64_wrapper);

// ---------------------------------------------------------------------------
// Utility: parse comma-separated sizes
// ---------------------------------------------------------------------------
static std::vector<int> parse_sizes(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        int v = std::atoi(tok.c_str());
        if (v > 0) out.push_back(v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// main — runs all registered transpose benchmarks
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int iters  = 3;
    int warmup = 1;
    // Default sizes (powers of 2)
    //std::string sizes_str = "256,512,1024,2048,4096,8192";
    std::string sizes_str = "1024,2048,4096";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--iters" || a == "-i") && i + 1 < argc)
            iters = std::max(1, std::atoi(argv[++i]));
        else if ((a == "--warmup" || a == "-w") && i + 1 < argc)
            warmup = std::max(0, std::atoi(argv[++i]));
        else if ((a == "--sizes" || a == "-s") && i + 1 < argc)
            sizes_str = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--iters N] [--warmup N] [--sizes 256,512,1024,...]\n";
            return 1;
        }
    }

    std::vector<int> sizes = parse_sizes(sizes_str);
    if (sizes.empty()) {
        std::cerr << "No valid sizes specified.\n";
        return 1;
    }

    std::cout << "iters=" << iters << ", warmup=" << warmup
              << ", sizes=" << sizes_str << "\n\n";

    for (int n : sizes) {
        std::cout << "=== N=" << n << " ===\n";
        print_table_header();

        for (auto *bench : BenchmarkRegistry::instance().all()) {
            if (n > bench->max_n()) {
                print_skip_row(bench->name());
                continue;
            }

            bench->setup(n);
            BenchmarkResult r = run_benchmark(bench, warmup, iters);
            bool ok = bench->verify();
            bench->teardown();

            // For transpose: report GB/s instead of GFLOP/s
            double gbs = (r.avg_ms > 0)
                ? bench->bytes_accessed(n) / (r.avg_ms / 1000.0) / 1e9
                : 0.0;
            
            // Reuse print_table_row but label the GFLOP/s column as GB/s
            std::cout << std::left << std::setw(20) << bench->name()
                      << std::right << std::fixed
                      << std::setprecision(3)
                      << std::setw(12) << r.avg_ms
                      << std::setw(12) << r.min_ms
                      << std::setw(12) << r.max_ms
                      << std::setprecision(2)
                      << std::setw(14) << gbs << " GB/s"
                      << std::setw(5) << (ok ? "PASS" : "FAIL")
                      << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
