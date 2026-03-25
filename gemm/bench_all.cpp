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
#include "dgemm.h"
#include "dgemm7olds.h"

#ifndef GEMM_HAVE_MKL
#define GEMM_HAVE_MKL 0
#endif

#if GEMM_HAVE_MKL
#include <mkl.h>
#endif

// B_T is declared extern in dgemm.h; we own the allocation here
double *B_T = nullptr;

// ---------------------------------------------------------------------------
// Reference DGEMM (used for verification)
// ---------------------------------------------------------------------------
static void reference_dgemm(double *C, const double *A, const double *B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double cij = C[i * n + j];
            for (int k = 0; k < n; ++k)
                cij += A[i * n + k] * B[k * n + j];
            C[i * n + j] = cij;
        }
    }
}

// ---------------------------------------------------------------------------
// GemmBenchmark — wraps a plain C DGEMM kernel function
// ---------------------------------------------------------------------------
class GemmBenchmark : public BenchmarkBase {
public:
    using KernelFn = void (*)(double *, double *, double *, int);

    GemmBenchmark(const char *name, KernelFn fn, int max_n_limit = INT_MAX)
        : name_(name), kernel_(fn), max_n_limit_(max_n_limit) {}

    const char *name() const override { return name_.c_str(); }
    int max_n() const override { return max_n_limit_; }

    void setup(int n) override {
        n_ = n;
        // dgemm7_ijk/kij/ikj use BLOCK_SIZE=16 and STRIDE=4 micro-tiles.
        // kernel_R4x4 iterates ii = [i, i+BLOCK_SIZE) even when i+BLOCK_SIZE > n,
        // accessing rows up to (ii+3) beyond n. We round up to the next
        // multiple of BLOCK_SIZE and add one extra block as a safe cushion.
        const int BLOCK_SIZE = 16;
        int n_padded = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
        size_t bytes = (size_t)n_padded * n_padded * sizeof(double);
        A_      = static_cast<double *>(_mm_malloc(bytes, 64));
        B_      = static_cast<double *>(_mm_malloc(bytes, 64));
        C_      = static_cast<double *>(_mm_malloc(bytes, 64));
        C_ref_  = static_cast<double *>(_mm_malloc(bytes, 64));
        C_base_ = static_cast<double *>(_mm_malloc(bytes, 64));
        B_T     = static_cast<double *>(_mm_malloc(bytes, 64));

        // Fill A, B, C_base with random doubles; zero B_T
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        size_t elems = (size_t)n * n;
        for (size_t i = 0; i < elems; ++i) {
            A_[i]      = dist(rng);
            B_[i]      = dist(rng);
            C_base_[i] = dist(rng);
        }
        std::fill_n(B_T, elems, 0.0);

        // Precompute reference result
        std::copy(C_base_, C_base_ + elems, C_ref_);
        reference_dgemm(C_ref_, A_, B_, n);
    }

    // run() resets C from C_base before each call so timing is consistent
    void run() override {
        size_t elems = (size_t)n_ * n_;
        std::copy(C_base_, C_base_ + elems, C_);
        kernel_(C_, A_, B_, n_);
    }

    bool verify() override {
        // Run kernel once on a fresh C and compare against precomputed C_ref_
        size_t elems = (size_t)n_ * n_;
        std::copy(C_base_, C_base_ + elems, C_);
        kernel_(C_, A_, B_, n_);
        constexpr double tol = 1e-6;
        for (size_t i = 0; i < elems; ++i) {
            if (std::abs(C_[i] - C_ref_[i]) > tol)
                return false;
        }
        return true;
    }

    double flops(int n) const override { return 2.0 * n * n * n; }
    double bytes_accessed(int n) const override {
        return 3.0 * (double)n * n * sizeof(double);
    }

    void teardown() override {
        _mm_free(A_);      A_      = nullptr;
        _mm_free(B_);      B_      = nullptr;
        _mm_free(C_);      C_      = nullptr;
        _mm_free(C_ref_);  C_ref_  = nullptr;
        _mm_free(C_base_); C_base_ = nullptr;
        _mm_free(B_T);     B_T     = nullptr;
        n_ = 0;
    }

private:
    std::string name_;
    KernelFn    kernel_;
    int         max_n_limit_;
    int         n_      = 0;
    double     *A_      = nullptr;
    double     *B_      = nullptr;
    double     *C_      = nullptr;
    double     *C_ref_  = nullptr;
    double     *C_base_ = nullptr;
};

// ---------------------------------------------------------------------------
// Registration macros
// ---------------------------------------------------------------------------
// Regular kernel: runs at all sizes
#define REGISTER_GEMM(namestr, fn)                                             \
    static GemmBenchmark _gbench_##fn(namestr, fn);                            \
    static int _greg_##fn = (BenchmarkRegistry::instance().add(&_gbench_##fn), 0)

// Slow kernel: skipped above max_n_limit
#define REGISTER_GEMM_SLOW(namestr, fn, limit)                                 \
    static GemmBenchmark _gbench_##fn(namestr, fn, limit);                     \
    static int _greg_##fn = (BenchmarkRegistry::instance().add(&_gbench_##fn), 0)

// ---------------------------------------------------------------------------
// Register all GEMM kernels
// (reference and early dgemm variants are slow at large n — cap at 1000)
// ---------------------------------------------------------------------------

// Wrap reference_dgemm in the correct signature for registration
static void reference_wrapper(double *C, double *A, double *B, int n) {
    reference_dgemm(C, A, B, n);
}

#if GEMM_HAVE_MKL
static void dgemm_mkl_wrapper(double *C, double *A, double *B, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, A, n,
                B, n,
                1.0, C, n);
}
#endif

// Slow O(n^3) naive kernels: capped at n<=1000 to avoid multi-minute runtimes
REGISTER_GEMM_SLOW("reference",  reference_wrapper, 1000);
REGISTER_GEMM_SLOW("dgemm1",     dgemm1,            1000);
REGISTER_GEMM_SLOW("dgemm3",     dgemm3,            1000);
REGISTER_GEMM_SLOW("dgemm3v2",   dgemm3v2,          1000);
REGISTER_GEMM     ("dgemmBT1",   dgemmBT1);
REGISTER_GEMM     ("dgemm71",    dgemm71);
REGISTER_GEMM     ("dgemm72",    dgemm72);
REGISTER_GEMM     ("dgemm74",    dgemm74);
REGISTER_GEMM     ("dgemm7_raw", dgemm7_raw);
REGISTER_GEMM     ("dgemm7_ijk", dgemm7_ijk);
REGISTER_GEMM     ("dgemm7_kij", dgemm7_kij);
REGISTER_GEMM     ("dgemm7_ikj", dgemm7_ikj);
#if GEMM_HAVE_MKL
REGISTER_GEMM     ("mkl_dgemm",  dgemm_mkl_wrapper);
#endif
// dgemmAVX lacks cache blocking and becomes memory-bandwidth-bound beyond n~1500;
// cap at 2000 to keep benchmark runtime reasonable
REGISTER_GEMM_SLOW("dgemmAVX",   dgemmAVX,          2000);
#ifdef __AVX512F__
REGISTER_GEMM     ("dgemm7",       dgemm7);
REGISTER_GEMM     ("dgemm7_v2",    dgemm7_v2);
REGISTER_GEMM     ("dgemmAVX512",  dgemmAVX512);
REGISTER_GEMM     ("dgemmAVX512B", dgemmAVX512B);
#endif
#ifdef __AVX2__
REGISTER_GEMM     ("dgemm7_v2_avx2", dgemm7_v2_avx2);
#endif

// ---------------------------------------------------------------------------
// main
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

int main(int argc, char **argv) {
    int iters  = 3;
    int warmup = 1;
    // Sizes are multiples of lcm(16, 3)=48 so all blocked kernels
    // (BLOCK_SIZE=16 for dgemm7_*/dgemmBT1, STRIDE=3 for dgemm3v2)
    // iterate exactly to the matrix boundary without out-of-bounds access.
    std::string sizes_str = "96,480,960,1920";

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
                      << " [--iters N] [--warmup N] [--sizes 100,500,1000,...]\n";
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
        std::cout << "=== M=N=K=" << n << " ===\n";
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

            double gflops = (r.avg_ms > 0)
                ? bench->flops(n) / (r.avg_ms / 1000.0) / 1e9
                : 0.0;
            print_table_row(bench->name(), r, gflops, ok);
        }
        std::cout << "\n";
    }

    return 0;
}
