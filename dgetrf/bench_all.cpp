#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../include/benchmark.h"
#include "dgetrf/common.h"
#include "func_call.c"

static bool g_use_pool = true;

class SolveBenchmark : public BenchmarkBase {
public:
    using SolverFn = void (*)(double*, double*, int);

    SolveBenchmark(const char* name, SolverFn fn, int max_n_limit = INT_MAX)
        : name_(name), solver_(fn), max_n_limit_(max_n_limit) {}

    const char* name() const override { return name_.c_str(); }
    int max_n() const override { return max_n_limit_; }

    void setup(int n) override {
        n_ = n;
        const size_t matrix_elems = static_cast<size_t>(n_) * static_cast<size_t>(n_);
        const size_t vector_elems = static_cast<size_t>(n_);

        size_t bytes_per_iteration = (matrix_elems + vector_elems) * sizeof(double);
        size_t target_pool_bytes = 100ULL * 1024 * 1024;
        pool_size_ = g_use_pool ? (std::max<std::size_t>)(1, target_pool_bytes / bytes_per_iteration) : 1;

        A_base_.resize(pool_size_);
        b_base_.resize(pool_size_);
        A_work_.resize(pool_size_);
        x_work_.resize(pool_size_);

        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (size_t p = 0; p < pool_size_; ++p) {
            A_base_[p] = static_cast<double*>(_mm_malloc(matrix_elems * sizeof(double), 64));
            b_base_[p] = static_cast<double*>(_mm_malloc(vector_elems * sizeof(double), 64));
            A_work_[p] = static_cast<double*>(_mm_malloc(matrix_elems * sizeof(double), 64));
            x_work_[p] = static_cast<double*>(_mm_malloc(vector_elems * sizeof(double), 64));

            for (size_t i = 0; i < matrix_elems; ++i)
                A_base_[p][i] = dist(rng);

            // Diagonal dominance keeps random matrices well-conditioned for solve checks.
            for (int i = 0; i < n_; ++i)
                A_base_[p][static_cast<size_t>(i) * n_ + i] += static_cast<double>(n_);

            for (size_t i = 0; i < vector_elems; ++i)
                b_base_[p][i] = dist(rng);
        }
        current_idx_ = 0;
    }

    void run() override {
        reset_working_set(current_idx_);
        solver_(A_work_[current_idx_], x_work_[current_idx_], n_);
        current_idx_ = (current_idx_ + 1) % pool_size_;
    }

    bool verify() override {
        current_idx_ = 0;
        reset_working_set(0);
        solver_(A_work_[0], x_work_[0], n_);

        const double b_norm = l2_norm(b_base_[0], n_);
        const double residual_norm = compute_residual_norm(0);
        const double rel_residual = residual_norm / (std::max)(1e-12, b_norm);
        return std::isfinite(rel_residual) && rel_residual < 1e-6;
    }

    double flops(int n) const override {
        // LU factorization + two triangular solves.
        return (2.0 / 3.0) * n * n * n + 2.0 * n * n;
    }

    double bytes_accessed(int n) const override {
        return (static_cast<double>(n) * n + 3.0 * n) * sizeof(double);
    }

    void teardown() override {
        for (size_t p = 0; p < pool_size_; ++p) {
            _mm_free(A_base_[p]);
            _mm_free(b_base_[p]);
            _mm_free(A_work_[p]);
            _mm_free(x_work_[p]);
        }
        A_base_.clear();
        b_base_.clear();
        A_work_.clear();
        x_work_.clear();
        n_ = 0;
    }

private:
    void reset_working_set(size_t p) {
        const size_t matrix_elems = static_cast<size_t>(n_) * static_cast<size_t>(n_);
        const size_t vector_elems = static_cast<size_t>(n_);
        std::copy(A_base_[p], A_base_[p] + matrix_elems, A_work_[p]);
        std::copy(b_base_[p], b_base_[p] + vector_elems, x_work_[p]);
    }

    static double l2_norm(const double* x, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += x[i] * x[i];
        return std::sqrt(sum);
    }

    double compute_residual_norm(size_t p) const {
        double sum = 0.0;
        for (int i = 0; i < n_; ++i) {
            double ax = 0.0;
            const size_t row = static_cast<size_t>(i) * n_;
            for (int j = 0; j < n_; ++j)
                ax += A_base_[p][row + j] * x_work_[p][j];
            const double r = ax - b_base_[p][i];
            sum += r * r;
        }
        return std::sqrt(sum);
    }

    std::string name_;
    SolverFn solver_;
    int max_n_limit_;
    int n_ = 0;
    std::vector<double*> A_base_;
    std::vector<double*> b_base_;
    std::vector<double*> A_work_;
    std::vector<double*> x_work_;
    std::size_t pool_size_ = 1;
    std::size_t current_idx_ = 0;
};

#define REGISTER_SOLVER(namestr, fn)                                           \
    static SolveBenchmark _sbench_##fn(namestr, fn);                           \
    static int _sreg_##fn = (BenchmarkRegistry::instance().add(&_sbench_##fn), 0)

#define REGISTER_SOLVER_SLOW(namestr, fn, limit)                               \
    static SolveBenchmark _sbench_##fn(namestr, fn, limit);                    \
    static int _sreg_##fn = (BenchmarkRegistry::instance().add(&_sbench_##fn), 0)

REGISTER_SOLVER("my", my_f);
#ifdef __AVX512F__
REGISTER_SOLVER("my_block", my_block_f);
#endif
#if DGETRF_HAVE_EXTERNAL_BLAS_LAPACK
REGISTER_SOLVER_SLOW("lapack", lapack_f, 4096);
#endif

static std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int v = std::atoi(tok.c_str());
        if (v > 0)
            out.push_back(v);
    }
    return out;
}

int main(int argc, char** argv) {
    int iters = 3;
    int warmup = 1;
    std::string sizes_str = "512,1024,2048";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--iters" || a == "-i") && i + 1 < argc)
            iters = (std::max)(1, std::atoi(argv[++i]));
        else if ((a == "--warmup" || a == "-w") && i + 1 < argc)
            warmup = (std::max)(0, std::atoi(argv[++i]));
        else if ((a == "--sizes" || a == "-s") && i + 1 < argc)
            sizes_str = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--iters N] [--warmup N] [--sizes 256,512,1024,...]" << std::endl;
            return 1;
        }
    }

    const std::vector<int> sizes = parse_sizes(sizes_str);
    if (sizes.empty()) {
        std::cerr << "No valid sizes specified." << std::endl;
        return 1;
    }

    std::cout << "iters=" << iters << ", warmup=" << warmup
              << ", sizes=" << sizes_str << std::endl << std::endl;

    for (int n : sizes) {
        for (bool use_pool : {true, false}) {
            g_use_pool = use_pool;
            std::cout << "=== N=" << n << (use_pool ? " (Pool Mode)" : " (Fixed Memory)") << " ===" << std::endl;
            print_table_header();

            for (auto* bench : BenchmarkRegistry::instance().all()) {
                if (n > bench->max_n()) {
                    print_skip_row(bench->name());
                    continue;
                }

                bench->setup(n);
                const BenchmarkResult r = run_benchmark(bench, warmup, iters);
                const bool ok = bench->verify();
                bench->teardown();

                const double gflops = (r.avg_ms > 0)
                    ? bench->flops(n) / (r.avg_ms / 1000.0) / 1e9
                    : 0.0;
                print_table_row(bench->name(), r, gflops, ok);
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
