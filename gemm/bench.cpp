#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <immintrin.h>

#include "dgemm.h"

struct KernelEntry {
    std::string name;
    std::function<void(double *, double *, double *, int)> fn;
};

double *B_T = nullptr;

double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void fill_random(double *dst, int count, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < count; ++i) {
        dst[i] = dist(rng);
    }
}

void copy_matrix(const double *src, double *dst, int count) {
    std::copy(src, src + count, dst);
}

bool verify_matrix(const double *ref, const double *out, int count, double tol = 1e-6) {
    for (int i = 0; i < count; ++i) {
        const double diff = std::abs(ref[i] - out[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

void reference_dgemm(double *C, const double *A, const double *B, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double cij = C[i * n + j];
            for (int k = 0; k < n; ++k) {
                cij += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = cij;
        }
    }
}

std::vector<KernelEntry> build_kernels() {
    return {
        {"reference", reference_dgemm},
        {"dgemm1", dgemm1},
        {"dgemm3", dgemm3},
        {"dgemm3v2", dgemm3v2},
        {"dgemmBT1", dgemmBT1},
#ifdef __AVX512F__
        {"dgemm7", dgemm7},
#endif
        {"dgemm7_ijk", dgemm7_ijk},
        {"dgemm7_kij", dgemm7_kij},
        {"dgemm7_ikj", dgemm7_ikj},
        {"dgemmAVX", dgemmAVX},
#ifdef __AVX512F__
        {"dgemmAVX512", dgemmAVX512},
        {"dgemmAVX512B", dgemmAVX512B},
#endif
    };
}

void print_kernel_menu(const std::vector<KernelEntry> &kernels) {
    std::cout << "Please select a kernel by index:\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        std::cout << "  " << i << ": " << kernels[i].name << '\n';
    }
}

int main(int argc, char **argv) {
    const int SIZE[] = {
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
        1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
        3000,
    };
    constexpr int size_count = static_cast<int>(sizeof(SIZE) / sizeof(SIZE[0]));
    constexpr int max_size = 3000;
    constexpr int repeats = 3;

    auto kernels = build_kernels();
    
    // Special argument to print max kernel index
    if (argc == 2 && (std::string(argv[1]) == "--count" || std::string(argv[1]) == "-c")) {
        std::cout << kernels.size() - 1 << std::endl;
        return 0;
    }
    
    if (argc != 2) {
        std::cout << "Usage: ./bench <kernel_index>\n";
        std::cout << "       ./bench --count   # Print max kernel index\n";
        print_kernel_menu(kernels);
        return 1;
    }

    const int kernel_idx = std::atoi(argv[1]);
    if (kernel_idx < 0 || kernel_idx >= static_cast<int>(kernels.size())) {
        std::cerr << "Please enter a valid kernel number (0-" << kernels.size() - 1 << ").\n";
        print_kernel_menu(kernels);
        return 2;
    }

    const KernelEntry &kernel = kernels[kernel_idx];
    std::cout << "Selected kernel [" << kernel_idx << "] " << kernel.name << "\n";

    const std::size_t max_elems = static_cast<std::size_t>(max_size) * static_cast<std::size_t>(max_size);
    const std::size_t bytes = max_elems * sizeof(double);

    double *A = static_cast<double *>(_mm_malloc(bytes, 64));
    double *B = static_cast<double *>(_mm_malloc(bytes, 64));
    double *C = static_cast<double *>(_mm_malloc(bytes, 64));
    double *C_ref = static_cast<double *>(_mm_malloc(bytes, 64));
    double *C_base = static_cast<double *>(_mm_malloc(bytes, 64));
    B_T = static_cast<double *>(_mm_malloc(bytes, 64));

    if (!A || !B || !C || !C_ref || !C_base || !B_T) {
        std::cerr << "Failed to allocate buffers for size " << max_size << "\n";
        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
        _mm_free(C_ref);
        _mm_free(C_base);
        _mm_free(B_T);
        return 3;
    }

    std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    fill_random(A, static_cast<int>(max_elems), rng);
    fill_random(B, static_cast<int>(max_elems), rng);
    fill_random(C_base, static_cast<int>(max_elems), rng);
    std::fill_n(B_T, max_elems, 0.0);

    const int upper_limit = (kernel_idx <= 4 && kernel_idx != 0) ? 10 : size_count;

    for (int idx = 0; idx < upper_limit; ++idx) {
        const int n = SIZE[idx];
        const int elems = n * n;

        copy_matrix(C_base, C, elems);
        copy_matrix(C_base, C_ref, elems);

        if (kernel_idx == 0) {
            kernel.fn(C_ref, A, B, n);
        } else {
            kernel.fn(C, A, B, n);
            reference_dgemm(C_ref, A, B, n);
            if (!verify_matrix(C_ref, C, elems)) {
                std::cerr << "Failed correctness verification at size " << n << ".\n";
                _mm_free(A);
                _mm_free(B);
                _mm_free(C);
                _mm_free(C_ref);
                _mm_free(C_base);
                _mm_free(B_T);
                return 4;
            }
        }

        copy_matrix(C_ref, C, elems);

        const double t0 = now_sec();
        for (int r = 0; r < repeats; ++r) {
            kernel.fn(C, A, B, n);
        }
        const double t1 = now_sec();
        const double avg_time = (t1 - t0) / repeats;
        const double gflops = 2.0 * 1e-9 * repeats * n * n * n / (t1 - t0);

        std::cout << "M=N=K=" << n << ": average elapsed " << avg_time
                  << " s, performance " << gflops << " GFLOPS" << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);
    _mm_free(C_base);
    _mm_free(B_T);

    return 0;
}