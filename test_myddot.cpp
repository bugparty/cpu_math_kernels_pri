#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include "include/aligned_buffer.h"

double myddot_original(int n, const double *x,  const double *y){
    register double sum = 0.0;
    register int i;
    for(i=0;i<n;++i){
        sum += x[i]*y[i];
    }
    return sum;
}

double myddot_avx2(int n, const double *x,  const double *y){
    register double sum = 0.0;
    register int i = 0;
    __m256d sum_v = _mm256_setzero_pd();
    for(;i+3<n;i+=4){
        sum_v = _mm256_fmadd_pd(_mm256_loadu_pd(&x[i]), _mm256_loadu_pd(&y[i]), sum_v);
    }
    __m128d t1 = _mm_add_pd(_mm256_extractf128_pd(sum_v, 0), _mm256_extractf128_pd(sum_v, 1));
    __m128d t2 = _mm_add_pd(t1, _mm_shuffle_pd(t1, t1, 1));
    sum = _mm_cvtsd_f64(t2);
    for(;i<n;++i){
        sum += x[i]*y[i];
    }
    return sum;
}

double myddot_avx512(int n, const double *x,  const double *y){
    register double sum = 0.0;
    register int i = 0;
    __m512d sum_v = _mm512_setzero_pd();
    for(;i+7<n;i+=8){
        sum_v = _mm512_fmadd_pd(_mm512_loadu_pd(&x[i]), _mm512_loadu_pd(&y[i]), sum_v);
    }
    sum = _mm512_reduce_add_pd(sum_v);
    for(;i<n;++i){
        sum += x[i]*y[i];
    }
    return sum;
}

int main() {
    std::size_t n = 16384;
    std::vector<double> data1(n, 1.0);
    std::vector<double> data2(n, 1.0);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 100000; ++k) myddot_original(n, data1.data(), data2.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "myddot_original: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 100000; ++k) myddot_avx2(n, data1.data(), data2.data());
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "myddot_avx2: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms\n";

    auto t5 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 100000; ++k) myddot_avx512(n, data1.data(), data2.data());
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "myddot_avx512: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms\n";
}
