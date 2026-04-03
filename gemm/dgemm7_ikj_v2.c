#include <immintrin.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

// ⚡ Thunderbolt: AVX2 Register-blocked (4x8) Tiled IKJ DGEMM
// Target: AVX2 (Haswell+)
// Reason: Original inner loop was memory-bound. Tiling ensures L1/L2 residency, while 4x8 register blocking plus unrolling the K-loop by 4 maximizes the compute-to-memory ratio and hides FMA latency.
// Expected gain: ~4x throughput improvement (from ~4 GFLOPS to ~16+ GFLOPS)
#pragma GCC target("avx2,fma")
void dgemm7_ikj_v2(double *C, double *A, double *B, int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE) {
        int i_end = (i + BLOCK_SIZE < n) ? (i + BLOCK_SIZE) : n;
        for (k = 0; k < n; k += BLOCK_SIZE) {
            int k_end = (k + BLOCK_SIZE < n) ? (k + BLOCK_SIZE) : n;
            for (j = 0; j < n; j += BLOCK_SIZE) {
                int j_end = (j + BLOCK_SIZE < n) ? (j + BLOCK_SIZE) : n;

                for (int ii = i; ii <= i_end - 4; ii+=4) {
                    int jj = j;
                    for (; jj <= j_end - 8; jj += 8) {
                        __m256d c00 = _mm256_loadu_pd(&C[ii * n + jj]);
                        __m256d c01 = _mm256_loadu_pd(&C[ii * n + jj + 4]);

                        __m256d c10 = _mm256_loadu_pd(&C[(ii+1) * n + jj]);
                        __m256d c11 = _mm256_loadu_pd(&C[(ii+1) * n + jj + 4]);

                        __m256d c20 = _mm256_loadu_pd(&C[(ii+2) * n + jj]);
                        __m256d c21 = _mm256_loadu_pd(&C[(ii+2) * n + jj + 4]);

                        __m256d c30 = _mm256_loadu_pd(&C[(ii+3) * n + jj]);
                        __m256d c31 = _mm256_loadu_pd(&C[(ii+3) * n + jj + 4]);

                        int kk = k;
                        for (; kk <= k_end - 4; kk += 4) {
                            // kk
                            __m256d b0 = _mm256_loadu_pd(&B[kk * n + jj]);
                            __m256d b1 = _mm256_loadu_pd(&B[kk * n + jj + 4]);

                            __m256d a0 = _mm256_set1_pd(A[ii * n + kk]);
                            c00 = _mm256_fmadd_pd(a0, b0, c00);
                            c01 = _mm256_fmadd_pd(a0, b1, c01);

                            __m256d a1 = _mm256_set1_pd(A[(ii+1) * n + kk]);
                            c10 = _mm256_fmadd_pd(a1, b0, c10);
                            c11 = _mm256_fmadd_pd(a1, b1, c11);

                            __m256d a2 = _mm256_set1_pd(A[(ii+2) * n + kk]);
                            c20 = _mm256_fmadd_pd(a2, b0, c20);
                            c21 = _mm256_fmadd_pd(a2, b1, c21);

                            __m256d a3 = _mm256_set1_pd(A[(ii+3) * n + kk]);
                            c30 = _mm256_fmadd_pd(a3, b0, c30);
                            c31 = _mm256_fmadd_pd(a3, b1, c31);

                            // kk+1
                            b0 = _mm256_loadu_pd(&B[(kk+1) * n + jj]);
                            b1 = _mm256_loadu_pd(&B[(kk+1) * n + jj + 4]);

                            a0 = _mm256_set1_pd(A[ii * n + kk + 1]);
                            c00 = _mm256_fmadd_pd(a0, b0, c00);
                            c01 = _mm256_fmadd_pd(a0, b1, c01);

                            a1 = _mm256_set1_pd(A[(ii+1) * n + kk + 1]);
                            c10 = _mm256_fmadd_pd(a1, b0, c10);
                            c11 = _mm256_fmadd_pd(a1, b1, c11);

                            a2 = _mm256_set1_pd(A[(ii+2) * n + kk + 1]);
                            c20 = _mm256_fmadd_pd(a2, b0, c20);
                            c21 = _mm256_fmadd_pd(a2, b1, c21);

                            a3 = _mm256_set1_pd(A[(ii+3) * n + kk + 1]);
                            c30 = _mm256_fmadd_pd(a3, b0, c30);
                            c31 = _mm256_fmadd_pd(a3, b1, c31);

                            // kk+2
                            b0 = _mm256_loadu_pd(&B[(kk+2) * n + jj]);
                            b1 = _mm256_loadu_pd(&B[(kk+2) * n + jj + 4]);

                            a0 = _mm256_set1_pd(A[ii * n + kk + 2]);
                            c00 = _mm256_fmadd_pd(a0, b0, c00);
                            c01 = _mm256_fmadd_pd(a0, b1, c01);

                            a1 = _mm256_set1_pd(A[(ii+1) * n + kk + 2]);
                            c10 = _mm256_fmadd_pd(a1, b0, c10);
                            c11 = _mm256_fmadd_pd(a1, b1, c11);

                            a2 = _mm256_set1_pd(A[(ii+2) * n + kk + 2]);
                            c20 = _mm256_fmadd_pd(a2, b0, c20);
                            c21 = _mm256_fmadd_pd(a2, b1, c21);

                            a3 = _mm256_set1_pd(A[(ii+3) * n + kk + 2]);
                            c30 = _mm256_fmadd_pd(a3, b0, c30);
                            c31 = _mm256_fmadd_pd(a3, b1, c31);

                            // kk+3
                            b0 = _mm256_loadu_pd(&B[(kk+3) * n + jj]);
                            b1 = _mm256_loadu_pd(&B[(kk+3) * n + jj + 4]);

                            a0 = _mm256_set1_pd(A[ii * n + kk + 3]);
                            c00 = _mm256_fmadd_pd(a0, b0, c00);
                            c01 = _mm256_fmadd_pd(a0, b1, c01);

                            a1 = _mm256_set1_pd(A[(ii+1) * n + kk + 3]);
                            c10 = _mm256_fmadd_pd(a1, b0, c10);
                            c11 = _mm256_fmadd_pd(a1, b1, c11);

                            a2 = _mm256_set1_pd(A[(ii+2) * n + kk + 3]);
                            c20 = _mm256_fmadd_pd(a2, b0, c20);
                            c21 = _mm256_fmadd_pd(a2, b1, c21);

                            a3 = _mm256_set1_pd(A[(ii+3) * n + kk + 3]);
                            c30 = _mm256_fmadd_pd(a3, b0, c30);
                            c31 = _mm256_fmadd_pd(a3, b1, c31);
                        }
                        for (; kk < k_end; kk++) {
                            __m256d b0 = _mm256_loadu_pd(&B[kk * n + jj]);
                            __m256d b1 = _mm256_loadu_pd(&B[kk * n + jj + 4]);

                            __m256d a0 = _mm256_set1_pd(A[ii * n + kk]);
                            c00 = _mm256_fmadd_pd(a0, b0, c00);
                            c01 = _mm256_fmadd_pd(a0, b1, c01);

                            __m256d a1 = _mm256_set1_pd(A[(ii+1) * n + kk]);
                            c10 = _mm256_fmadd_pd(a1, b0, c10);
                            c11 = _mm256_fmadd_pd(a1, b1, c11);

                            __m256d a2 = _mm256_set1_pd(A[(ii+2) * n + kk]);
                            c20 = _mm256_fmadd_pd(a2, b0, c20);
                            c21 = _mm256_fmadd_pd(a2, b1, c21);

                            __m256d a3 = _mm256_set1_pd(A[(ii+3) * n + kk]);
                            c30 = _mm256_fmadd_pd(a3, b0, c30);
                            c31 = _mm256_fmadd_pd(a3, b1, c31);
                        }

                        _mm256_storeu_pd(&C[ii * n + jj], c00);
                        _mm256_storeu_pd(&C[ii * n + jj + 4], c01);

                        _mm256_storeu_pd(&C[(ii+1) * n + jj], c10);
                        _mm256_storeu_pd(&C[(ii+1) * n + jj + 4], c11);

                        _mm256_storeu_pd(&C[(ii+2) * n + jj], c20);
                        _mm256_storeu_pd(&C[(ii+2) * n + jj + 4], c21);

                        _mm256_storeu_pd(&C[(ii+3) * n + jj], c30);
                        _mm256_storeu_pd(&C[(ii+3) * n + jj + 4], c31);
                    }

                    // J remainder
                    for (; jj < j_end; jj++) {
                        double c0 = C[ii * n + jj];
                        double c1 = C[(ii+1) * n + jj];
                        double c2 = C[(ii+2) * n + jj];
                        double c3 = C[(ii+3) * n + jj];
                        for (int kk = k; kk < k_end; kk++) {
                            double b = B[kk * n + jj];
                            c0 += A[ii * n + kk] * b;
                            c1 += A[(ii+1) * n + kk] * b;
                            c2 += A[(ii+2) * n + kk] * b;
                            c3 += A[(ii+3) * n + kk] * b;
                        }
                        C[ii * n + jj] = c0;
                        C[(ii+1) * n + jj] = c1;
                        C[(ii+2) * n + jj] = c2;
                        C[(ii+3) * n + jj] = c3;
                    }
                }

                // II remainder
                for (int ii = i_end - (i_end - i) % 4; ii < i_end; ii++) {
                    int jj = j;
                    for (; jj <= j_end - 4; jj += 4) {
                        __m256d c0 = _mm256_loadu_pd(&C[ii * n + jj]);
                        for (int kk = k; kk < k_end; kk++) {
                            __m256d b0 = _mm256_loadu_pd(&B[kk * n + jj]);
                            __m256d a0 = _mm256_set1_pd(A[ii * n + kk]);
                            c0 = _mm256_fmadd_pd(a0, b0, c0);
                        }
                        _mm256_storeu_pd(&C[ii * n + jj], c0);
                    }
                    for (; jj < j_end; jj++) {
                        double c0 = C[ii * n + jj];
                        for (int kk = k; kk < k_end; kk++) {
                            c0 += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = c0;
                    }
                }
            }
        }
    }
}
