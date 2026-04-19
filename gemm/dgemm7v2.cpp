#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#define BLOCK_SIZE 64

static inline int min_int(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

static void kernel_scalar_tail(double *C, const double *A, const double *B,
                               int n, int row_begin, int row_end,
                               int col_begin, int col_end,
                               int k_begin, int k_end)
{
    for (int row = row_begin; row < row_end; ++row)
    {
        for (int col = col_begin; col < col_end; ++col)
        {
            double cij = C[row * n + col];
            for (int kk = k_begin; kk < k_end; ++kk)
            {
                cij += A[row * n + kk] * B[kk * n + col];
            }
            C[row * n + col] = cij;
        }
    }
}

static void kernel_Avx2_S4_v2(double *C, const double *A, const double *B,
                              int n, int row_begin, int row_end,
                              int col_begin, int col_end,
                              int k_begin, int k_end)
{
    const int col_vec_end = col_begin + ((col_end - col_begin) / 8) * 8;
    const int k_vec_end = k_begin + ((k_end - k_begin) / 4) * 4;

    for (int row = row_begin; row < row_end; ++row)
    {
        int col = col_begin;
        for (; col < col_vec_end; col += 8)
        {
            const int c_offset = row * n + col;
            __m256d c_lo = _mm256_loadu_pd(&C[c_offset]);
            __m256d c_hi = _mm256_loadu_pd(&C[c_offset + 4]);

            int kk = k_begin;
            for (; kk < k_vec_end; kk += 4)
            {
                const int a_offset = row * n + kk;

                const __m256d a0 = _mm256_set1_pd(A[a_offset]);
                const __m256d a1 = _mm256_set1_pd(A[a_offset + 1]);
                const __m256d a2 = _mm256_set1_pd(A[a_offset + 2]);
                const __m256d a3 = _mm256_set1_pd(A[a_offset + 3]);

                const int b0_offset = kk * n + col;
                const int b1_offset = b0_offset + n;
                const int b2_offset = b1_offset + n;
                const int b3_offset = b2_offset + n;

                const __m256d b0_lo = _mm256_loadu_pd(&B[b0_offset]);
                const __m256d b0_hi = _mm256_loadu_pd(&B[b0_offset + 4]);
                const __m256d b1_lo = _mm256_loadu_pd(&B[b1_offset]);
                const __m256d b1_hi = _mm256_loadu_pd(&B[b1_offset + 4]);
                const __m256d b2_lo = _mm256_loadu_pd(&B[b2_offset]);
                const __m256d b2_hi = _mm256_loadu_pd(&B[b2_offset + 4]);
                const __m256d b3_lo = _mm256_loadu_pd(&B[b3_offset]);
                const __m256d b3_hi = _mm256_loadu_pd(&B[b3_offset + 4]);

                c_lo = _mm256_fmadd_pd(a0, b0_lo, c_lo);
                c_hi = _mm256_fmadd_pd(a0, b0_hi, c_hi);
                c_lo = _mm256_fmadd_pd(a1, b1_lo, c_lo);
                c_hi = _mm256_fmadd_pd(a1, b1_hi, c_hi);
                c_lo = _mm256_fmadd_pd(a2, b2_lo, c_lo);
                c_hi = _mm256_fmadd_pd(a2, b2_hi, c_hi);
                c_lo = _mm256_fmadd_pd(a3, b3_lo, c_lo);
                c_hi = _mm256_fmadd_pd(a3, b3_hi, c_hi);
            }

            for (; kk < k_end; ++kk)
            {
                const __m256d a = _mm256_set1_pd(A[row * n + kk]);
                const int b_offset = kk * n + col;
                c_lo = _mm256_fmadd_pd(a, _mm256_loadu_pd(&B[b_offset]), c_lo);
                c_hi = _mm256_fmadd_pd(a, _mm256_loadu_pd(&B[b_offset + 4]), c_hi);
            }

            _mm256_storeu_pd(&C[c_offset], c_lo);
            _mm256_storeu_pd(&C[c_offset + 4], c_hi);
        }

        if (col < col_end)
        {
            kernel_scalar_tail(C, A, B, n, row, row + 1, col, col_end, k_begin, k_end);
        }
    }
}

extern "C" void dgemm7_v2_avx2(double *C, double *A, double *B, int n)
{
    for (int block_i = 0; block_i < n; block_i += BLOCK_SIZE)
    {
        const int row_end = min_int(block_i + BLOCK_SIZE, n);
        for (int block_k = 0; block_k < n; block_k += BLOCK_SIZE)
        {
            const int k_end = min_int(block_k + BLOCK_SIZE, n);
            for (int block_j = 0; block_j < n; block_j += BLOCK_SIZE)
            {
                const int col_end = min_int(block_j + BLOCK_SIZE, n);
                kernel_Avx2_S4_v2(C, A, B, n, block_i, row_end, block_j, col_end, block_k, k_end);
            }
        }
    }
}

#ifdef __AVX512F__
static void kernel_Avx512_S4_v2(double *C, const double *A, const double *B,
                                int n, int row_begin, int row_end,
                                int col_begin, int col_end,
                                int k_begin, int k_end)
{
    const int col_vec_end = col_begin + ((col_end - col_begin) / 8) * 8;
    const int k_vec_end = k_begin + ((k_end - k_begin) / 4) * 4;

    for (int row = row_begin; row < row_end; ++row)
    {
        int col = col_begin;
        for (; col < col_vec_end; col += 8)
        {
            const int c_offset = row * n + col;
            __m512d c_vec = _mm512_loadu_pd(&C[c_offset]);

            int kk = k_begin;
            for (; kk < k_vec_end; kk += 4)
            {
                const int a_offset = row * n + kk;
                const __m512d a0 = _mm512_set1_pd(A[a_offset]);
                const __m512d a1 = _mm512_set1_pd(A[a_offset + 1]);
                const __m512d a2 = _mm512_set1_pd(A[a_offset + 2]);
                const __m512d a3 = _mm512_set1_pd(A[a_offset + 3]);

                const int b0_offset = kk * n + col;
                const int b1_offset = b0_offset + n;
                const int b2_offset = b1_offset + n;
                const int b3_offset = b2_offset + n;

                c_vec = _mm512_fmadd_pd(a0, _mm512_loadu_pd(&B[b0_offset]), c_vec);
                c_vec = _mm512_fmadd_pd(a1, _mm512_loadu_pd(&B[b1_offset]), c_vec);
                c_vec = _mm512_fmadd_pd(a2, _mm512_loadu_pd(&B[b2_offset]), c_vec);
                c_vec = _mm512_fmadd_pd(a3, _mm512_loadu_pd(&B[b3_offset]), c_vec);
            }

            for (; kk < k_end; ++kk)
            {
                const __m512d a = _mm512_set1_pd(A[row * n + kk]);
                c_vec = _mm512_fmadd_pd(a, _mm512_loadu_pd(&B[kk * n + col]), c_vec);
            }

            _mm512_storeu_pd(&C[c_offset], c_vec);
        }

        if (col < col_end)
        {
            kernel_scalar_tail(C, A, B, n, row, row + 1, col, col_end, k_begin, k_end);
        }
    }
}

void dgemm7_v2(double *C,double *A,double *B,int n)
{
    for (int block_i = 0; block_i < n; block_i += BLOCK_SIZE)
    {
        const int row_end = min_int(block_i + BLOCK_SIZE, n);
        for (int block_k = 0; block_k < n; block_k += BLOCK_SIZE)
        {
            const int k_end = min_int(block_k + BLOCK_SIZE, n);
            for (int block_j = 0; block_j < n; block_j += BLOCK_SIZE)
            {
                const int col_end = min_int(block_j + BLOCK_SIZE, n);
                kernel_Avx512_S4_v2(C, A, B, n, block_i, row_end, block_j, col_end, block_k, k_end);
            }
        }
    }
}
#endif
