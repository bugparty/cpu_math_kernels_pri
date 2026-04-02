#include <immintrin.h>
#include <stdlib.h>
#define BLOCK_SIZE 64

#ifdef __AVX512F__
void kernel_Avx512_S4_v2(double *C,double *A,double *B_T,int n,int i,int j,int k)
{
register int m,n_,k_;
for (m=i;m<i+BLOCK_SIZE;m++)
{
    for (n_=j;n_<j+BLOCK_SIZE;n_+=8)
    {
    register int c_offset = m*n+n_;
    //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
    register __m512d c_result_vec = _mm512_loadu_pd(&C[c_offset]);
    for (k_=k;k_<k+BLOCK_SIZE;k_+=4)
    {

        //Cij =     Ai0*B0j+...+Aik*Bkj + Ai(n-1)B(n-1)j
        //Ci(j+1) = Ai0*B0(j+1)
        //c00 = a00*b00+a01*b10+a02*b20+a03*b30
        //c01 = a00*b01+a01*b11+a02*b21+a03*b31
        //c02 = a00*b02+a01*b12+a03*b32+a03*b32
        //c03 = a00*b03+a01*b13+a03*b33+a03*b33
        //...
        //c07 = a00*b07+...
        //c00...c03 += a00*b00, a00*b01,a00*b02,a00*b03
        int a_offset=m*n+k_;
        // Load 4 consecutive elements from A row and broadcast each to a 512-bit vector
        __m512d a_broadcasted[4]; //4*8 doubles
        a_broadcasted[0] = _mm512_broadcastsd_pd(_mm_load_sd(&A[a_offset]));
        a_broadcasted[1] = _mm512_broadcastsd_pd(_mm_load_sd(&A[a_offset+1]));
        a_broadcasted[2] = _mm512_broadcastsd_pd(_mm_load_sd(&A[a_offset+2]));
        a_broadcasted[3] = _mm512_broadcastsd_pd(_mm_load_sd(&A[a_offset+3]));
        // B_T stores original B columns as contiguous rows
        int bt_offset = n_ * n + k_;
        __m512d b_rows[4];//4*8 doubles
        b_rows[0] = _mm512_loadu_pd(&B_T[bt_offset]);
        b_rows[1] = _mm512_loadu_pd(&B_T[bt_offset + n]);
        b_rows[2] = _mm512_loadu_pd(&B_T[bt_offset + 2 * n]);
        b_rows[3] = _mm512_loadu_pd(&B_T[bt_offset + 3 * n]);
        c_result_vec = _mm512_fmadd_pd(a_broadcasted[0], b_rows[0], c_result_vec);
        c_result_vec = _mm512_fmadd_pd(a_broadcasted[1], b_rows[1], c_result_vec);
        c_result_vec = _mm512_fmadd_pd(a_broadcasted[2], b_rows[2], c_result_vec);
        c_result_vec = _mm512_fmadd_pd(a_broadcasted[3], b_rows[3], c_result_vec);
    }
    //C[i*n+j]=c;
    _mm512_storeu_pd(&C[c_offset], c_result_vec);
    }
    }
}
#endif // __AVX512F__

#ifdef __AVX512F__
//state of art 
void dgemm7_v2(double *C,double *A,double *B,int n)
{
    double *B_T = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    int row, col;
    int block_i, block_k, block_j;

    if (B_T == NULL)
    {
        return;
    }

    for (row = 0; row < n; ++row)
    {
        for (col = 0; col < n; ++col)
        {
            B_T[col * n + row] = B[row * n + col];
        }
    }

    // Iterate over 64x64 blocks in blocking order: row blocks -> k-accumulation blocks -> column blocks
    for (block_i = 0; block_i < n; block_i += BLOCK_SIZE)
        for (block_k = 0; block_k < n; block_k += BLOCK_SIZE)
            for (block_j = 0; block_j < n; block_j += BLOCK_SIZE)
            {
                kernel_Avx512_S4_v2(C,A,B_T,n,block_i,block_j,block_k);
            }

    free(B_T);
}
#endif // __AVX512F__
