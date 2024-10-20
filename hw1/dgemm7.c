#include <immintrin.h>
#define BLOCK_SIZE 64


void kernel_Avx512_S4(double *C,double *A,double *B,int n,int i,int j,int k)
{
register int ii,jj,kk;
for (ii=i;ii<i+BLOCK_SIZE;ii++)
{
    for (jj=j;jj<j+BLOCK_SIZE;jj+=8)
    {
    register int ij = ii*n+jj;
    //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
    register __m512d Cijh8 = _mm512_loadu_pd(&C[ij]);
    for (kk=k;kk<k+BLOCK_SIZE;kk+=4)
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
        int ik=ii*n+kk;
        __m512d Aikx8 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik]));//A a(i,k)  a(i,k) a(i,k) a(i,k)
        __m512d Aikx8t = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+1]));
        __m512d Aikx8tt = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+2]));
        __m512d Aikx8ttt = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+3]));
        // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
        //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),
        int kj = kk*n+jj;
        __m512d Bkjh8 = _mm512_loadu_pd(&B[kj]);
        __m512d Bkjh8t = _mm512_loadu_pd(&B[kj+n]);
        __m512d Bkjh8tt = _mm512_loadu_pd(&B[kj+2*n]);
        __m512d Bkjh8ttt = _mm512_loadu_pd(&B[kj+3*n]);
        Cijh8 = _mm512_fmadd_pd(Aikx8, Bkjh8, Cijh8);
        Cijh8 = _mm512_fmadd_pd(Aikx8t, Bkjh8t, Cijh8);
        Cijh8 = _mm512_fmadd_pd(Aikx8tt, Bkjh8tt, Cijh8);
        Cijh8 = _mm512_fmadd_pd(Aikx8ttt, Bkjh8ttt, Cijh8);
        // c += A[ink]*B_T[jnk];
        // c += A[ink+1]*B_T[jnk+1];
        // c += A[ink+2]*B_T[jnk+2];
        // c += A[ink+3]*B_T[jnk+3];
    }
    //C[i*n+j]=c;
    _mm512_storeu_pd(&C[ij], Cijh8);
    }
    }
}
//state of art 
void dgemm7(double *C,double *A,double *B,int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (k = 0; k < n; k += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_Avx512_S4(C,A,B,n,i,j,k);
            }
}