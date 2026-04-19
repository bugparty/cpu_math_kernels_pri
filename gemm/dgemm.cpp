#include "dgemm.h"
#include "common.h"
#include <immintrin.h>

extern "C" {

void dgemm3(double *C,double *A,double *B,int n)
{
    for(int i=0;i<n;i+=2){
        for(int j=0;j<n;j+=4){
            register double c00,c01,c10,c11;
            register double c20,c21,c30,c31;
            register int t1,t2;
            t1 = i*n+j;
            t2 = (i+1)*n+j;
            c00=C[t1];
            c01=C[t1+1];c10=C[t2];c11=C[t2+1];
            c20=C[t1+2];c21=C[t1+3];c30=C[t2+2];c31=C[t2+3];
            for(int k=0;k<n;k+=2){
                register int ta = i*n+k; register int tta = ta+n; register int tb = k*n+j; register int ttb = tb+n;
                register int tb2 = tb+2,  ttb2 = tb2+n;
                register double a00 = A[ta], a10 = A[tta], b00 = B[tb], b01 = B[tb+1];
                register double a01 = A[ta+1], a11 = A[tta+1];
                c00 += a00*b00 ; c01 += a00*b01 ; c10 += a10*b00 ; c11 += a10*b01 ;
                b00 = B[ttb]; b01 = B[ttb+1];
                c00 += a01*b00 ; c01 += a01*b01 ; c10 += a11*b00 ; c11 += a11*b01 ;
                b00 = B[tb+2]; b01 = B[tb+3];
                c20 += a00*b00 ; c21 += a00*b01 ; c30 += a10*b00 ; c31 += a10*b01 ;
                b00 = B[ttb2]; b01 = B[ttb2+1];
                c20 += a01*b00 ; c21 += a01*b01 ; c30 += a11*b00 ; c31 += a11*b01 ;
            }
            C[t1]=c00;
            C[t1+1]=c01;
            C[t1+2]=c20;
            C[t1+3]=c21;
            C[t2]=c10;
            C[t2+1]=c11;
            C[t2+2]=c30;
            C[t2+3]=c31;
        }
    }

}
void dgemm3v2(double *C,double *A,double *B,int n_)
{
int i;
register int j,k;
register const int n=n_;
const int STRIDE = 3;
register double c00,c01,c02, c10,c11,c12,c20,c21,c22;
register double a00, a10,a20, b00, b01, b02;
register int t,tt,ttt;
for(i=0;i<n;i+=STRIDE){
     for(j=0;j<n;j+=STRIDE){
          t = i*n+j;
          c00=C[t];c01=C[t+1];c02=C[t+2];
          tt= t+n;
          c10=C[tt];c11=C[tt+1];c12=C[tt+2];
          ttt=tt+n;
          c20=C[ttt];c21=C[ttt+1];c22=C[ttt+2];
          for(k=0;k<n;k+=STRIDE){
               register int ta = i*n+k; register int tb = k*n+j;
               register int tta = ta+n;register int ttb = tb+n;
               register int ttta = tta+n;register int tttb = ttb+n;
               // Load elements of A,B into temporary variables
               a00 = A[ta]; a10 = A[tta]; a20 = A[ttta];
               b00 = B[tb]; b01 = B[tb+1]; b02 = B[tb+2];

               //below uses register a00 a10 a20  b00 b01 b02
               c00 += a00 * b00;c01 += a00 * b01;c02 += a00 * b02;
               c10 += a10 * b00;c11 += a10 * b01;c12 += a10 * b02 ;
               c20 += a20 * b00 ;c21 += a20 * b01;c22 += a20 * b02;

               //after this line a00,a10,a20,b00,b01,b02 will not be used anymore
               // we need to load a01 a02 a11 a12 a21 a22 b10 b11 b12 b20 b21 b22 12 register, but 7 available
               //if we select a01,a11,a21 we need b10 b11 b12
               // so we load a00=A01, a10=A11, a20=A21, b00=B10 b01=B11 b02=B12
               a00 = A[ta+1]; a10 = A[tta+1]; a20 = A[ttta+1];
               b00 = B[ttb];  b01 = B[ttb+1]; b02 = B[ttb+2];
               c00 += a00 * b00;c01 += a00 * b01 ;c02 += a00 * b02 ;
               c10 += a10 * b00 ;c11 += a10 * b01;c12 += a10 * b02;
               c20 += a20 * b00;c21 += a20 * b01; c22 += a20 * b02;
               //after this line, a00,a10,a20, b00,b01,b02, b10,b11,b12 is not being used anymore
               // we still need a02 a12 a22 b20 b21 b22
               // so we load a00=A02, a10=A12, a20=A22, b00=B20 b01=B21 b02=B22
               a00 = A[ta+2];a10 = A[tta+2];a20 = A[ttta+2];
               b00 = B[tttb]; b01 = B[tttb+1]; b02 = B[tttb+2];
               c00 += a00 * b00;
               c01 += a00 * b01;
               c02 += a00 * b02;

               c10 += a10 * b00;
               c11 += a10 * b01;
               c12 += a10 * b02;

               c20 += a20 * b00;
               c21 += a20 * b01;
               c22 += a20 * b02;
          }
          C[t]=c00;C[t+1]=c01;C[t+2]=c02;
          C[tt]=c10;C[tt+1]=c11;C[tt+2]=c12;
          C[ttt]=c20;C[ttt+1]=c21;C[ttt+2]=c22;
     }
}

}
void transpose( const double  * const A, double * const  A_T, int n) {
    transpose_tiled(A, A_T, n);
}
void dgemm1(double *C,double *A,double *B,int n)
{
    int i,j,k;
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j++)
        {
            for (k=0;k<n;k++)
            {
                C[i*n+j] += A[i*n+k]*B[k*n+j];
            }
        }
    }
}

void dgemm1T(double *C,double *A,double *B,int n)
{
    transpose(B,B_T, n);
    register int i,j,k;
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j++)
        {
            register double c = C[i*n+j];
            for (k=0;k<n;k++)
            {
                c += A[i*n+k]*B_T[j*n+k];
            }
            C[i*n+j]=c;
        }
    }
}

#define BLOCK_SIZE 16
#define STRIDE 4

void kernel_R4x4(double *C,double *A,double *B,int n,int i,int j,int k)
{
    register int ii,jj,kk;
for (ii = i; ii < (i + BLOCK_SIZE); ii += STRIDE)
    for (jj = j; jj < (j + BLOCK_SIZE); jj += STRIDE)
    {
        register int t = ii*n + jj;
        register int tt = t + n;
        register int ttt = tt + n;
        register int tttt = ttt + n;
        register double c00 = C[t], c01 = C[t + 1], c02 = C[t + 2], c03 = C[t + 3];
        register double c10 = C[tt], c11 = C[tt + 1], c12 = C[tt + 2], c13 = C[tt + 3];
        register double c20 = C[ttt], c21 = C[ttt + 1], c22 = C[ttt + 2], c23 = C[ttt + 3];
        register double c30 = C[tttt], c31 = C[tttt + 1], c32 = C[tttt + 2], c33 = C[tttt + 3];

        for(kk = k; kk < (k + BLOCK_SIZE); kk += STRIDE)
        {
            register int ta = ii*n + kk;
            register int tta = ta + n;
            register int ttta = tta + n;
            register int tttta = ttta + n;
            register double a00 = A[ta];
            register double a10 = A[tta];
            register double a20 = A[ttta];
            register double a30 = A[tttta];

            register int tb = kk*n + jj;
            register double b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];


            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

            tb += n;
            a00 = A[ta + 1], a10 = A[tta + 1], a20 = A[ttta + 1], a30 = A[tttta + 1];
            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

            tb += n;
            a00 = A[ta + 2], a10 = A[tta + 2], a20 = A[ttta + 2], a30 = A[tttta + 2];
            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

            tb += n;
            a00 = A[ta + 3], a10 = A[tta + 3], a20 = A[ttta + 3], a30 = A[tttta + 3];
            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;
        }

        C[t] = c00; C[t + 1] = c01; C[t + 2] = c02; C[t + 3] = c03;
        C[tt] = c10; C[tt + 1] = c11; C[tt + 2] = c12; C[tt + 3] = c13;
        C[ttt] = c20; C[ttt + 1] = c21; C[ttt + 2] = c22; C[ttt + 3] = c23;
        C[tttt] = c30; C[tttt + 1] = c31; C[tttt + 2] = c32; C[tttt + 3] = c33;
    }
}
void dgemm7_2(double *C,double *A,double *B,int n)
{
    int i, j, k, ii, jj, kk;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (j = 0; j < n; j += BLOCK_SIZE)
            for (k = 0; k < n; k += BLOCK_SIZE)
                for (ii = i; ii < (i + BLOCK_SIZE); ii += STRIDE)
                    for (jj = j; jj < (j + BLOCK_SIZE); jj += STRIDE)
                    {
                        register int t = ii*n + jj;
                        register int tt = t + n;
                        register int ttt = tt + n;
                        register int tttt = ttt + n;
                        register double c00 = C[t], c01 = C[t + 1], c02 = C[t + 2], c03 = C[t + 3];
                        register double c10 = C[tt], c11 = C[tt + 1], c12 = C[tt + 2], c13 = C[tt + 3];
                        register double c20 = C[ttt], c21 = C[ttt + 1], c22 = C[ttt + 2], c23 = C[ttt + 3];
                        register double c30 = C[tttt], c31 = C[tttt + 1], c32 = C[tttt + 2], c33 = C[tttt + 3];

                        for(kk = k; kk < (k + BLOCK_SIZE); kk += STRIDE)
                        {
                            register int ta = ii*n + kk;
                            register int tta = ta + n;
                            register int ttta = tta + n;
                            register int tttta = ttta + n;
                            register double a00 = A[ta]; //, a01 = A[ta + 1], a02 = A[ta + 2], a03 = A[ta + 3];
                            register double a10 = A[tta]; //, a11 = A[tta + 1], a12 = A[tta + 2], a13 = A[tta + 3];
                            register double a20 = A[ttta]; //, a21 = A[ttta + 1], a22 = A[ttta + 2], a23 = A[ttta + 3];
                            register double a30 = A[tttta]; //, a31 = A[tttta + 1], a32 = A[tttta + 2], a33 = A[tttta + 3];

                            register int tb = kk*n + jj;
                            register double b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];


                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 1], a10 = A[tta + 1], a20 = A[ttta + 1], a30 = A[tttta + 1];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 2], a10 = A[tta + 2], a20 = A[ttta + 2], a30 = A[tttta + 2];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 3], a10 = A[tta + 3], a20 = A[ttta + 3], a30 = A[tttta + 3];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;
                        }

                        C[t] = c00; C[t + 1] = c01; C[t + 2] = c02; C[t + 3] = c03;
                        C[tt] = c10; C[tt + 1] = c11; C[tt + 2] = c12; C[tt + 3] = c13;
                        C[ttt] = c20; C[ttt + 1] = c21; C[ttt + 2] = c22; C[ttt + 3] = c23;
                        C[tttt] = c30; C[tttt + 1] = c31; C[tttt + 2] = c32; C[tttt + 3] = c33;
                    }
}
void dgemm7_ijk(double *C,double *A,double *B,int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (j = 0; j < n; j += BLOCK_SIZE)
            for (k = 0; k < n; k += BLOCK_SIZE)
            {
                kernel_R4x4(C,A,B,n,i,j,k);
            }
}
void dgemm7_kij(double *C,double *A,double *B,int n)
{
    int i, j, k;
    for (k = 0; k < n; k += BLOCK_SIZE)
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (j = 0; j < n; j += BLOCK_SIZE)
            {

                kernel_R4x4(C,A,B,n,i,j,k);
            }
}
void dgemm7_ikj_unused(double *C,double *A,double *B,int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
    for (k = 0; k < n; k += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_R4x4(C,A,B,n,i,j,k);
            }
}
void dgemmBT1(double *C,double *A,double *B,int n)
{
    transpose(B,B_T, n);
    register int i,j,k;
    int oi,oj,ok;
    //16 is the best value

    for (oi=0;oi<n;oi+=BLOCK_SIZE)
        for (oj=0;oj<n;oj+=BLOCK_SIZE)
            for (ok=0;ok<n;ok+=BLOCK_SIZE)
            {
                for (i=oi;i<oi+BLOCK_SIZE;i++)
                {
                    for (j=oj;j<oj+BLOCK_SIZE;j++)
                    {
                        register double c = C[i*n+j];
                        for (k=ok;k<ok+BLOCK_SIZE;k++)
                        {
                            c += A[i*n+k]*B_T[j*n+k];
                        }
                        C[i*n+j]=c;
                    }
                }
            }
}
/*40s
 */
void dgemmAVX(double *C,double *A,double *B,int n)
{
    // __builtin_prefetch(&A[0], 0, 3);
    // __builtin_prefetch(&B[0], 0, 3);
    // __builtin_prefetch(&B_T[0], 0, 3);
    // __builtin_prefetch(&C[0], 0, 3);
    register int i,j,k;
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j+=4)
        {
            register int ij = i*n+j;
            //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
            __m256d Cijh4 = _mm256_loadu_pd(&C[ij]);
            for (k=0;k<n;k++)
            {
                int ik=i*n+k;
                //Cij =     Ai0*B0j+...+Aik*Bkj + Ai(n-1)B(n-1)j
                //Ci(j+1) = Ai0*B0(j+1)
                //c00 = a00*b00+a01*b10+a02*b20+a03*b30
                //c01 = a00*b01+a01*b11+a02*b21+a03*b31
                //c02 = a00*b02+a01*b12+a03*b32+a03*b32
                //c03 = a00*b03+a01*b13+a03*b33+a03*b33

                //c00...c03 += a00*b00, a00*b01,a00*b02,a00*b03
                //A a(i,k)  a(i,k) a(i,k) a(i,k)
                __m256d Aikx4 = _mm256_broadcast_sd(&A[ik]);
                // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
                //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),
                __m256d Bkjh4 = _mm256_loadu_pd(&B[k*n+j]);
                Cijh4 = _mm256_fmadd_pd(Aikx4, Bkjh4, Cijh4);
                                // c += A[ink]*B_T[jnk];
                // c += A[ink+1]*B_T[jnk+1];
                // c += A[ink+2]*B_T[jnk+2];
                // c += A[ink+3]*B_T[jnk+3];

            }
            //C[i*n+j]=c;
            _mm256_storeu_pd(&C[ij], Cijh4);
        }
    }
}
/*
 *28s n2048
 */
#ifdef __AVX512F__
void dgemmAVX512(double *C,double *A,double *B,int n)
{

    register int i,j,k;
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j+=8)
        {
            register int ij = i*n+j;
            //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
            __m512d Cijh8 = _mm512_loadu_pd(&C[ij]);
            for (k=0;k<n;k++)
            {
                int ik=i*n+k;
                //Cij =     Ai0*B0j+...+Aik*Bkj + Ai(n-1)B(n-1)j
                //Ci(j+1) = Ai0*B0(j+1)
                //c00 = a00*b00+a01*b10+a02*b20+a03*b30
                //c01 = a00*b01+a01*b11+a02*b21+a03*b31
                //c02 = a00*b02+a01*b12+a03*b32+a03*b32
                //c03 = a00*b03+a01*b13+a03*b33+a03*b33

                //c00...c03 += a00*b00, a00*b01,a00*b02,a00*b03
                //A a(i,k)  a(i,k) a(i,k) a(i,k)
                __m512d Aikx8 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik]));
                // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
                //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),
                __m512d Bkjh8 = _mm512_loadu_pd(&B[k*n+j]);
                Cijh8 = _mm512_fmadd_pd(Aikx8, Bkjh8, Cijh8);
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
#else
void dgemmAVX512(double *C,double *A,double *B,int n) { (void)C; (void)A; (void)B; (void)n; }
#endif // __AVX512F__
// speed: 7s O3 1.3s
#ifdef __AVX512F__
void kernel_Avx512(double *C,double *A,double *B,int n,int i,int j,int k)
{
    register int ii,jj,kk;
    for (ii=i;ii<i+BLOCK_SIZE;ii++)
    {
        for (jj=j;jj<j+BLOCK_SIZE;jj+=8)
        {
            register int ij = ii*n+jj;
            //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
            register __m512d Cijh8 = _mm512_loadu_pd(&C[ij]);
            for (kk=k;kk<k+BLOCK_SIZE;kk++)
            {
                int ik=ii*n+kk;
                //Cij =     Ai0*B0j+...+Aik*Bkj + Ai(n-1)B(n-1)j
                //Ci(j+1) = Ai0*B0(j+1)
                //c00 = a00*b00+a01*b10+a02*b20+a03*b30
                //c01 = a00*b01+a01*b11+a02*b21+a03*b31
                //c02 = a00*b02+a01*b12+a03*b32+a03*b32
                //c03 = a00*b03+a01*b13+a03*b33+a03*b33

                //c00...c03 += a00*b00, a00*b01,a00*b02,a00*b03
                //A a(i,k)  a(i,k) a(i,k) a(i,k)
                __m512d Aikx8 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik]));
                // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
                //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),
                __m512d Bkjh8 = _mm512_loadu_pd(&B[kk*n+jj]);
                Cijh8 = _mm512_fmadd_pd(Aikx8, Bkjh8, Cijh8);
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
#endif // __AVX512F__
#ifdef __AVX512F__
void dgemmAVX512B(double *C,double *A,double *B,int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (k = 0; k < n; k += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_Avx512(C,A,B,n,i,j,k);
            }
}
#else
void dgemmAVX512B(double *C,double *A,double *B,int n) { (void)C; (void)A; (void)B; (void)n; }
#endif // __AVX512F__
#ifdef __AVX512F__
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
#endif // __AVX512F__
#ifdef __AVX512F__
void kernel_Avx512_S6(double *C,double *A,double *B,int n,int i,int j,int k)
{
    register int ii,jj,kk;
    for (ii=i;ii<i+BLOCK_SIZE;ii++)
    {
        for (jj=j;jj<j+BLOCK_SIZE;jj+=8)
        {
            register int ij = ii*n+jj;
            //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
            register __m512d Cijh8 = _mm512_loadu_pd(&C[ij]);
            for (kk=k;kk<k+BLOCK_SIZE;kk+=6)
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
                __m512d Aikx8t2 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+1]));
                __m512d Aikx8t3 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+2]));
                __m512d Aikx8t4 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+3]));
                __m512d Aikx8t5 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+4]));
                __m512d Aikx8t6 = _mm512_broadcastsd_pd(_mm_load_sd(&A[ik+5]));
                // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
                //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),
                int kj = kk*n+jj;
                __m512d Bkjh8 = _mm512_loadu_pd(&B[kj]);
                __m512d Bkjh8t2 = _mm512_loadu_pd(&B[kj+n]);
                __m512d Bkjh8t3 = _mm512_loadu_pd(&B[kj+2*n]);
                __m512d Bkjh8t4 = _mm512_loadu_pd(&B[kj+3*n]);
                __m512d Bkjh8t5 = _mm512_loadu_pd(&B[kj+4*n]);
                __m512d Bkjh8t6 = _mm512_loadu_pd(&B[kj+5*n]);
                Cijh8 = _mm512_fmadd_pd(Aikx8, Bkjh8, Cijh8);
                Cijh8 = _mm512_fmadd_pd(Aikx8t2, Bkjh8t2, Cijh8);
                Cijh8 = _mm512_fmadd_pd(Aikx8t3, Bkjh8t3, Cijh8);
                Cijh8 = _mm512_fmadd_pd(Aikx8t4, Bkjh8t4, Cijh8);
                Cijh8 = _mm512_fmadd_pd(Aikx8t5, Bkjh8t5, Cijh8);
                Cijh8 = _mm512_fmadd_pd(Aikx8t6, Bkjh8t6, Cijh8);
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
#endif // __AVX512F__
#ifdef __AVX512F__
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
#else
void dgemm7(double *C,double *A,double *B,int n) { (void)C; (void)A; (void)B; (void)n; }
#endif // __AVX512F__

#include "dgemm7_ikj.c"

} // extern "C"
