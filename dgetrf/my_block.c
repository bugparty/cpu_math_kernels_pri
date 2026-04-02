#ifndef __MY_BLOCK_C__
#define __MY_BLOCK_C__
#define DEBUGPRINT1
#include "dgetrf/common.h"
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#if defined(__clang__) || defined(__GNUC__)
#define DGETRF_PRAGMA_IMPL(x) _Pragma(#x)
#define DGETRF_UNROLL_4 DGETRF_PRAGMA_IMPL(GCC unroll 4)
#else
#define DGETRF_UNROLL_4
#endif

void printM2(double* A, int x1,int x2,int y1,int y2,int n){
#ifdef DEBUGPRINT
    puts("begin matrix");
    for(int i=x1;i<=x2;i++){
        for(int j=y1;j<=y2;j++){
            printf("%lf ", A[i*n+j]);
        }
        puts("");
    }
    puts("end matrix");
#endif
}
#define GEMM2_BLOCK_SIZE 128
int mydgetrf2(double *A,int x1,int x2,int y1,int y2, int *ipiv,int n){
    int k;
    int ii,jj;
    int needDiv=1;
    int divColumn;
    for(k=0;k<=y2-y1;++k){
        needDiv=1;
        divColumn=0;
        for(jj=y1+k;jj<=y2;++jj){
            if(needDiv){
                int topRowInd = x1+k;
                int curColumn = y1+k;
                int maxind=topRowInd;
                double max = fabs(A[topRowInd*n+curColumn]);
                for(int t=topRowInd+1;t<=x2;++t){
                    double absVal = fabs(A[t*n+curColumn]);
                    if(absVal >max){
                        maxind = t;max=absVal;
                    }
                }
                if(fabs(max) < 1e-8){
                    return 0;
                }else{
                    if (maxind != topRowInd){
                        int temp = ipiv[topRowInd];
                        ipiv[topRowInd] = ipiv[maxind];
                        ipiv[maxind] = temp;
                        swapRow2(A,n,topRowInd,maxind); //line 33 of mylu.m
                    }
                }
                /*     for ii=x1+k+1:x2 %x1+k is the top row, divide
                        A(ii,jj)=A(ii,jj)/A(x1+k,y1+k);
                 */
                DGETRF_UNROLL_4
                for(ii=x1+k+1;ii<=x2;++ii){
                    A[ii*n+jj] /=  A[(x1+k)*n+y1+k];
                }
                needDiv=0;
                divColumn=jj;

            }else{
//                                for ii=x1+k+1:x2 % subtract
//                        A(ii,jj)=A(ii,jj)-A(ii,divColumn)*A(x1+k,jj);
                DGETRF_UNROLL_4
                for(ii=x1+k+1;ii<=x2;++ii){
                    A[ii*n+jj] -=  A[ii*n+divColumn]*A[(x1+k)*n+jj];
                }

            }
        }
    }
    return 1;
}
void kernel_reg4(double *C,double *A,double *B,int i,int j,int k,int n)
{
register int ii, jj, kk;
    const int STRIDE = 2;
    for(ii=i;ii< i+GEMM2_BLOCK_SIZE;ii+=2){
         for(jj=j;jj<j+GEMM2_BLOCK_SIZE;jj+=2){
              register double c00,c01,c10,c11;
              register int t1,t2;
              t1 = ii*n+jj;t2 = t1+n;
              c00=C[t1];c01=C[t1+1];c10=C[t2];c11=C[t2+1];
              for(kk=k;kk<k+GEMM2_BLOCK_SIZE;kk+=2){
                   register int ta = i*n+kk; register int tta = ta+n; register int tb = kk*n+j; register int ttb = tb+n;
                   register double a00 = A[ta]; register double a10 = A[tta]; register double b00 = B[tb]; register double b01 = B[tb+1];

                   c00 -= a00*b00 ; c01 -= a00*b01 ; c10 -= a10*b00 ; c11 -= a10*b01 ;

                   a00 = A[ta+1]; a10 = A[tta+1]; b00 = B[ttb]; b01 = B[ttb+1];

                   c00 -= a00*b00 ; c01 -= a00*b01 ; c10 -= a10*b00 ; c11 -= a10*b01 ;
              }
              C[t1]=c00;
              C[t1+1]=c01;
              C[t2]=c10;
              C[t2+1]=c11;
         }
    }

}
void kernel_reg16(double *C,int n,int i,int j,int k){
     int ii, jj, kk;
    const int STRIDE = 4;
    // #pragma omp parallel for collapse(2) schedule(static)
    for (ii = i; ii < (i + GEMM2_BLOCK_SIZE); ii += STRIDE)
        //#pragma GCC unroll 4
        for (jj = j; jj < (j + GEMM2_BLOCK_SIZE); jj += STRIDE)
        {
             int t = ii*n + jj;
             int tt = t + n;
             int ttt = tt + n;
             int tttt = ttt + n;
            register double c00 = C[t], c01 = C[t + 1], c02 = C[t + 2], c03 = C[t + 3];
            register double c10 = C[tt], c11 = C[tt + 1], c12 = C[tt + 2], c13 = C[tt + 3];
            register double c20 = C[ttt], c21 = C[ttt + 1], c22 = C[ttt + 2], c23 = C[ttt + 3];
            register double c30 = C[tttt], c31 = C[tttt + 1], c32 = C[tttt + 2], c33 = C[tttt + 3];
            //#pragma GCC unroll 4
            for(kk = k; kk < (k + GEMM2_BLOCK_SIZE); kk += STRIDE)
            {
                 int ta = ii*n + kk;
                 int tb = kk*n + jj;
                 int tta = ii*n + kk + n;
                 int ttta = ii*n + kk + 2*n;
                 int tttta = ii*n + kk + 3*n;
                int tbb = kk*n + jj+n;
                int tbbb = kk*n + jj+n*2;
                int tb4 = kk*n + jj+n*3;
                register double a00 = C[ta]; //, a01 = A[ta + 1], a02 = A[ta + 2], a03 = A[ta + 3];
                register double a10 = C[tta]; //, a11 = A[tta + 1], a12 = A[tta + 2], a13 = A[tta + 3];
                register double a20 = C[ttta]; //, a21 = A[ttta + 1], a22 = A[ttta + 2], a23 = A[ttta + 3];
                register double a30 = C[tttta]; //, a31 = A[tttta + 1], a32 = A[tttta + 2], a33 = A[tttta + 3];



                register double b00 = C[tb], b01 = C[tb + 1], b02 = C[tb + 2], b03 = C[tb + 3];


                c00 -= a00 * b00; c10 -= a10 * b00; c20 -= a20 * b00; c30 -= a30 * b00;
                c01 -= a00 * b01; c11 -= a10 * b01; c21 -= a20 * b01; c31 -= a30 * b01;
                __builtin_prefetch(&C[tbb], 0, 3);


                c02 -= a00 * b02; c12 -= a10 * b02; c22 -= a20 * b02; c32 -= a30 * b02;
                c03 -= a00 * b03; c13 -= a10 * b03; c23 -= a20 * b03; c33 -= a30 * b03;

                a00 = C[ta + 1], a10 = C[tta + 1], a20 = C[ttta + 1], a30 = C[tttta + 1];
                b00 = C[tbb], b01 = C[tbb + 1], b02 = C[tbb + 2], b03 = C[tbb + 3];

                c00 -= a00 * b00; c10 -= a10 * b00; c20 -= a20 * b00; c30 -= a30 * b00;
                c01 -= a00 * b01; c11 -= a10 * b01; c21 -= a20 * b01; c31 -= a30 * b01;
                __builtin_prefetch(&C[tbbb], 0, 3);
                c02 -= a00 * b02; c12 -= a10 * b02; c22 -= a20 * b02; c32 -= a30 * b02;
                c03 -= a00 * b03; c13 -= a10 * b03; c23 -= a20 * b03; c33 -= a30 * b03;


                a00 = C[ta + 2], a10 = C[tta + 2], a20 = C[ttta + 2], a30 = C[tttta + 2];
                b00 = C[tbbb], b01 = C[tbbb + 1], b02 = C[tbbb + 2], b03 = C[tbbb + 3];

                c00 -= a00 * b00; c10 -= a10 * b00; c20 -= a20 * b00; c30 -= a30 * b00;
                c01 -= a00 * b01; c11 -= a10 * b01; c21 -= a20 * b01; c31 -= a30 * b01;
                __builtin_prefetch(&C[tb4], 0, 3);
                c02 -= a00 * b02; c12 -= a10 * b02; c22 -= a20 * b02; c32 -= a30 * b02;
                c03 -= a00 * b03; c13 -= a10 * b03; c23 -= a20 * b03; c33 -= a30 * b03;

                a00 = C[ta + 3], a10 = C[tta + 3], a20 = C[ttta + 3], a30 = C[tttta + 3];
                b00 = C[tb4], b01 = C[tb4 + 1], b02 = C[tb4 + 2], b03 = C[tb4 + 3];

                c00 -= a00 * b00; c10 -= a10 * b00; c20 -= a20 * b00; c30 -= a30 * b00;
                c01 -= a00 * b01; c11 -= a10 * b01; c21 -= a20 * b01; c31 -= a30 * b01;
                c02 -= a00 * b02; c12 -= a10 * b02; c22 -= a20 * b02; c32 -= a30 * b02;
                c03 -= a00 * b03; c13 -= a10 * b03; c23 -= a20 * b03; c33 -= a30 * b03;
            }

            C[t] = c00; C[t + 1] = c01; C[t + 2] = c02; C[t + 3] = c03;
            C[tt] = c10; C[tt + 1] = c11; C[tt + 2] = c12; C[tt + 3] = c13;
            C[ttt] = c20; C[ttt + 1] = c21; C[ttt + 2] = c22; C[ttt + 3] = c23;
            C[tttt] = c30; C[tttt + 1] = c31; C[tttt + 2] = c32; C[tttt + 3] = c33;

        }
}
void kernel_Avx512_S4(double *C,double *A,double *B,int n,int i,int j,int k)
{
 int ii,jj,kk;
static const int BLOCK_SIZE=64;
__m256d neg_one = _mm256_set1_pd(-1.0);
for (ii=i;ii<i+BLOCK_SIZE;ii++)
{
    DGETRF_UNROLL_4
    for (jj=j;jj<j+BLOCK_SIZE;jj+=8)
    {
     double *pCij = &C[ii*n+jj];
    //c(i,j) c(i,j+1) c(i,j+2) c(i,j+3)
     __m512d Cijh8 = _mm512_loadu_pd(pCij);
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
        const int ik=ii*n+kk;
        __m256d CikX4 = _mm256_loadu_pd(&C[ik]);
        const int kj = kk*n+jj;
        const double* pCkj = &C[kj];

        __builtin_prefetch(pCkj, 0, 3);

        __m256d negatedCikX4 = _mm256_mul_pd(CikX4, neg_one);
        const double * pCkj1n = pCkj+n;
        __m128d xmm0 = _mm256_extractf128_pd(negatedCikX4, 0); // Extract lower 128 bits
        const double * pCkj2n = pCkj1n+n;

        __m128d xmm1 = _mm256_extractf128_pd(negatedCikX4, 1); // Extract upper 128 bits
          const double * pCkj3n = pCkj2n+n;
        __builtin_prefetch(pCkj1n, 0, 3);
        __m512d Aikx8 = _mm512_broadcastsd_pd(_mm_permute_pd(xmm0, 0b00));//A a(i,k)  a(i,k) a(i,k) a(i,k)
        __m512d Aikx8t = _mm512_broadcastsd_pd(_mm_permute_pd(xmm0, 0b11));
        __m512d Aikx8tt = _mm512_broadcastsd_pd(_mm_permute_pd(xmm1, 0b00));
        __m512d Aikx8ttt = _mm512_broadcastsd_pd(_mm_permute_pd(xmm1, 0b11));
        // B b(k,j) b(k,j+1) b(k,j+2) b(k,j+3)
        //so result are a(i,k)*b(k,j),  a(i,k)*b(k,j+1), a(i,k)*b(k,j+2), a(i,k)*b(k,j+3),

        __m512d Bkjh8 = _mm512_loadu_pd(pCkj);
        __m512d Bkjh8t = _mm512_loadu_pd(pCkj1n);
        __m512d Bkjh8tt = _mm512_loadu_pd(pCkj2n);
        __m512d Bkjh8ttt = _mm512_loadu_pd(pCkj3n);
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
    _mm512_storeu_pd(pCij, Cijh8);
    }
    }
}
void kernel_naive(double *C,double *A, double*B, int n,int i,int j,int k){
    int ii,jj,kk;
    for(ii=i;ii<i+GEMM2_BLOCK_SIZE;ii++) {
        register int iin = ii * n;
        for (kk = k; kk < k + GEMM2_BLOCK_SIZE; kk++) {
            register double r = A[iin + kk];
            for (jj = j; jj < j + GEMM2_BLOCK_SIZE; jj++) {
                C[iin + jj] -= B[kk * n + jj] * r;
            }
        }
    }
}
void mydgemm(double *A,double *B,int n,int bid,int b)
{
    //TODO
    //Implement a matrix multiplication here following dgemm7 in HW1
    //The first matrix is A[(bid+1)*b:n,bid*b:(bid+1)*b]
    //The second matrix is B[bid*b:(bid+1)*b,(bid+1)*b:n]
    //b is the block size for dgetrf
}
void gemm1(double*A,int n,int iend,int ib){
    for (int i = iend; i < n; i++) {
        for (int j = iend; j < n; j++) {
            double temp = 0.0;
            for (int k = ib; k <= iend; k++) {
                temp += A[i*n+k] * A[k*n+j];
            }
            A[i*n+j] -= temp;
        }
    }
}

void gemm2_kernel_ijk(double*A, int n, int i, int j, int k){
    for (int ii = i; ii < i+GEMM2_BLOCK_SIZE; ii++) {
        for (int jj = j; jj < j+GEMM2_BLOCK_SIZE; jj++) {
            double temp = 0.0;
            for (int kk = k; kk <= k+GEMM2_BLOCK_SIZE; kk++) {
                temp += A[ii * n + kk] * A[kk * n + jj];
            }
            A[ii * n + jj] -= temp;
        }
    }
}
void gemm2_kernel_ikj(double*A, int n, int i, int j, int k){
    int ii,jj,kk;
     for(ii=i;ii<i+GEMM2_BLOCK_SIZE;ii++){
        register int iin = ii*n;
        for(kk=k;kk<k+GEMM2_BLOCK_SIZE;kk++)
        {
            register double r = A[iin+kk];
            for(jj=j;jj<j+GEMM2_BLOCK_SIZE;jj++){
                A[iin+jj] -= A[kk*n+jj] * r;
            }
        }
     }
}

void gemm2_ikj(double*A, int n, int iend, int ib){
    int i, j, k;

    for (i = iend; i < n; i += GEMM2_BLOCK_SIZE)
        for (k = ib; k <= iend; k += GEMM2_BLOCK_SIZE)
            for (j = iend; j < n; j += GEMM2_BLOCK_SIZE)
            {
                //gemm2_kernel_ijk(A, n, i, j, k);
                kernel_Avx512_S4(A,A,A,n,i,j,k);
                //kernel_reg16(A,n,i,j,k);
                //kernel_reg4(A,A,A,i,j,k,n);
                //kernel_reg4(A,A,A,i,j,k,n);
                //kernel_naive(A,A,A,n,i,j,k);
            }
}
void gemm2_kij(double*A, int n, int iend, int ib){
    register int i, j, k;
     for (k = ib; k <= iend; k += GEMM2_BLOCK_SIZE)
    for (i = iend; i < n; i += GEMM2_BLOCK_SIZE)
            for (j = iend; j < n; j += GEMM2_BLOCK_SIZE)
            {
                //gemm2_kernel_ijk(A, n, i, j, k);
                //kernel_Avx512_S4(A,A,A,n,i,j,k);
                //kernel_reg16(A,n,i,j,k);
               // kernel_naive(A,A,A,n,i,j,k);
                kernel_reg4(A,A,A,i,j,k,n);
            }
}


int mydgetrf_block(double *A,int *ipiv,int n)
{
    int b=2;//MODIFY
    for (int ib=0;ib<n;ib+=b){
        int iend = ib+b-1;


    }
    return 1;
}
void geppUv1(double* A, int x1,int x2,int y1,int y2, int multiColumn,int n ){
    for(int k=0;k<=y2-y1;++k){
        int x1kn=(x1+k)*n;
        for(int ii=x1+k+1;ii<=x2;++ii){
            int iin=ii*n;
            double A_iinPmc = A[iin+multiColumn];
        for(int jj=y1+k;jj<=y2;++jj)
            {
                A[iin+jj] -= A_iinPmc*A[x1kn+jj];
            }
        }
    }
}
/* function geppU(multiColumn, x1,x2,y1,y2) %update the U block
    for k = 0:y2-y1
        for jj=y1+k:y2
                for ii=x1+k+1:x2 % subtract
                        A(ii,jj)=A(ii,jj)-A(ii,multiColumn)*A(x1+k,jj);
                end
        end
    end
end */
void geppU(double* A, int x1,int x2,int y1,int y2, int multiColumn,int n ){
    for(int k=0;k<=y2-y1;++k){
        int x1kn=(x1+k)*n;
        for(int ii=x1+k+1;ii<=x2;++ii){
            int iin=ii*n;
            double A_iinPmc = A[iin+multiColumn];
        for(int jj=y1+k;jj<=y2;++jj)
            {
                A[iin+jj] -= A_iinPmc*A[x1kn+jj];
            }
        }
    }
}
void swapRowBlock(double *A, int n, int y1,int y2, int first, int second){
    int i;
    for(i=0;i<n;i++){
        double t = A[first*n+i];
        A[first*n+i]= A[second*n+i];
        A[second*n+i] =t;
    }
}
int my_block_dgetrf(double *A,int x1,int x2,int y1,int y2,int *ipiv,int n)
{
    int i=0,t;
    int maxind;
    double max;
    int rows = x2-x1+1;
    int columns = y2-y1+1;
    int minRC = MIN(rows, columns);
    for (i=x1;i<minRC;++i){// line 16 of mylu.m
        maxind=i;
        max = fabs(A[i*n+i]);
        for(t=i+1;t<=x2;++t){
            if( fabs(A[t*n+i] > max)){
                maxind = t;
                max = fabs(A[t*n+i]);//line 21 of mylu.m
            }
        }
        if( fabs(max -0) < 1e-8){//line 24  of mylu.m
            return 0;
        }else{
            if (maxind != i){
                ipiv[i] = ipiv[maxind];
                swapRowBlock(A,n,y1,y2,i,maxind); //line 33 of mylu.m
            }
        }
        int j,k;
        for(j=i+1; j<=x2; ++j){
            int tjni = j*n+i;
            double Ajni = A[tjni];
            Ajni /=  A[i*n+i];
            for(k=i+1;k<=y2;++k){
                A[j*n+k] -= Ajni*A[i*n+k];
            }
            A[tjni]=Ajni;
        }
    }
    //The return value (an integer) can be 0 or 1
    //If 0, the matrix is irreducible and the result will be ignored
    //If 1, the result is valid
    return 1;
}
void updateMaByPivot(double*A, int x1,int x2,int y1,int y2,int *ipiv,int n){
    for(int i=x1;i<=x2;i++){
        if(i!= ipiv[i]){
            swapRowBlock(A,n,y1,y2,i,ipiv[i]);
        }
    }
}
void my_block_f(double *A,double *B,int n)
{
    int *ipiv=(int*)malloc(n*sizeof(int));
    int b;
    for (int i=0;i<n;i++)
        ipiv[i]=i;
    if (n > 300){
        b=GEMM2_BLOCK_SIZE;
    }else{
        b=2;
    }
    for (int ib=0;ib<n;ib+=b){
        int iend = ib+b-1;
        if (my_block_dgetrf(A, ib,n-1,ib,iend, ipiv, n)==0)
        {
            printf("LU factoration failed: coefficient matrix is singular.\n");
            return ;
        }
        //printM2(A,ib,n-1,ib,n-1,n);
        updateMaByPivot(A, ib,n-1, 0, ib-1, ipiv, n);
        geppU(A, ib,iend, iend+1,n-1, ib,n);
        //printM2(A,ib,n-1,ib,n-1,n);
        updateMaByPivot(A, ib,n-1, iend+1, n-1, ipiv, n);
        //A(iend+1:n,iend+1:n)=A(iend+1:n,iend+1:n)-A(iend+1:n,ib:iend)*A(ib:iend,iend+1:n);
        //gemm2_kij(A, n, iend, ib);
        gemm2_ikj(A, n, iend, ib);
       // printM2(A,ib,n-1,ib,n-1,n);

    }

    mydtrsv('L',A,B,n,ipiv);
    mydtrsv('U',A,B,n,ipiv);
}

#endif
