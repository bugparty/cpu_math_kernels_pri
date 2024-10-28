#ifndef __MY_BLOCK_C__
#define __MY_BLOCK_C__
#define DEBUGPRINT1
#include "include.h"
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
                    if(fabs(fabs(A[t*n+curColumn]))>max){
                        maxind = t;max=fabs(A[t*n+curColumn]);
                    }
                }
                if(fabs(max) < 1e-8){
                    return 0;
                }else{
                    if (maxind != topRowInd){
                        int temp = ipiv[topRowInd];
                        ipiv[topRowInd] = ipiv[maxind];
                        ipiv[maxind] = temp;
                        swapRow(A,n,topRowInd,maxind); //line 33 of mylu.m
                    }
                }
                /*     for ii=x1+k+1:x2 %x1+k is the top row, divide
                        A(ii,jj)=A(ii,jj)/A(x1+k,y1+k);
                 */
                for(ii=x1+k+1;ii<=x2;++ii){
                    A[ii*n+jj] = A[ii*n+jj] / A[(x1+k)*n+y1+k];
                }
                needDiv=0;
                divColumn=jj;

            }else{
//                                for ii=x1+k+1:x2 % subtract
//                        A(ii,jj)=A(ii,jj)-A(ii,divColumn)*A(x1+k,jj);
                for(ii=x1+k+1;ii<=x2;++ii){
                    A[ii*n+jj] = A[ii*n+jj] -  A[ii*n+divColumn]*A[(x1+k)*n+jj];
                }

            }
        }
    }
    return 1;
}

void kernel_Avx512_S4(double *C,double *A,double *B,int n,int i,int j,int k)
{
register int ii,jj,kk;
static const int BLOCK_SIZE=64;
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
void kernel_naive(double *C,double *A, double*B, int n,int i,int j,int k,int blockSize){
    int ii,jj,kk;
    for(ii=i;ii<i+blockSize;ii++) {
        register int iin = ii * n;
        for (kk = k; kk < k + blockSize; kk++) {
            register double r = A[iin + kk];
            for (jj = j; jj < j + blockSize; jj++) {
                C[iin + jj] += B[kk * n + jj] * r;
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
static const int GEMM2_BLOCK_SIZE = 8;
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
                gemm2_kernel_ijk(A, n, i, j, k);
            }
}
void gemm2_kij(double*A, int n, int iend, int ib){
    int i, j, k;
     for (k = ib; k <= iend; k += GEMM2_BLOCK_SIZE)
    for (i = iend; i < n; i += GEMM2_BLOCK_SIZE)

            for (j = iend; j < n; j += GEMM2_BLOCK_SIZE)
            {
                gemm2_kernel_ijk(A, n, i, j, k);
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
        for(int jj=y1+k;jj<=y2;++jj){
            for(int ii=x1+k+1;ii<=x2;++ii){
                A[ii*n+jj] -= A[ii*n+multiColumn]*A[(x1+k)*n+jj];
            }
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
        if (mydgetrf2(A, ib,n-1,ib,iend, ipiv, n)==0)
        {
            printf("LU factoration failed: coefficient matrix is singular.\n");
            return ;
        }
        printM2(A,ib,n-1,ib,n-1,n);
        geppU(A, ib,iend, iend+1,n-1, ib,n);
        printM2(A,ib,n-1,ib,n-1,n);
        //A[(bid+1)*b:n,bid*b:(bid+1)*b]
        //A(ib+b:n,ib:ib+b-1)
        //A(iend+1:n,iend+1:n)=A(iend+1:n,iend+1:n)-A(iend+1:n,ib:iend)*A(ib:iend,iend+1:n);
        gemm2_ikj(A, n, iend, ib);
        printM2(A,ib,n-1,ib,n-1,n);

    }

    mydtrsv('L',A,B,n,ipiv);
    mydtrsv('U',A,B,n,ipiv);
}

#endif