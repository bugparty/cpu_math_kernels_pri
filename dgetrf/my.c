#ifndef __MY_C__
#define __MY_C__

#include "include.h"
#define DEBUGPRINT1
void swapRow(double *A, int n, int first, int second){
    int i;
    for(i=0;i<n;i++){
        double t = A[first*n+i];
        A[first*n+i]= A[second*n+i];
        A[second*n+i] =t;
    }
}
#ifdef __AVX512F__
void swapRow3(double *A, int n, int first, int second) {
    int i;

    // Pointers to the beginning of the rows
    double* row1 = A + first * n;
    double* row2 = A + second * n;

    // Process 8 doubles (512 bits) at a time
    for (i = 0; i < n; i += 16) {

        // Load 8 doubles from each row into AVX-512 registers
        __m512d vec1 = _mm512_load_pd(&row1[i]);
        __m512d vec2 = _mm512_load_pd(&row2[i]);
        _mm512_store_pd(&row1[i], vec2);
        _mm512_store_pd(&row2[i], vec1);
        __m512d vec3 = _mm512_load_pd(&row1[i+8]);
        __m512d vec4 = _mm512_load_pd(&row2[i+8]);
        // Swap the contents of the two rows

        _mm512_store_pd(&row1[i+8], vec4);
        _mm512_store_pd(&row2[i+8], vec3);

    }
}
void swapRow2(double *A, int n, int first, int second) {
    register int i;

    // Pointers to the beginning of the rows
    double* row1 = A + first * n;
    double* row2 = A + second * n;
PREFETCH(row1, 3);
PREFETCH(row2, 3);
    // Process 8 doubles (512 bits) at a time
    for (i = 0; i < n; i += 8) {
        double * row1p = &row1[i];
        double * row2p = &row2[i];
        // Load 8 doubles from each row into AVX-512 registers
        __m512d vec1 = _mm512_load_pd(row1p);
        __m512d vec2 = _mm512_load_pd(row2p);
        _mm512_store_pd(row1p, vec2);
        _mm512_store_pd(row2p, vec1);
        PREFETCH(row1p+16, 3);
        PREFETCH(row2p+16, 3);
    }
}
#else
void swapRow3(double *A, int n, int first, int second) {
    swapRow(A, n, first, second);
}
void swapRow2(double *A, int n, int first, int second) {
    swapRow(A, n, first, second);
}
#endif

int mydgetrf(double *A,int *ipiv,int n)
{
    int i=0,t;
    int maxind;
    double max;
    for (i=0;i<n-1;++i){// line 16 of mylu.m
        maxind=i;
        max = fabs(A[i*n+i]);
        for(t=i+1;t<n;++t){
            if( fabs(A[t*n+i] > max)){
                maxind = t;
                max = fabs(A[t*n+i]);//line 21 of mylu.m
            }
        }
        if( fabs(max -0) < 1e-8){//line 24  of mylu.m
            return 0;
        }else{
            if (maxind != i){
                int temp = ipiv[i];
                ipiv[i] = ipiv[maxind];
                ipiv[maxind] = temp;
                swapRow2(A,n,i,maxind); //line 33 of mylu.m
            }
        }
        int j,k;
        for(j=i+1; j<n; ++j){
            A[j*n+i] /=  A[i*n+i];
            for(k=i+1;k<n;++k){
                A[j*n+k] -= A[j*n+i]*A[i*n+k];
            }
        }
    }
    //The return value (an integer) can be 0 or 1
    //If 0, the matrix is irreducible and the result will be ignored
    //If 1, the result is valid
    return 1;
}
// ddot - double sum of dot product
double myddot(int n, const double *x,  const double *y){
    register double sum = 0.0;
    register int i;
    for(i=0;i<n;++i){
        sum += x[i]*y[i];
    }
    return sum;
}
void forward(double *A,double *B,int n,int *ipiv){
    double * y = (double*)malloc(n*sizeof(double));
    int i;
        /*
    y(1) = b(pvt(1));
    for i = 2 : n,
        y(i) = b(pvt(i)) - sum( y(1:i-1) .* A(i,1:i-1) );
    end
     */
    y[0] = B[ipiv[0]];
    for(i=1;i<n;++i){
        y[i] = B[ipiv[i]]- myddot(i,y, &A[i*n]);
    }
    memcpy(B,y,n*sizeof(double));
    free(y);
}
void backward(double *A,double *B,int n,int *ipiv){
    double * x = (double*)malloc(n*sizeof(double));
    int i;
        /*
    y(1) = b(pvt(1));
    for i = 2 : n,
        y(i) = b(pvt(i)) - sum( y(1:i-1) .* A(i,1:i-1) );
    end
     */
    double * y = B;
    x[n-1] = y[n-1] /A[(n-1)*n+n-1];
    for(i=n-2;i>=0;--i){
        // x(n) = y(n) / A(n,n);
        //for i = n-1 : -1 : 1,
        //    x(i) = ( y(i) - sum( x(i+1:n) .* A(i, i+1:n) ) ) / A(i,i);
        //end
        //matlab: if n=3, i= 2 1, i+1:n= 3:3 2:3
        // in c if n=3 i= 1 0 len = 1 2
        //
        int len = n-1-i;
        x[i] = (y[i] - myddot(len,&x[i+1], &A[i*n+i+1]))/A[i*n+i];
    }
    for(i=0;i<n;i++){
        B[i] = x[i];
    }
    memcpy(B,x,n*sizeof(double));
    free(x);
}
void mydtrsv(char UPLO,double *A,double *B,int n,int *ipiv)
{
    double * y = (double*)malloc(n*sizeof(double));
    double * orig = y;
    switch(UPLO){
        case 'L':
            forward(A,B,n,ipiv);
            break;
        case 'U':
            backward(A,B,n,ipiv);
            break;
        default:
            break;
    }
    free(orig);
}
void printPivot(int* p,int n){
    for(int i=0;i<n;i++){
            printf("%d ", p[i]);
    }
     puts("");
}

void printM(double* A, int columns,int rows){
    puts("begin matrix");
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            printf("%lf ", A[i*rows+j]);
        }
        puts("");
    }
    puts("end matrix");
}
void my_f(double *A,double *B,int n)
{
    size_t alignment = 32;
    int *ipiv=(int*)_mm_malloc(n*sizeof(int), alignment);
    for (int i=0;i<n;i++)
        ipiv[i]=i;
    //my_block_dgetrf(A,0,n-1,0,n-1,ipiv,n);
    if (mydgetrf(A,ipiv,n)==0)
    {
        printf("LU factoration failed: coefficient matrix is singular.\n");
        return;
    }
#ifdef DEBUGPRINT
    puts("LU");
    printM(A, n,n);
    puts("B");
    printM(B, n,1);
    puts("pivot");
    printPivot(ipiv, n);
#endif
    mydtrsv('L',A,B,n,ipiv);
    #ifdef DEBUGPRINT
    puts("L");
    printM(B, n,1);
#endif
    mydtrsv('U',A,B,n,ipiv);
#ifdef DEBUGPRINT
    puts("U");
    printM(B, n,1);
#endif
    _mm_free(ipiv);
}

#endif