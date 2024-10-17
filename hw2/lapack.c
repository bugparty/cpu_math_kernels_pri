#ifndef __LAPACK_C__
#define __LAPACK_C__

#include "include.h"

void lapack_f(double *A,double *B,int n)
{
    int *ipiv=(int*)malloc(n*sizeof(int));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR,n,n,A,n,(long long*)ipiv);
    for (int i=0;i<n;i++)
    {
        double tmp=B[ipiv[i]-1];
        B[ipiv[i]-1]=B[i];
        B[i]=tmp;
    }
    cblas_dtrsv(CblasRowMajor,CblasLower,CblasNoTrans,CblasUnit,n,A,n,B,1);
    cblas_dtrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,B,1);
}

#endif