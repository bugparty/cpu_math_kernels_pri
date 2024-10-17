#ifndef __MY_BLOCK_C__
#define __MY_BLOCK_C__

#include "include.h"

void mydgemm(double *A,double *B,int n,int bid,int b)
{
    //TODO
    //Implement a matrix multiplication here following dgemm7 in HW1
    //The first matrix is A[(bid+1)*b:n,bid*b:(bid+1)*b]
    //The second matrix is B[bid*b:(bid+1)*b,(bid+1)*b:n]
    //b is the block size for dgetrf
}

int mydgetrf_block(double *A,int *ipiv,int n)
{
    int b=1;//MODIFY
    //TODO
    return 1;
}

void my_block_f(double *A,double *B,int n)
{
    int *ipiv=(int*)malloc(n*sizeof(int));
    for (int i=0;i<n;i++)
        ipiv[i]=i;
    if (mydgetrf_block(A,ipiv,n)==0) 
    {
        printf("LU factoration failed: coefficient matrix is singular.\n");
        return;
    }
    mydtrsv('L',A,B,n,ipiv);
    mydtrsv('U',A,B,n,ipiv);
}

#endif