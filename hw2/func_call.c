#ifndef __FUNC_CALL_C__
#define __FUNC_CALL_C__

#include "lapack.c"
#include "my.c"
#include "my_block.c"

void func_call(char *func_name,double *A,double *B,int n)
{
    if (strcmp(func_name,"lapack")==0)
    {
        lapack_f(A,B,n);
        return; 
    }
    if (strcmp(func_name,"my")==0)
    {
        my_f(A,B,n);
        return;
    }
    if (strcmp(func_name,"my_block")==0)
    {
        my_block_f(A,B,n);
        return;
    }
    printf("Error: Invalid function name\n");
    exit(0);
}

#endif