#include <iostream>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <string>
#include <thread>
#include <chrono>
#include <immintrin.h>
#include "dgemm.h"
using namespace std;
std::unordered_map<std::string, std::function<void (double *C,double *A,double *B,int n)>> funcs;

#define REGISTER_GEMM_FUNC(func_name) funcs[#func_name] = func_name;

void init()
{
    REGISTER_GEMM_FUNC(dgemmAVX);
    REGISTER_GEMM_FUNC(dgemmAVX512);
    REGISTER_GEMM_FUNC(dgemm7);
    REGISTER_GEMM_FUNC(dgemm7_ijk);
    REGISTER_GEMM_FUNC(dgemm7_kij);
    REGISTER_GEMM_FUNC(dgemm7_ikj);
    REGISTER_GEMM_FUNC(dgemmAVX512B);
    //REGISTER_GEMM_FUNC(dgemmAVX512B2);
}

void func_call(const string& func, double *C,double *A,double *B,int n){
    auto it = funcs.find(func);
    if (it != funcs.end())
    {
        it->second(C,A,B,n);
    }else
    {
        cout << "the function " << func << "does not exist"<<endl;
        exit(-1);
    }
}
double *B_T;
void bench_tranpose( double *A,int n)
{
    auto start = std::chrono::steady_clock::now();
    transpose(A,B_T, n);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("tranpose time=%lfs\n",elapsed.count()/ 1000.0);
}
void debugFill(double *A,double*A_backup,double*B,double*B_backup,
    double*C,double*C_backup,int n)
{
    int i;
    for (i=0;i<n*n;i++)
    {
        A_backup[i]=1;
        A[i]=A_backup[i];
        B_backup[i]=i;
        B[i]=B_backup[i];
        C_backup[i]=1;
        C[i]=C_backup[i];
    }
}

int main(int argc, char **argv)
{
    init();
    if (argc!=4)
    {
        printf("Usage: ./main -func -n -pad\n");
        exit(0);
    }
    char *func_name=argv[1];
    int n=atoi(argv[2]);
    int pad=atoi(argv[3]);
    n=((n+pad-1)/pad)*pad;
    printf("n=%d\n",n);
    int i,t;
    size_t alignment = 32;
    double *A_backup=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *B_backup=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *C_backup=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *A=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *B=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    B_T =  (double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *C=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    srand(time(NULL));
    if (n<=16)
    {
        debugFill(A,A_backup,B,B_backup,C,C_backup,n);
    }else
    {
        for (i=0;i<n*n;i++)
        {
            A_backup[i]=((double)rand()/RAND_MAX)*2-1;
            A[i]=A_backup[i];
            B_backup[i]=((double)rand()/RAND_MAX)*2-1;
            B[i]=B_backup[i];
            C_backup[i]=((double)rand()/RAND_MAX)*2-1;
            C[i]=C_backup[i];
        }
    }
    //bench_tranpose(A,n);
    auto start = std::chrono::steady_clock::now();
    func_call(func_name,C,A,B,n);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("time=%lfs\n",elapsed.count()/ 1000.0);
    for (t=0;t<10;t++)
    {
        int i=rand()%n;
        int j=rand()%n;
        int k;
        double standard=C_backup[i*n+j];
        for (k=0;k<n;k++)
            standard+=A_backup[i*n+k]*B_backup[k*n+j];
        if (fabs(C[i*n+j]-standard)>1e-5)
            printf("Error at (%d,%d): standard=%lf, output=%lf\n",i,j,standard,C[i*n+j]);
    }
    _mm_free(A);_mm_free(A_backup);
    _mm_free(B);_mm_free(B_backup);_mm_free(B_T);
    _mm_free(C);_mm_free(C_backup);


}