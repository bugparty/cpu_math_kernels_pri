#pragma once

#ifdef __cplusplus
extern "C" {
#endif
    extern double *B_T;
    void dgemm1(double *C,double *A,double *B,int n);
    void dgemm7(double *C,double *A,double *B,int n);
    void dgemm1T(double *C,double *A,double *B,int n);
    void dgemm3(double *C,double *A,double *B,int n);
    void dgemm3v2(double *C,double *A,double *B,int n);
    void dgemmBT1(double *C,double *A,double *B,int n);
    void dgemmAVX(double *C,double *A,double *B,int n);
#ifdef __AVX512F__
    void dgemmAVX512(double *C,double *A,double *B,int n);
    void dgemmAVX512B(double *C,double *A,double *B,int n);
#endif
    void dgemm71(double *C,double *A,double *B,int n);
    void dgemm72(double *C,double *A,double *B,int n);
    void dgemm74(double *C,double *A,double *B,int n);
    void dgemm7_raw(double *C,double *A,double *B,int n);
    void dgemm7_ijk(double *C,double *A,double *B,int n);
    void dgemm7_kij(double *C,double *A,double *B,int n);
    void dgemm7_ikj(double *C,double *A,double *B,int n);
    void dgemm7_ikj_v2(double *C,double *A,double *B,int n);
#ifdef __AVX512F__
    void dgemm7(double *C,double *A,double *B,int n);
    void dgemm7_v2(double *C,double *A,double *B,int n);
#endif
#ifdef __AVX2__
    void dgemm7_v2_avx2(double *C,double *A,double *B,int n);
#endif

    void transpose( const double  * const A, double * const  A_T, int n);
#ifdef __cplusplus
}
#endif
