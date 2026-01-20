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
    void dgemmAVX512(double *C,double *A,double *B,int n);
    void dgemmAVX512B(double *C,double *A,double *B,int n);
    void dgemmAVX_T_B16(double *C,double *A,double *B,int n);
    void dgemmAVX512B2(double *C,double *A,double *B,int n);
    void dgemm7_ijk(double *C,double *A,double *B,int n);
    void dgemm7_kij(double *C,double *A,double *B,int n);
    void dgemm7_ikj(double *C,double *A,double *B,int n);
    void dgemm7(double *C,double *A,double *B,int n);

    void transpose( const double  * const A, double * const  A_T, int n);
#ifdef __cplusplus
}
#endif
