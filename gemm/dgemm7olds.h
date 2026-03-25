#pragma once

#if defined(_MSC_VER) && !defined(__builtin_prefetch)
#include <xmmintrin.h>
#define __builtin_prefetch(addr, rw, locality) \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#endif

template<int BLOCK_SIZE, int STRIDE>
void kernel_R4x4(double *C, double *A, double *B, int n, int i, int j, int k)
{
    register int ii, jj, kk;
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

template<int BLOCK_SIZE, int STRIDE>
void dgemm74_template(double*C, double*A, double*B, int n){
    int i, j, k;
    for (k = 0; k < n; k += BLOCK_SIZE)
        for (i = 0; i < n; i += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_R4x4<BLOCK_SIZE, STRIDE>(C, A, B, n, i, j, k);
            }
}

template<int BLOCK_SIZE, int STRIDE>
void dgemm7_raw_template(double *C, double *A, double *B, int n)
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

template<int BLOCK_SIZE, int STRIDE>
void dgemm7_ijk_template(double *C, double *A, double *B, int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (j = 0; j < n; j += BLOCK_SIZE)
            for (k = 0; k < n; k += BLOCK_SIZE)
            {
                kernel_R4x4<BLOCK_SIZE, STRIDE>(C, A, B, n, i, j, k);
            }
}

template<int BLOCK_SIZE, int STRIDE>
void dgemm7_kij_template(double *C, double *A, double *B, int n)
{
    int i, j, k;
    for (k = 0; k < n; k += BLOCK_SIZE)
        for (i = 0; i < n; i += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_R4x4<BLOCK_SIZE, STRIDE>(C, A, B, n, i, j, k);
            }
}

template<int BLOCK_SIZE, int STRIDE>
void dgemm7_ikj_template(double *C, double *A, double *B, int n)
{
    int i, j, k;
    for (i = 0; i < n; i += BLOCK_SIZE)
        for (k = 0; k < n; k += BLOCK_SIZE)
            for (j = 0; j < n; j += BLOCK_SIZE)
            {
                kernel_R4x4<BLOCK_SIZE, STRIDE>(C, A, B, n, i, j, k);
            }
}

template<int BLOCK_SIZE, int STRIDE>
void dgemm71_template(double *C, double *A, double *B, int n)
{
    int i, j, k;
    register int ii, jj, kk;
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
                        __builtin_prefetch(&C[t], 1, 3);
                        __builtin_prefetch(&C[tt], 1, 3);
                        __builtin_prefetch(&C[ttt], 1, 3);
                        __builtin_prefetch(&C[tttt], 1, 3);
                        register double c00 = C[t], c01 = C[t + 1], c02 = C[t + 2], c03 = C[t + 3];
                        register double c10 = C[tt], c11 = C[tt + 1], c12 = C[tt + 2], c13 = C[tt + 3];
                        register double c20 = C[ttt], c21 = C[ttt + 1], c22 = C[ttt + 2], c23 = C[ttt + 3];
                        register double c30 = C[tttt], c31 = C[tttt + 1], c32 = C[tttt + 2], c33 = C[tttt + 3];

                        for(kk = k; kk < (k + BLOCK_SIZE); kk += STRIDE)
                        {
                            register int ta = ii*n + kk;
                            __builtin_prefetch(&A[ta], 0, 3);
                            __builtin_prefetch(&A[ta+n], 0, 3);
                            __builtin_prefetch(&A[ta+2*n], 0, 3);
                            __builtin_prefetch(&A[ta+3*n], 0, 3);
                            register int tta = ta + n;
                            register int ttta = tta + n;
                            register int tttta = ttta + n;
                            register double a00 = A[ta];
                            register double a10 = A[tta];
                            register double a20 = A[ttta];
                            register double a30 = A[tttta];
                            register int tb = kk*n + jj;
                            __builtin_prefetch(&B[tb], 0, 3);
                            __builtin_prefetch(&B[tb+n], 0, 3);
                            __builtin_prefetch(&B[tb+2*n], 0, 3);
                            __builtin_prefetch(&B[tb+3*n], 0, 3);

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

template<int BLOCK_SIZE, int STRIDE>
void dgemm72_template(double *C, double *A, double *B, int n)
{
    int i, j, k;
    register int ii, jj, kk;

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
                            register double a00 = A[ta];
                            register double a10 = A[tta];
                            register double a20 = A[ttta];
                            register double a30 = A[tttta];

                            register int tb = kk*n + jj;
                            register double b00 = B[tb];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            register double b01 = B[tb + 1];
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            register double b02 = B[tb + 2];
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            register double  b03 = B[tb + 3];
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

// C wrapper functions with default BLOCK_SIZE=16, STRIDE=4
// Note: dgemm7_ijk, dgemm7_kij, dgemm7_ikj are already defined in dgemm.cpp
extern "C" {
    void dgemm74(double*C, double*A, double*B, int n) {
        dgemm74_template<16, 4>(C, A, B, n);
    }

    void dgemm7_raw(double *C, double *A, double *B, int n) {
        dgemm7_raw_template<16, 4>(C, A, B, n);
    }

    void dgemm71(double *C, double *A, double *B, int n) {
        dgemm71_template<16, 4>(C, A, B, n);
    }

    void dgemm72(double *C, double *A, double *B, int n) {
        dgemm72_template<16, 4>(C, A, B, n);
    }
}
