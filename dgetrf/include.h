#ifndef __INCLUDE_H__
#define __INCLUDE_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
static inline int gettimeofday(struct timeval* tv, void* tz)
{
    (void)tz;
    FILETIME ft;
    unsigned long long t;
    GetSystemTimeAsFileTime(&ft);
    t = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= 116444736000000000ULL;
    tv->tv_sec = (long)(t / 10000000ULL);
    tv->tv_usec = (long)((t % 10000000ULL) / 10ULL);
    return 0;
}
#else
#include <sys/time.h>
#endif
#include <immintrin.h>
#include <math.h>

#ifdef __cplusplus
#define PREFETCH(addr, hint) _mm_prefetch(reinterpret_cast<const char*>(addr), static_cast<_mm_hint>(hint))
#else
#define PREFETCH(addr, hint) _mm_prefetch((const char*)(addr), (enum _mm_hint)(hint))
#endif

#if defined(__has_include)
#  if __has_include(<mkl.h>)
#    include <mkl.h>
#    define DGETRF_HAS_MKL_HEADER 1
#  endif

#  if !defined(DGETRF_HAS_MKL_HEADER) && __has_include(<cblas.h>)
#    include <cblas.h>
#    define DGETRF_HAS_CBLAS_HEADER 1
#  elif !defined(DGETRF_HAS_MKL_HEADER) && __has_include(<mkl_cblas.h>)
#    include <mkl_cblas.h>
#    define DGETRF_HAS_CBLAS_HEADER 1
#  endif

#  if !defined(DGETRF_HAS_MKL_HEADER) && __has_include(<lapacke.h>)
#    include <lapacke.h>
#    define DGETRF_HAS_LAPACKE_HEADER 1
#  elif !defined(DGETRF_HAS_MKL_HEADER) && __has_include(<mkl_lapacke.h>)
#    include <mkl_lapacke.h>
#    define DGETRF_HAS_LAPACKE_HEADER 1
#  endif
#else
#include <cblas.h>
#include <lapacke.h>
#define DGETRF_HAS_CBLAS_HEADER 1
#define DGETRF_HAS_LAPACKE_HEADER 1
#endif

#if !defined(DGETRF_HAS_MKL_HEADER) && (!defined(DGETRF_HAS_CBLAS_HEADER) || !defined(DGETRF_HAS_LAPACKE_HEADER))
#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif

#ifndef lapack_int
typedef int lapack_int;
#endif

#ifndef CblasRowMajor
enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
#endif

#ifdef __cplusplus
extern "C" {
#endif
lapack_int LAPACKE_dgetrf(int matrix_layout, lapack_int m, lapack_int n, double* a, lapack_int lda, lapack_int* ipiv);
void cblas_dtrsv(const int layout, const int Uplo, const int TransA, const int Diag,
                 const int N, const double* A, const int lda, double* X, const int incX);
#ifdef __cplusplus
}
#endif
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
