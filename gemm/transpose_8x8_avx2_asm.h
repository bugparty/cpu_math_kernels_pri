//
// Created by fancy on 2026/3/24.
//

#ifndef CS211_HPC_COLLECTION_TRANSPOSE_8X8_AVX2_ASM_H
#define CS211_HPC_COLLECTION_TRANSPOSE_8X8_AVX2_ASM_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void transpose_8x8_double_avx2(
    const double *src,
    double       *dst,
    ptrdiff_t     src_stride,
    ptrdiff_t     dst_stride);

#ifdef __cplusplus
}
#endif
#endif //CS211_HPC_COLLECTION_TRANSPOSE_8X8_AVX2_ASM_H
