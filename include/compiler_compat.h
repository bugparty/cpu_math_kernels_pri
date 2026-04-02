#pragma once

#if defined(_MSC_VER)
#ifndef __restrict__
#define __restrict__ __restrict
#endif

#ifndef __builtin_expect
#define __builtin_expect(x, expected) (x)
#endif

#ifndef __builtin_prefetch
#include <xmmintrin.h>
#define __builtin_prefetch(addr, rw, locality) \
    _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#endif
#endif
