#ifndef __INCLUDE_H__
#define __INCLUDE_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#ifdef __cplusplus
extern "C" {
#endif

void sieve0(unsigned long long *global_count, unsigned long long n, int pnum, int pid);

void sieve1(uint64_t *global_count, uint64_t n, int pnum, int pid);

void sieve2(uint64_t *global_count, uint64_t n, int pnum, int pid);

void sieve3(unsigned long long *global_count, unsigned long long n, int pnum, int pid);

#ifdef __cplusplus
}
#endif
#endif