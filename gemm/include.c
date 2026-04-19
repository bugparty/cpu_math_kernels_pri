#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
// you can pass  gcc -BLOCK_SIZE=2048 to alter that
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#include "dgemm0.c"
#include "dgemm1.c"
#include "dgemm2.c"
#include "dgemm3.c"
#include "dgemm6_ijk.c"
#include "dgemm6_ikj.c"
#include "dgemm6_jik.c"
#include "dgemm6_jki.c"
#include "dgemm6_kij.c"
#include "dgemm6_kji.c"
#include "dgemm7.c"