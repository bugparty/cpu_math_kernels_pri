cmake_minimum_required(VERSION 3.10)
project(gemm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(OPTIMIZATION_LEVEL "-O3" CACHE STRING "Optimization level (e.g., -O1, -O2)")
add_executable(gemm_bench
        dgemm.cpp
        bench.cpp
        dgemm.h)
target_compile_options(gemm_bench PRIVATE -march=native  ${OPTIMIZATION_LEVEL})

add_executable(gemm_bench_all
        dgemm.cpp
        bench_all.cpp
        dgemm.h
        dgemm7olds.h)
target_compile_options(gemm_bench_all PRIVATE -march=native ${OPTIMIZATION_LEVEL})

add_executable(transpose_bench
        transpose_bench.cpp
        common.h
        transpose_avx512.h
        transpose_recursive.h
        transpose_prefetch.h
        transpose_nontemporal.h
        transpose_2level.h)
target_compile_options(transpose_bench PRIVATE -march=native ${OPTIMIZATION_LEVEL})