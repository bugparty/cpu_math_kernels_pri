cmake_minimum_required(VERSION 3.10)
project(gemm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(OPTIMIZATION_LEVEL "-O3" CACHE STRING "Optimization level (e.g., -O1, -O2)")
add_executable(gemm_bench
        dgemm.c
        bench.cpp
        dgemm.h)
target_compile_options(gemm_bench PRIVATE -march=native  ${OPTIMIZATION_LEVEL})