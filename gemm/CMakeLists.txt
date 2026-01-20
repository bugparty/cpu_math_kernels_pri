cmake_minimum_required(VERSION 3.10)
project(hpc_playground)
set(CMAKE_CXX_STANDARD 14)
set(OPTIMIZATION_LEVEL "" CACHE STRING "Optimization level (e.g., -O1, -O2)")
add_executable(bench
        dgemm.c
        bench.cpp
        dgemm.h)
target_compile_options(bench PRIVATE -march=native  ${OPTIMIZATION_LEVEL})