cmake_minimum_required(VERSION 3.16)

project(cs211_hpc_collection LANGUAGES C CXX)

option(BUILD_HW1 "Build hw1 GEMM benchmarks" ON)
option(BUILD_HW2 "Build hw2 large linear system solvers" ON)
option(BUILD_HW3 "Build hw3 sieve of Eratosthenes" ON)

if(BUILD_HW1)
    add_subdirectory(gemm)
endif()

if(BUILD_HW2)
    add_subdirectory(dgetrf)
endif()

if(BUILD_HW3)
    add_subdirectory(mpi_prime)
endif()
