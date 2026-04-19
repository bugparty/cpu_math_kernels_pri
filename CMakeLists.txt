cmake_minimum_required(VERSION 3.16)

project(cs211_hpc_collection LANGUAGES C CXX)

# Intel MKL support for Windows (OneAPI)
if(WIN32)
    # Try to find Intel MKL from OneAPI installation
    if(DEFINED ENV{MKLROOT})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}/cmake")
    endif()
    if(DEFINED ENV{ONEAPI_ROOT})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ONEAPI_ROOT}/mkl/latest/cmake")
    endif()
endif()

option(BUILD_HW1 "Build hw1 GEMM benchmarks" ON)
option(BUILD_HW2 "Build hw2 large linear system solvers" ON)
option(BUILD_HW3 "Build hw3 sieve of Eratosthenes" OFF)
option(BUILD_ML_KERNELS "Build experimental ML kernels workspace" ON)

include(CTest)

if(BUILD_HW1)
    add_subdirectory(gemm)
endif()

if(BUILD_HW2)
    add_subdirectory(dgetrf)
endif()

if(BUILD_HW3)
    add_subdirectory(mpi_prime)
endif()

if(BUILD_ML_KERNELS)
    add_subdirectory(ml_kernels)
endif()
