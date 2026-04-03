cmake_minimum_required(VERSION 3.16)

project(ml_kernels LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ml_kernel_smoke
    src/naive_ops.cpp
    src/kernel_smoke.cpp
)

target_include_directories(ml_kernel_smoke PRIVATE include)

add_executable(ml_kernel_bench
    src/naive_ops.cpp
    src/kernel_bench.cpp
        include/ml_kernels/relu.h
)

target_include_directories(ml_kernel_bench PRIVATE include ${CMAKE_SOURCE_DIR}/include)

add_executable(ml_kernel_test
    src/naive_ops.cpp
    src/kernel_test.cpp
)

target_include_directories(ml_kernel_test PRIVATE include)

if(MSVC)
    target_compile_options(ml_kernel_smoke PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
    target_compile_options(ml_kernel_bench PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
    target_compile_options(ml_kernel_test PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
else()
    target_compile_options(ml_kernel_smoke PRIVATE -O3 -march=native)
    target_compile_options(ml_kernel_bench PRIVATE -O3 -march=native)
    target_compile_options(ml_kernel_test PRIVATE -O3 -march=native)
endif()
