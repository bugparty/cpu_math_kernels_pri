cmake_minimum_required(VERSION 3.16)

project(ml_kernels LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ml_kernel_smoke
    src/kernel_smoke.cpp
)

target_include_directories(ml_kernel_smoke PRIVATE include)

if(MSVC)
    target_compile_options(ml_kernel_smoke PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
else()
    target_compile_options(ml_kernel_smoke PRIVATE -O3 -march=native)
endif()
