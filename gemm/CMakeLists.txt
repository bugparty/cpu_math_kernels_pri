cmake_minimum_required(VERSION 3.10)
project(gemm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

if(MSVC)
    set(_DEFAULT_OPTIMIZATION_LEVEL "/O2")
    set(_DEFAULT_ARCH_FLAGS "/arch:AVX2")
    get_filename_component(_MSVC_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_MSVC_HOST_BIN_DIR "${_MSVC_COMPILER_BIN_DIR}" DIRECTORY)
    get_filename_component(_MSVC_BIN_DIR "${_MSVC_HOST_BIN_DIR}" DIRECTORY)
    get_filename_component(_MSVC_TOOL_ROOT "${_MSVC_BIN_DIR}" DIRECTORY)
    set(_MSVC_TOOL_INCLUDE_DIR "${_MSVC_TOOL_ROOT}/include")
    file(GLOB _WINDOWS_KITS_INCLUDE_VERSIONS LIST_DIRECTORIES TRUE "C:/Program Files (x86)/Windows Kits/10/Include/*")
    list(SORT _WINDOWS_KITS_INCLUDE_VERSIONS)
    list(REVERSE _WINDOWS_KITS_INCLUDE_VERSIONS)
    foreach(_windows_kits_version_dir IN LISTS _WINDOWS_KITS_INCLUDE_VERSIONS)
        if(EXISTS "${_windows_kits_version_dir}/ucrt/crtdbg.h")
            set(_WINDOWS_KITS_INCLUDE_DIRS
                "${_windows_kits_version_dir}/ucrt"
                "${_windows_kits_version_dir}/shared"
                "${_windows_kits_version_dir}/um"
                "${_windows_kits_version_dir}/winrt"
                "${_windows_kits_version_dir}/cppwinrt")
            break()
        endif()
    endforeach()
else()
    set(_DEFAULT_OPTIMIZATION_LEVEL "-O3")
    set(_DEFAULT_ARCH_FLAGS "-march=native")
endif()
set(OPTIMIZATION_LEVEL "${_DEFAULT_OPTIMIZATION_LEVEL}" CACHE STRING "Optimization level (MSVC: /O2, GCC/Clang: -O3)")
set(ARCH_FLAGS "${_DEFAULT_ARCH_FLAGS}" CACHE STRING "Architecture flags (MSVC: /arch:AVX2, GCC/Clang: -march=native)")

# Optional Intel MKL support (used by gemm_bench_all / transpose_bench)
set(GEMM_HAVE_MKL FALSE)
set(GEMM_MKL_ROOT "" CACHE PATH "Path to Intel MKL root directory")
set(GEMM_MKL_INCLUDE_DIR "")
set(GEMM_MKL_LIB_DIR "")
set(GEMM_MKL_LIBS "")

# Detection priority:
# 1) explicit GEMM_MKL_ROOT cache
# 2) MKL_ROOT cache from parent/project
# 3) environment variables MKLROOT / ONEAPI_ROOT
# 4) common default oneAPI install paths on Windows
if(NOT GEMM_MKL_ROOT)
    if(DEFINED MKL_ROOT AND MKL_ROOT)
        set(GEMM_MKL_ROOT "${MKL_ROOT}" CACHE PATH "Path to Intel MKL root directory" FORCE)
    elseif(WIN32 AND DEFINED ENV{MKLROOT})
        set(GEMM_MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "Path to Intel MKL root directory" FORCE)
    elseif(WIN32 AND DEFINED ENV{ONEAPI_ROOT})
        set(GEMM_MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest" CACHE PATH "Path to Intel MKL root directory" FORCE)
    elseif(WIN32)
        set(_GEMM_MKL_CANDIDATES
            "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
            "C:/Program Files/Intel/oneAPI/mkl/latest")
        foreach(_mkl_candidate IN LISTS _GEMM_MKL_CANDIDATES)
            if(EXISTS "${_mkl_candidate}/include/mkl.h")
                set(GEMM_MKL_ROOT "${_mkl_candidate}" CACHE PATH "Path to Intel MKL root directory" FORCE)
                break()
            endif()
        endforeach()
    endif()
endif()

if(GEMM_MKL_ROOT AND (EXISTS "${GEMM_MKL_ROOT}/include/mkl.h" OR EXISTS "${GEMM_MKL_ROOT}/include/mkl/mkl.h"))
    set(GEMM_HAVE_MKL TRUE)
    if(EXISTS "${GEMM_MKL_ROOT}/include/mkl.h")
        set(GEMM_MKL_INCLUDE_DIR "${GEMM_MKL_ROOT}/include")
    else()
        set(GEMM_MKL_INCLUDE_DIR "${GEMM_MKL_ROOT}/include/mkl")
    endif()

    if(MSVC)
        if(EXISTS "${GEMM_MKL_ROOT}/lib/intel64")
            set(GEMM_MKL_LIB_DIR "${GEMM_MKL_ROOT}/lib/intel64")
        else()
            set(GEMM_MKL_LIB_DIR "${GEMM_MKL_ROOT}/lib")
        endif()
        set(GEMM_MKL_LIBS mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib)
    else()
        if(EXISTS "${GEMM_MKL_ROOT}/lib/intel64")
            set(GEMM_MKL_LIB_DIR "${GEMM_MKL_ROOT}/lib/intel64")
        else()
            set(GEMM_MKL_LIB_DIR "${GEMM_MKL_ROOT}/lib")
        endif()
        set(GEMM_MKL_LIBS mkl_intel_lp64 mkl_sequential mkl_core)
    endif()

    message(STATUS "GEMM/Transpose benchmark: Intel MKL enabled (${GEMM_MKL_ROOT})")
else()
    message(STATUS "GEMM/Transpose benchmark: Intel MKL not found, MKL kernels disabled")
endif()

add_executable(gemm_bench
        dgemm.cpp
        bench.cpp
        dgemm.h
        transpose_8x8_avx2_asm.h)
if(MSVC)
    target_compile_options(gemm_bench PRIVATE ${ARCH_FLAGS} $<$<NOT:$<CONFIG:Debug>>:${OPTIMIZATION_LEVEL}>)
    if(EXISTS "${_MSVC_TOOL_INCLUDE_DIR}/immintrin.h")
        target_include_directories(gemm_bench PRIVATE "${_MSVC_TOOL_INCLUDE_DIR}")
    endif()
    if(_WINDOWS_KITS_INCLUDE_DIRS)
        target_include_directories(gemm_bench PRIVATE ${_WINDOWS_KITS_INCLUDE_DIRS})
    endif()
else()
    target_compile_options(gemm_bench PRIVATE ${ARCH_FLAGS} ${OPTIMIZATION_LEVEL})
endif()

add_executable(gemm_bench_all
        dgemm.cpp
        dgemm7v2.cpp
        bench_all.cpp
        dgemm.h
        dgemm7olds.h)
if(MSVC)
    target_compile_options(gemm_bench_all PRIVATE ${ARCH_FLAGS} $<$<NOT:$<CONFIG:Debug>>:${OPTIMIZATION_LEVEL}>)
    if(EXISTS "${_MSVC_TOOL_INCLUDE_DIR}/immintrin.h")
        target_include_directories(gemm_bench_all PRIVATE "${_MSVC_TOOL_INCLUDE_DIR}")
    endif()
    if(_WINDOWS_KITS_INCLUDE_DIRS)
        target_include_directories(gemm_bench_all PRIVATE ${_WINDOWS_KITS_INCLUDE_DIRS})
    endif()
else()
    target_compile_options(gemm_bench_all PRIVATE ${ARCH_FLAGS} ${OPTIMIZATION_LEVEL})
endif()

target_compile_definitions(gemm_bench_all PRIVATE GEMM_HAVE_MKL=$<IF:$<BOOL:${GEMM_HAVE_MKL}>,1,0>)
if(GEMM_HAVE_MKL)
    target_include_directories(gemm_bench_all PRIVATE "${GEMM_MKL_INCLUDE_DIR}")
    if(GEMM_MKL_LIB_DIR)
        target_link_directories(gemm_bench_all PRIVATE "${GEMM_MKL_LIB_DIR}")
    endif()
    target_link_libraries(gemm_bench_all PRIVATE ${GEMM_MKL_LIBS})
endif()

add_executable(transpose_bench
        transpose_bench.cpp
        common.h
        transpose_avx512.h
        transpose_recursive.h
        transpose_prefetch.h
        transpose_nontemporal.h
        transpose_2level.h)
if(MSVC)
    target_compile_options(transpose_bench PRIVATE ${ARCH_FLAGS} $<$<NOT:$<CONFIG:Debug>>:${OPTIMIZATION_LEVEL}>)
    if(EXISTS "${_MSVC_TOOL_INCLUDE_DIR}/immintrin.h")
        target_include_directories(transpose_bench PRIVATE "${_MSVC_TOOL_INCLUDE_DIR}")
    endif()
    if(_WINDOWS_KITS_INCLUDE_DIRS)
        target_include_directories(transpose_bench PRIVATE ${_WINDOWS_KITS_INCLUDE_DIRS})
    endif()
else()
    target_compile_options(transpose_bench PRIVATE ${ARCH_FLAGS} ${OPTIMIZATION_LEVEL})
endif()

target_compile_definitions(transpose_bench PRIVATE GEMM_HAVE_MKL=$<IF:$<BOOL:${GEMM_HAVE_MKL}>,1,0>)
if(GEMM_HAVE_MKL)
    target_include_directories(transpose_bench PRIVATE "${GEMM_MKL_INCLUDE_DIR}")
    if(GEMM_MKL_LIB_DIR)
        target_link_directories(transpose_bench PRIVATE "${GEMM_MKL_LIB_DIR}")
    endif()
    target_link_libraries(transpose_bench PRIVATE ${GEMM_MKL_LIBS})
endif()
