cmake_minimum_required(VERSION 3.10)
project(matlab)
set(CMAKE_CXX_STANDARD 14)

add_compile_options(-march=native)
# Set CMake build type to Release for optimization
#set(CMAKE_BUILD_TYPE Release)

# Set the necessary include directories and linker flags for MKL
include_directories(/act/opt/intel/composer_xe_2013.3.163/mkl/include)
link_directories(/act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64)
link_directories(./extern)
set(LINK_FLAGS "-O3 -march=native -DMKL_ILP64 -lmkl_avx2 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -m64")

# Add the executable target for the main application
add_executable(main main.c)
set_target_properties(main PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

add_executable(test1 test1.c)
set_target_properties(test1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

