cmake_minimum_required(VERSION 3.10)

# Set project name
project(cs211_hw3 LANGUAGES C CXX)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
# Include directories
include_directories(/home/act-software/mpich-3.2.1/include)

# Link directories
link_directories(/home/act-software/mpich-3.2.1/lib)

# Add executable
add_executable(hw3_main main.c
        func_call.c sieve0.c sieve1.c sieve2.cpp sieve3.cpp)

# Link libraries
target_link_libraries(hw3_main mpi m)
