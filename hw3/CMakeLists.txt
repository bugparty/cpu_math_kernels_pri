cmake_minimum_required(VERSION 3.10)

# Set project name
project(MainProject)

# Specify the C standard
set(CMAKE_C_STANDARD 11)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3")

# Include directories
include_directories(/home/act-software/mpich-3.2.1/include)

# Link directories
link_directories(/home/act-software/mpich-3.2.1/lib)

# Add executable
add_executable(main main.c
        func_call.c sieve0.c sieve1.c sieve2.cpp sieve3.c)

# Link libraries
target_link_libraries(main mpi m)
