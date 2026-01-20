cmake_minimum_required(VERSION 3.10)

# Set project name
project(cs211_hw3 LANGUAGES C CXX)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})

# Add executable
add_executable(hw3_main main.c
        func_call.c sieve0.c sieve1.c)

# Link libraries
target_link_libraries(hw3_main ${MPI_C_LIBRARIES} m)
