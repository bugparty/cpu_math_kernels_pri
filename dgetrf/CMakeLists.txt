cmake_minimum_required(VERSION 3.10)
project(cs211_hw2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

add_compile_options(-march=native)
# Set CMake build type to Release for optimization
#set(CMAKE_BUILD_TYPE Release)

# Try to find OpenBLAS as alternative to MKL
find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
    set(LINK_FLAGS "-O3 -march=native")
    set(MATH_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    # Fallback to manual OpenBLAS
    set(LINK_FLAGS "-O3 -march=native")
    set(MATH_LIBS "-lopenblas -llapack")
endif()

# Add the executable target for the main application
add_executable(hw2_main main.c)
set_target_properties(hw2_main PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
target_link_libraries(hw2_main ${MATH_LIBS} m pthread)

add_executable(hw2_test1 test1.c)
set_target_properties(hw2_test1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
target_link_libraries(hw2_test1 ${MATH_LIBS} m pthread)

