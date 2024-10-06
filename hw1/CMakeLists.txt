cmake_minimum_required(VERSION 3.10)

project(DGEMM_Project C)

set(SOURCE_FILES
    dgemm0.c
    dgemm1.c
    dgemm2.c
    dgemm3.c
    dgemm7.c
    dgemm6_ijk.c
    dgemm6_ikj.c
    dgemm6_jik.c
    dgemm6_jki.c
    dgemm6_kij.c
    dgemm6_kji.c
    main.c
)

add_executable(main main.c)
