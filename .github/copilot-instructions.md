# Copilot Instructions

## Project Guidelines
- In CMake files, compiler flags should distinguish between GCC/Clang and MSVC (e.g., avoid using GCC-style optimization flags for MSVC).

## Performance Optimization
- you can try reusing AVX register variables (e.g., reuse `r0`/`r1`) instead of introducing many per-row temporaries in micro-kernel code.