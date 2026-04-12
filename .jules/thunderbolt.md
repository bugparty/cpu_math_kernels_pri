## 2024-10-24 - AVX2 Vectorized Softmax Implementation

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, standard Horner's method (`p = _mm256_fmadd_ps(p, r, c)`) creates a strict dependency chain bounded by the 4-cycle FMA latency. Estrin's scheme can break this chain and yield higher ILP. Additionally, standard library headers like `<algorithm>` for `std::max` should always be explicitly included even when not strictly required by the current benchmark/compiler, to avoid cross-platform compilation errors.

**Evidence:** The initial scalar `softmax_naive` hovered around 0.6 GFLOP/s, while `softmax_v2` using AVX2 range reduction, Taylor polynomial approximation, and vectorized reduction achieved 4.6 GFLOP/s (~7.6x speedup) on an N=100000 benchmark.

**Action:** In future mathematical kernel implementations with high-degree polynomials, investigate Estrin's scheme for better FMA latency hiding. Always double-check standard include requirements, especially for heavily templated functionality like `<algorithm>`.

## 2024-10-25 - FMA Compiler Flags and Tree Reductions

**Learning:** When compiling test harnesses containing FMA intrinsics (like `_mm256_fmadd_ps`) outside of the main CMake setup (e.g., using raw `g++`), explicitly passing `-mfma` along with `-mavx2` is required. Omitting it leads to GCC throwing `inlining failed in call to 'always_inline' '_mm256_fmadd_ps': target specific option mismatch` errors. Additionally, replacing scalar memory reductions with 4x loop unrolling and in-register tree reductions (`_mm256_permute2f128_ps`, `_mm256_permute_ps`) for `max` and `sum` yields a consistent ~20-30% throughput gain over simple vectorized versions that fall back to memory arrays.

**Evidence:** A benchmark script tracking `softmax_v2` (array reduction) vs `softmax_v3` (in-register reduction + 4x unroll) showed `softmax_v3` reaching 4.98 GFLOP/s compared to 3.02 GFLOP/s for `softmax_v2` on N=100k. GCC 13 failed to compile `test_estrin.cpp` without `-mfma`.

**Action:** Always include `-mfma` when manually compiling AVX2 + FMA code to verify microbenchmarks. Always aggressively target horizontal reductions; push them into register-level shuffles and unroll main loops to hide FMA latency instead of spilling vectors to arrays.
