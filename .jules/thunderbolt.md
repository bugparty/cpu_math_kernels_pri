## 2024-10-24 - AVX2 Vectorized Softmax Implementation

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, standard Horner's method (`p = _mm256_fmadd_ps(p, r, c)`) creates a strict dependency chain bounded by the 4-cycle FMA latency. Estrin's scheme can break this chain and yield higher ILP. Additionally, standard library headers like `<algorithm>` for `std::max` should always be explicitly included even when not strictly required by the current benchmark/compiler, to avoid cross-platform compilation errors.

**Evidence:** The initial scalar `softmax_naive` hovered around 0.6 GFLOP/s, while `softmax_v2` using AVX2 range reduction, Taylor polynomial approximation, and vectorized reduction achieved 4.6 GFLOP/s (~7.6x speedup) on an N=100000 benchmark.

**Action:** In future mathematical kernel implementations with high-degree polynomials, investigate Estrin's scheme for better FMA latency hiding. Always double-check standard include requirements, especially for heavily templated functionality like `<algorithm>`.
