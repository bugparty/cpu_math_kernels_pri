## 2024-10-24 - AVX2 Vectorized Softmax Implementation

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, standard Horner's method (`p = _mm256_fmadd_ps(p, r, c)`) creates a strict dependency chain bounded by the 4-cycle FMA latency. Estrin's scheme can break this chain and yield higher ILP. Additionally, standard library headers like `<algorithm>` for `std::max` should always be explicitly included even when not strictly required by the current benchmark/compiler, to avoid cross-platform compilation errors.

**Evidence:** The initial scalar `softmax_naive` hovered around 0.6 GFLOP/s, while `softmax_v2` using AVX2 range reduction, Taylor polynomial approximation, and vectorized reduction achieved 4.6 GFLOP/s (~7.6x speedup) on an N=100000 benchmark.

**Action:** In future mathematical kernel implementations with high-degree polynomials, investigate Estrin's scheme for better FMA latency hiding. Always double-check standard include requirements, especially for heavily templated functionality like `<algorithm>`.
## 2024-04-10 - [AVX2 Softmax Unrolling and In-Register Reduction]
**Learning:** When writing AVX2 softmax kernels, standard vector loops can be bottlenecked by instruction latency, particularly for heavy exponential approximations (`exp256_ps`). Unrolling the loop by 4x and using independent accumulators hides this latency. Furthermore, performing the final sum/max reduction in-register via `_mm256_permute2f128_ps` and `_mm256_shuffle_ps` completely eliminates the scalar bottleneck of extracting to an array, improving throughput significantly.
**Evidence:** `softmax_v3` achieved 5.79 GFLOP/s vs `softmax_v2` at 4.15 GFLOP/s on N=4096 (Fixed Memory mode), a ~39% performance gain.
**Action:** Default to 4x unrolling and in-register horizontal reductions via shuffle when writing bound AVX2 map-reduce kernels instead of vectorizing single loop iterations and dropping to scalar array reductions at the end.
