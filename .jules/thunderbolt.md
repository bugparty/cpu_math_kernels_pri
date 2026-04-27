## 2024-10-24 - AVX2 Vectorized Softmax Implementation

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, standard Horner's method (`p = _mm256_fmadd_ps(p, r, c)`) creates a strict dependency chain bounded by the 4-cycle FMA latency. Estrin's scheme can break this chain and yield higher ILP. Additionally, standard library headers like `<algorithm>` for `std::max` should always be explicitly included even when not strictly required by the current benchmark/compiler, to avoid cross-platform compilation errors.

**Evidence:** The initial scalar `softmax_naive` hovered around 0.6 GFLOP/s, while `softmax_v2` using AVX2 range reduction, Taylor polynomial approximation, and vectorized reduction achieved 4.6 GFLOP/s (~7.6x speedup) on an N=100000 benchmark.

**Action:** In future mathematical kernel implementations with high-degree polynomials, investigate Estrin's scheme for better FMA latency hiding. Always double-check standard include requirements, especially for heavily templated functionality like `<algorithm>`.
## 2024-04-10 - [AVX2 Softmax Unrolling and In-Register Reduction]
**Learning:** When writing AVX2 softmax kernels, standard vector loops can be bottlenecked by instruction latency, particularly for heavy exponential approximations (`exp256_ps`). Unrolling the loop by 4x and using independent accumulators hides this latency. Furthermore, performing the final sum/max reduction in-register via `_mm256_permute2f128_ps` and `_mm256_shuffle_ps` completely eliminates the scalar bottleneck of extracting to an array, improving throughput significantly.
**Evidence:** `softmax_v3` achieved 5.79 GFLOP/s vs `softmax_v2` at 4.15 GFLOP/s on N=4096 (Fixed Memory mode), a ~39% performance gain.
**Action:** Default to 4x unrolling and in-register horizontal reductions via shuffle when writing bound AVX2 map-reduce kernels instead of vectorizing single loop iterations and dropping to scalar array reductions at the end.
## 2024-10-25 - AVX2 Softmax Unrolling vs. Polynomial Evaluation Schemes

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, breaking the FMA latency chain can be done via Estrin's scheme. However, if the main loop is already explicitly unrolled (e.g., 4x) to interleave multiple independent FMA streams, Horner's method actually outperforms Estrin's. Estrin's scheme creates higher execution port pressure, whereas 4x interleaved Horner's chains saturate the execution units perfectly while naturally hiding the latency. Additionally, replacing the high-latency `_mm256_round_ps` instruction with the sequence `_mm256_cvtepi32_ps(_mm256_cvtps_epi32(x))` achieves round-to-nearest-even with lower latency, improving throughput in range reduction.

**Evidence:** Microbenchmarking `exp256_ps` independently with a 4x unroll loop showed Horner's evaluating in 419ms vs. Estrin's 548ms. Integrating this (`exp256_ps_v2`) into `softmax_v5` resulted in a ~13.8% speedup (5.1 GFLOP/s vs `softmax_v4`'s 4.48 GFLOP/s).

**Action:** When a loop is heavily unrolled to hide FMA latency, default to Horner's scheme rather than Estrin's to reduce instruction count and port pressure. Reserve Estrin's scheme for dependency-bound single-stream calculations. Always use `cvtps_epi32` over `round_ps` if the default MXCSR rounding mode (round-to-nearest) is acceptable.
## 2024-04-24 - AVX2 Max Reduction Optimization
**Learning:** The naive scalar max reduction (`max_naive`) suffers from a strict loop-carried dependency (each element must be compared to the running `current_max`), which limits ILP and severely restricts throughput. By unrolling the loop 4x and vectorizing with AVX2 (`_mm256_max_ps`), multiple independent accumulators can be maintained, allowing the processor to compute 32 elements per loop iteration. The final result can be determined efficiently via an in-register horizontal reduction instead of sequentially extracting elements.
**Evidence:** The benchmark `max_v2` achieved ~2.8-2.9 GFLOP/s vs `max_naive`'s ~0.63 GFLOP/s on N=16384000 (a ~4.5x speedup), confirming that breaking the dependency chain hides execution latency.
**Action:** When implementing scalar reductions (e.g., max, sum) over large arrays, prioritize vectorization with 4x-8x unrolling and multiple independent accumulators to break latency bounds, then merge the accumulators via tree-reduction at the end of the hot loop.
## 2024-10-26 - AVX2 Max Reduction 8x Unrolling

**Learning:** While 4x unrolling breaks some loop-carried dependencies, `_mm256_max_ps` has a 4-cycle latency. A 4x unroll only issues 4 instructions, leaving execution ports idle while waiting for the dependency chain to resolve. Unrolling 8x maintains 8 independent accumulators, perfectly matching the latency and fully saturating the execution ports, transitioning the kernel from latency-bound to throughput-bound.

**Evidence:** Microbenchmarking showed a 2x speedup (99ms -> 49ms) for max_v3 over max_v2 on L1-hot arrays. End-to-end framework benchmarks showed an 8% throughput increase (4.03 -> 4.36 GFLOP/s) on large fixed-memory allocations (N=6553600).

**Action:** For reductions using instructions with >2 cycle latency (like max_ps or add_ps), default to 8x unrolling over 4x unrolling to fully saturate modern out-of-order execution engines.
