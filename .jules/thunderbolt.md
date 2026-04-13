## 2024-10-24 - AVX2 Vectorized Softmax Implementation

**Learning:** When vectorizing transcendental functions like `exp` in AVX2, standard Horner's method (`p = _mm256_fmadd_ps(p, r, c)`) creates a strict dependency chain bounded by the 4-cycle FMA latency. Estrin's scheme can break this chain and yield higher ILP. Additionally, standard library headers like `<algorithm>` for `std::max` should always be explicitly included even when not strictly required by the current benchmark/compiler, to avoid cross-platform compilation errors.

**Evidence:** The initial scalar `softmax_naive` hovered around 0.6 GFLOP/s, while `softmax_v2` using AVX2 range reduction, Taylor polynomial approximation, and vectorized reduction achieved 4.6 GFLOP/s (~7.6x speedup) on an N=100000 benchmark.

**Action:** In future mathematical kernel implementations with high-degree polynomials, investigate Estrin's scheme for better FMA latency hiding. Always double-check standard include requirements, especially for heavily templated functionality like `<algorithm>`.

## 2024-10-25 - Compiler Intrinsic Semantics

**Learning:** When using compiler intrinsics that map directly to x86 instructions requiring immediate values (like `_mm256_round_ps` requiring an `imm8` for the rounding control mode), you cannot use a dynamically typed local variable (`auto rnd_mode = ...`) without explicitly marking it as `constexpr`. Doing so strips the constant-expression status, causing compilation to fail with `argument must be a constant integer`. It's safer to either pass the macro flags (like `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC`) directly inline, or use `constexpr auto`.

**Evidence:** The initial attempt to clean up the code by extracting the rounding mode to `auto rnd_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;` caused `gcc -mavx2` to throw a hard compile error. Changing it back to inline fixed it immediately.

**Action:** When extracting intrinsic control flags to variables to improve readability, always use `constexpr auto` or `const int` rather than plain `auto`.
