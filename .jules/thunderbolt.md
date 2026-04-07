## 2024-05-24 - AVX2 Softmax Vectorization
**Learning:** When implementing vectorized math in `ml_kernels` (e.g., AVX2 Softmax) without SVML, use range reduction (`exp(x) = 2^n * 2^f`) alongside Taylor polynomials to maintain accuracy for large negative values, and normalize by multiplying by the inverse sum instead of using scalar division to maximize throughput.
**Evidence:** `softmax_v2` improved performance from 0.56 GFLOPS to 4.10 GFLOPS (~7.3x speedup) on N=1000000 (Fixed Memory).
**Action:** Always use inverse multiplication for normalizations and Taylor expansions with range reduction for transcendental functions in vectorized kernels.
