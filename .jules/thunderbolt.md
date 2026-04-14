## 2024-05-18 - In-register Tree Reduction vs. Scalar Fallback
**Learning:** Extracting lanes from a SIMD register (via `_mm256_storeu_ps` to array) and doing a scalar reduction adds unnecessary latency. An in-register reduction via shuffles (`_mm_max_ps`, `_mm_add_ps` along with `_mm_movehl_ps` and `_mm_shuffle_ps`) scales much better, especially when combining it with multiple accumulators.
**Evidence:** Throughput improved, and it removed explicit scalar loop fallback code.
**Action:** Use `hmax256_ps` and `hsum256_ps` wrappers utilizing `_mm256_castps256_ps128` and `_mm256_extractf128_ps` whenever cross-lane summation/max is required.
