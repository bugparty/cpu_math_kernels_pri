## 2024-11-20 - AVX2 Polynomial Exp requires rigorous clamping
**Learning:** Approximating `exp` using a Taylor polynomial + exponent shifting (`e^x = 2^floor(x/ln2) * poly(...)`) is very fast, but mathematically fragile on large negative inputs. If `x` is less than approximately `-87.3f` (where `exp(x)` becomes zero in fp32), the range reduction integer `m` will be smaller than `-127`. Adding `127` to form the IEEE exponent then underflows, producing negative floats or NaNs instead of `0.0f`.
**Evidence:** The benchmark `verify` pass initially failed and the output showed NaN/inf.
**Action:** Always insert a clamping `_mm256_max_ps` (e.g. `x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));`) when manually reconstructing IEEE-754 floats to guarantee the exponent stays in bounds.
