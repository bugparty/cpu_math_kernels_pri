# Agent-Optimized Transpose Kernels for AI Infrastructure

This repository demonstrates a focused idea: software agents can help discover high-performance CPU kernels for the memory-layout operations underneath AI systems. The current showcase is matrix transpose, a simple-looking operation that often decides whether downstream matrix multiply, attention, and tensor-processing code reads memory efficiently or stalls on data movement.

The benchmark target is Intel MKL `mkl_domatcopy`, Intel's production-grade matrix copy and transpose routine. In the captured run below, the best agent-generated transpose variants outperform both a naive transpose and MKL's direct transpose primitive on the tested shapes.

## Why Investors Should Care

AI infrastructure is constrained by more than raw accelerator FLOPs. Data has to be rearranged, copied, batched, normalized, and fed into compute-heavy kernels. Those "small" CPU-side primitives can become latency and cost bottlenecks at scale.

Transpose is a useful proof point because it is memory-bound, easy to verify, and hard to make fast without understanding caches, vector instructions, prefetch behavior, and write patterns. If an agent can repeatedly search this space and produce faster kernels, that points toward a broader opportunity: automated performance engineering for the invisible math layer that AI systems rely on.

## Performance Snapshot

Benchmark configuration:

- Command shape: `iters=3`, `warmup=1`, `sizes=96,480,960,1920`
- Benchmark driver: `gemm/transpose_bench.cpp`
- Benchmark framework: `include/benchmark.h`
- Metric: GB/s of effective transpose bandwidth, higher is better.
- Verification: every reported row below passed correctness checks.

Pool Mode rotates through a memory pool and is a closer proxy for repeated workload execution. The 1920 row is conservative because the pasted 1920 output is truncated after the early AVX2 rows; it uses the best complete custom row visible in that captured section.

| Matrix size | Naive transpose | Intel MKL `mkl_domatcopy` | Best custom kernel in captured output | Speedup vs naive | Speedup vs MKL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 x 96 | 4.56 GB/s | 2.83 GB/s | 31.86 GB/s (`2level_tuned_avx2_nt_pf`) | 7.0x | 11.3x |
| 480 x 480 | 5.27 GB/s | 4.01 GB/s | 28.85 GB/s (`2level_tuned_avx2_nt_pf_256_64`) | 5.5x | 7.2x |
| 960 x 960 | 3.57 GB/s | 4.13 GB/s | 22.05 GB/s (`2level_tuned_avx2_nt_pf`) | 6.2x | 5.3x |
| 1920 x 1920 | 4.70 GB/s | 5.24 GB/s | 18.11 GB/s (`2level_tuned_256_64`) | 3.9x | 3.5x |

Fixed Memory mode isolates kernel execution more aggressively. On complete captured Fixed Memory rows, the custom kernels reached:

| Matrix size | Intel MKL `mkl_domatcopy` | Best custom kernel | Speedup vs MKL |
| ---: | ---: | ---: | ---: |
| 96 x 96 | 8.63 GB/s | 80.87 GB/s (`2level_tuned_avx2_pf_store`) | 9.4x |
| 480 x 480 | 11.03 GB/s | 46.48 GB/s (`2level_tuned_avx2_nt_pf_ab_tile128`) | 4.2x |
| 960 x 960 | 10.81 GB/s | 39.77 GB/s (`transpose_tiled_v3`) | 3.7x |

## What Intel MKL Is

Intel MKL, now part of Intel oneAPI Math Kernel Library, is Intel's optimized numerical computing library. It includes BLAS routines for vector and matrix math, LAPACK routines for linear solvers, FFTs, sparse routines, vector math, and matrix copy/transpose operations.

In plain language: MKL is the vendor-tuned math layer many developers trust when they need CPU performance. Beating a naive implementation is expected. Beating MKL's own transpose routine in a controlled benchmark is a stronger signal because MKL represents years of hardware-specific optimization work.

## Why MKL Matters in AI Frameworks

AI frameworks such as TensorFlow and PyTorch rely on optimized CPU backends for matrix multiply, tensor transforms, fallback execution, preprocessing, and serving workloads that do not always sit entirely on GPUs. Intel's CPU AI stack includes oneMKL for linear algebra and oneDNN for deep-learning primitives such as convolution, matrix multiplication, normalization, and activation paths.

That is why this repository compares against MKL: it is a credible production-grade baseline for the kind of low-level performance layer AI infrastructure depends on.

## What Is Implemented Here

- `gemm/transpose_bench.cpp`: standalone transpose benchmark with naive, tiled, MKL, and optimized two-level variants.
- `gemm/transpose_2level.h`: cache-tiled transpose kernels with AVX2, prefetch, and non-temporal store experiments.
- `gemm/common.h`: production entry point used by GEMM paths when a transpose is needed.
- `include/benchmark.h`: shared benchmark registration, timing, verification, and table output framework.
- `gemm/`: related GEMM kernels that consume memory-layout improvements from transpose work.
- `ml_kernels/`: staging area for additional AI primitives such as ReLU and softmax.

## Reproduce

Install system dependencies:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin intel-mkl libopenblas-dev liblapack-dev liblapacke-dev
```

Build:

```bash
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
cd ..
```

From the repository root, run the transpose benchmark:

```bash
./build/gemm/transpose_bench --iters 3 --warmup 1 --sizes 96,480,960,1920
```

## Next Milestones

- Regenerate a full 1920 x 1920 Fixed Memory table and add it to this summary.
- Add CPU model, compiler, flags, and memory bandwidth context to make the result easier to audit.
- Extend the agent search loop across AVX-512, AMX, BF16, and AI inference-oriented tensor shapes.
- Report latency per operation and cost-oriented metrics beside raw GB/s.

## Sources

- Intel oneAPI Math Kernel Library developer reference: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/overview-001.html
- Intel AI Tools release notes covering oneDNN, TensorFlow, and PyTorch optimization context: https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-ai-analytics-toolkit-release-notes.html
- Intel newsroom note on oneDNN optimizations in TensorFlow: https://download.intel.com/newsroom/archive/2025/en-us-2022-05-25-intel-onednn-ai-optimizations-enabled-as-default-in-tensorflow.pdf
