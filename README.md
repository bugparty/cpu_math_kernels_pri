# cs211monorepo

![GEMM Benchmarks](https://github.com/bugparty/cpu_math_kernels/actions/workflows/benchmark.yml/badge.svg)

## Install Dependencies

```bash
sudo apt install -y libopenmpi-dev openmpi-bin intel-mkl libopenblas-dev liblapack-dev liblapacke-dev
```

## Quick Start

### Build
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Run GEMM Benchmarks

See [gemm/BENCHMARK_README.md](gemm/BENCHMARK_README.md) for detailed instructions.

```bash
# List available kernels
./build/gemm/gemm_bench

# Run specific kernel
./build/gemm/gemm_bench 0

# Run all benchmarks
cd gemm
./run_all_benchmarks.sh
```

## GitHub Actions

Benchmarks run automatically on every push. See [.github/ACTIONS_README.md](.github/ACTIONS_README.md) for details.  