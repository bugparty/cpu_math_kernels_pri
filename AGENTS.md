# Repository Guidelines

## Project Structure & Module Organization
This repository is a small HPC coursework monorepo organized by assignment:

- `gemm/`: HW1 matrix-multiplication kernels, benchmark drivers, CSV outputs, and helper scripts.
- `dgetrf/`: HW2 LU / linear-solver experiments and benchmark helpers.
- `mpi_prime/`: HW3 MPI sieve implementations and cluster-oriented scripts.
- Root files such as `CMakeLists.txt`, `benchmark.h`, and `run_transpose_bench.sh` coordinate shared builds and top-level runs.

Build artifacts should stay under `build/`. Keep generated benchmark output in module-local result folders rather than committing it at the repo root.

## Build, Test, and Development Commands
Install system dependencies first:

```bash
sudo apt install -y libopenmpi-dev openmpi-bin intel-mkl libopenblas-dev liblapack-dev liblapacke-dev
```

Primary build flow:

```bash
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

Useful targets and scripts:

- `./build/gemm/gemm_bench`: list or run GEMM kernels.
- `./build/gemm/gemm_bench --count`: show available kernel indices.
- `cd gemm && ./run_all_benchmarks.sh`: run all GEMM kernels and save per-kernel logs.
- `cd gemm && ./quick_bench.sh results.csv`: generate a CSV benchmark summary.
- `./build/dgetrf/hw2_main <func_name> <n>`: run HW2 solver experiments.
- `mpirun -np 4 ./build/mpi_prime/hw3_main <func_name> <n>`: run MPI sieve variants.

## Coding Style & Naming Conventions
Follow the existing style in each module:

- C and C++ are the primary languages; preserve the current procedural style.
- Use 4-space indentation and keep braces on their own lines for function bodies.
- Match established names such as `dgemm7_ikj`, `hw2_main`, and `sieve1.c`: lowercase, underscores, and numeric suffixes where they describe kernel variants.
- Prefer small, local changes; do not reformat whole files unless necessary.

## Testing Guidelines
There is no centralized unit-test suite. Verification is executable- and benchmark-driven:

- Rebuild from a clean `build/` directory after touching `CMakeLists.txt` or shared headers.
- For numeric kernels, run the relevant binary and confirm correctness output before comparing timings.
- When changing `gemm/`, include the exact benchmark command used.
- When changing `mpi_prime/`, test with a concrete `mpirun -np ...` example.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages such as `Add new DGEMM implementations and kernel optimizations`. Keep that pattern:

- Start with a verb: `Add`, `Refactor`, `Fix`, `Update`.
- Mention the affected module when useful.
- Keep PRs scoped to one assignment or benchmarking topic.

PR descriptions should include the reason for the change, commands run for verification, and any benchmark deltas or environment assumptions (`AVX512`, `OpenMPI`, `BLAS/LAPACK`).
