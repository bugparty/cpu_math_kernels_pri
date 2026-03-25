# GEMM Benchmark Scripts

## Overview

These scripts help automate running all available GEMM (General Matrix Multiply) kernel benchmarks.

## Scripts

### 1. `run_all_benchmarks.sh`

Runs all available kernels and saves individual log files for each.

**Usage:**
```bash
./run_all_benchmarks.sh [output_directory]
```

**Example:**
```bash
# Save results to default directory (./benchmark_results)
./run_all_benchmarks.sh

# Save results to custom directory
./run_all_benchmarks.sh ./my_results
```

**Output:**
- Individual log files: `kernel_0.log`, `kernel_1.log`, etc.
- Each file contains the full benchmark output for that kernel

### 2. `quick_bench.sh`

Runs all kernels and generates a single CSV file with results.

**Usage:**
```bash
./quick_bench.sh [output.csv]
```

**Example:**
```bash
# Save to default file (benchmark_results.csv)
./quick_bench.sh

# Save to custom CSV file
./quick_bench.sh results.csv
```

**Output:**
- CSV file with columns: kernel_index, kernel_name, matrix_size, avg_time_sec, gflops
- Easy to import into spreadsheet software or analysis tools

**View CSV results:**
```bash
# Pretty print in terminal
column -t -s, results.csv | less -S

# Or use any CSV viewer
libreoffice results.csv
```

## Using gemm_bench Directly

### Get number of available kernels:
```bash
./build/gemm/gemm_bench --count
# Output: 8 (means kernels 0-8 are available)
```

### Run specific kernel:
```bash
./build/gemm/gemm_bench 0  # Run kernel 0 (reference)
./build/gemm/gemm_bench 1  # Run kernel 1 (dgemm1)
# etc.
```

### List all available kernels:
```bash
./build/gemm/gemm_bench
# Shows usage and kernel list
```

## Available Kernels

The available kernels depend on CPU features. On systems without AVX512:

- 0: reference
- 1: dgemm1
- 2: dgemm3
- 3: dgemm3v2
- 4: dgemmBT1
- 5: dgemm7_ijk
- 6: dgemm7_kij
- 7: dgemm7_ikj
- 8: dgemmAVX

On systems with AVX512 support, additional kernels are available:
- dgemm7
- dgemmAVX512
- dgemmAVX512B

## Notes

- The reference kernel (index 0) is slower but used for correctness verification
- Benchmarks test matrix sizes from 100x100 to 3000x3000
- Each size is tested 3 times and averaged
- Performance is reported in GFLOPS (Giga Floating Point Operations Per Second)
