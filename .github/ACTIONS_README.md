  # GitHub Actions Workflow

## Build Workflow

The `build.yml` workflow compiles all three subprojects on every push and pull request.

### What it does

1. **Installs dependencies**:
   - Build tools (CMake, GCC)
   - OpenMPI
   - Intel MKL
   - OpenBLAS and LAPACK

2. **Builds all subprojects**:
   - `gemm`
   - `dgetrf`
   - `mpi_prime`

3. **Uploads build artifacts**:
   - Main executables from each subproject

## Benchmark Workflow

The `benchmark_gemm.yml` workflow automatically runs GEMM benchmarks on every push and pull request.

### What it does

1. **Installs dependencies**:
   - Build tools (CMake, GCC)
   - OpenMPI
   - OpenBLAS and LAPACK

2. **Builds the project**:
   - Compiles `gemm_bench_all` with optimizations
   - Skips AVX512 functions on CPUs without support

3. **Runs benchmarks**:
   - Executes all available kernels
   - Generates individual log files
   - Creates CSV summary

4. **Uploads artifacts**:
   - Benchmark results (kept for 30 days)
   - Compiled executable (kept for 7 days)

### Viewing Results

After the workflow completes:

1. Go to the **Actions** tab in GitHub
2. Click on the workflow run
3. Check the **Summary** section for quick results
4. Download artifacts:
   - `benchmark-results`: Full logs and CSV
   - `gemm-bench-all-executable`: The compiled binary

### Manual Triggering

You can manually trigger the workflow:

1. Go to **Actions** tab
2. Select **benchmark_gemm** workflow
3. Click **Run workflow** button

### Configuration

Edit `.github/workflows/benchmark_gemm.yml` to customize:

- `timeout-minutes`: Maximum runtime (default: 120 minutes)
- `retention-days`: How long to keep artifacts
- Build options: Modify CMake flags

### Local Testing

To test the workflow locally before pushing:

```bash
# Install dependencies
sudo apt-get install -y build-essential cmake libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev liblapacke-dev

# Build
mkdir -p build && cd build
cmake .. -DBUILD_HW1=ON -DBUILD_HW2=OFF -DBUILD_HW3=OFF
make gemm_bench_all -j$(nproc)

# Run benchmarks
cd ../gemm
./run_all_benchmarks.sh benchmark_results
```
