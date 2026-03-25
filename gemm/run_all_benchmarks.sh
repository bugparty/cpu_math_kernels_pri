#!/bin/bash

# Script to run all available gemm benchmarks
# Usage: ./run_all_benchmarks.sh [output_dir]

# Default output directory
OUTPUT_DIR="${1:-./benchmark_results}"
mkdir -p "$OUTPUT_DIR"

# Get the benchmark executable path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
BENCH_EXEC="$BUILD_DIR/gemm/gemm_bench"

# Check if benchmark executable exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo "Error: gemm_bench not found at $BENCH_EXEC"
    echo "Please build the project first."
    exit 1
fi

# Get max kernel index
MAX_INDEX=$("$BENCH_EXEC" --count)
if [ $? -ne 0 ]; then
    echo "Error: Failed to get max kernel index"
    exit 1
fi

echo "Found $((MAX_INDEX + 1)) kernels (index 0 to $MAX_INDEX)"
echo "Results will be saved to: $OUTPUT_DIR"
echo "========================================="
echo ""

# Run each kernel
for i in $(seq 0 $MAX_INDEX); do
    echo "Running kernel $i..."
    OUTPUT_FILE="$OUTPUT_DIR/kernel_${i}.log"
    
    "$BENCH_EXEC" "$i" > "$OUTPUT_FILE" 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Kernel $i completed successfully"
        # Extract kernel name from output
        KERNEL_NAME=$(head -n1 "$OUTPUT_FILE" | grep -oP 'Selected kernel \[\d+\] \K.*')
        echo "    Name: $KERNEL_NAME"
    else
        echo "  ✗ Kernel $i failed with exit code $EXIT_CODE"
    fi
    echo ""
done

echo "========================================="
echo "All benchmarks completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Summary of kernels:"
for i in $(seq 0 $MAX_INDEX); do
    OUTPUT_FILE="$OUTPUT_DIR/kernel_${i}.log"
    if [ -f "$OUTPUT_FILE" ]; then
        KERNEL_NAME=$(head -n1 "$OUTPUT_FILE" | grep -oP 'Selected kernel \[\d+\] \K.*')
        echo "  $i: $KERNEL_NAME"
    fi
done
