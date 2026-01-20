#!/bin/bash

# Quick benchmark script - runs all kernels and saves results in CSV format
# Usage: ./quick_bench.sh [output.csv]

OUTPUT_CSV="${1:-benchmark_results.csv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
BENCH_EXEC="$BUILD_DIR/gemm/gemm_bench"

# Check if benchmark executable exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo "Error: gemm_bench not found at $BENCH_EXEC"
    exit 1
fi

# Get max kernel index
MAX_INDEX=$("$BENCH_EXEC" --count)
if [ $? -ne 0 ]; then
    echo "Error: Failed to get max kernel index"
    exit 1
fi

echo "Running benchmarks for $((MAX_INDEX + 1)) kernels..."
echo "Results will be saved to: $OUTPUT_CSV"

# Create CSV header
echo "kernel_index,kernel_name,matrix_size,avg_time_sec,gflops" > "$OUTPUT_CSV"

# Run each kernel
for i in $(seq 0 $MAX_INDEX); do
    printf "Kernel %d/%d..." "$i" "$MAX_INDEX"
    
    TEMP_FILE=$(mktemp)
    "$BENCH_EXEC" "$i" > "$TEMP_FILE" 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Extract kernel name
        KERNEL_NAME=$(grep "Selected kernel" "$TEMP_FILE" | sed -n 's/.*\] \(.*\)/\1/p')
        
        # Parse benchmark results and append to CSV
        grep "M=N=K=" "$TEMP_FILE" | while read -r line; do
            SIZE=$(echo "$line" | grep -oP 'M=N=K=\K\d+')
            TIME=$(echo "$line" | grep -oP 'average elapsed \K[0-9.]+')
            GFLOPS=$(echo "$line" | grep -oP 'performance \K[0-9.]+')
            echo "$i,$KERNEL_NAME,$SIZE,$TIME,$GFLOPS" >> "$OUTPUT_CSV"
        done
        
        echo " ✓"
    else
        echo " ✗ (failed)"
    fi
    
    rm -f "$TEMP_FILE"
done

echo "Done! Results saved to: $OUTPUT_CSV"
echo ""
echo "You can view the results with:"
echo "  column -t -s, $OUTPUT_CSV | less -S"
