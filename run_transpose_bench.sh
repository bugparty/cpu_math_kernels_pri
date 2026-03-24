
taskset -c 0,1,6,7,8,9,18,19 ./cmake-build-release/gemm/transpose_bench --sizes 1024,2048,4096 --iters 5 --warmup 2