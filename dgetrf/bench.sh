#!/bin/bash
export LD_LIBRARY_PATH=/home/bhan001/cs211-hw2-solving-large-linear-system-private-bugparty/extern:/act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64:$LD_LIBRARY_PATH

./dgetrf_bench_all --iters 3 --warmup 1 --sizes 1000,2000,3000,4000,5000
