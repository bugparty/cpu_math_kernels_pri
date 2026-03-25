#!/bin/bash
export LD_LIBRARY_PATH=/home/bhan001/cs211-hw2-solving-large-linear-system-private-bugparty/extern:/act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64:$LD_LIBRARY_PATH

sizes=(1000 4000 5000)
size=(5000)
make mainpg
srun ./mainpg my_block "$size"
gprof -l  ./mainpg
