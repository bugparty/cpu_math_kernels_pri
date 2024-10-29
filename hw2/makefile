CC = gcc
CXX= g++

LDFLAGS = -I /act/opt/intel/composer_xe_2013.3.163/mkl/include \
	-L /act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64 \
	-O3 -march=native -fprefetch-loop-arrays -DMKL_ILP64 -lmkl_avx2 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -m64

main: func_call.c include.h lapack.c main.c makefile my_block.c my.c
	rm -f main
	${CC} main.c -o main ${LDFLAGS}

bench: func_call.c include.h lapack.c bench.c makefile my_block.c my.c
	rm -f bench
	${CC} bench.c -o bench ${LDFLAGS}
.PHONY: clean
runmy: main
	./main my 4

clean:
	rm -f main  *.o