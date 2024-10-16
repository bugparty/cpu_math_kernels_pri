#!/bin/env python
import os
MAT_SIZE=2048
for b in range(1,8):
   os.system("rm bench6")
   os.system("gcc -DBLOCK_SIZE={} tester2.c -o bench6".format(2**b))
   os.system("./bench6 dgemm6_ijk2 {} 1".format(MAT_SIZE))
   os.system("./bench6 dgemm6_ikj2 {} 1".format(MAT_SIZE))
   os.system("./bench6 dgemm6_jik2 {} 1".format(MAT_SIZE))
   os.system("./bench6 dgemm6_jki2 {} 1".format(MAT_SIZE))
   os.system("./bench6 dgemm6_kij2 {} 1".format(MAT_SIZE))
   os.system("./bench6 dgemm6_kji2 {} 1".format(MAT_SIZE))
      

