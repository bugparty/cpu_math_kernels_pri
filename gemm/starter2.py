#!/bin/env python
import os

for i in range(0,4):
    os.system("make clean > /dev/null")
    os.system("make bench OPT=-O"+str(i)+" > /dev/null")
    os.system("srun bench dgemm7 2048 48")