import os
os.system("make tester > /dev/null")
os.system('echo "func, size, padding,time, correctness"')
for i in range(6,12):
   os.system("srun tester dgemm3 "+str(2**i)+" 3")
   

