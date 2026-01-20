import os
current_directory = os.getcwd()
intel64_path = os.path.join(current_directory, "extern")

os.environ["LD_LIBRARY_PATH"] = "/act/opt/intel/composer_xe_2013.3.163/mkl/lib/intel64:" + os.environ.get("LD_LIBRARY_PATH","")
os.environ["LD_LIBRARY_PATH"] = intel64_path + ":" + os.environ.get("LD_LIBRARY_PATH","")
os.system("make")
os.system("./main lapack 4096")
os.system("srun main lapack 4096")
