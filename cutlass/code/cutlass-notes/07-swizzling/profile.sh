# CSV output
# ncu --csv --log-file swizzling.csv  --metrics gpu__time_duration.sum --kernel-name "swizzling" python swizzling.py

# ncu-rep output
ncu -o ncu_prof_7 --import-source 1 --set full --kernel-name "swizzling" -f python swizzling.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_7 python swizzling.py
