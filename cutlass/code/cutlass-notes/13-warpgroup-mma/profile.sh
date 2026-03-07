# CSV output
# ncu --csv --log-file warpgroup_mma.csv  --metrics gpu__time_duration.sum --kernel-name "warpgroup_mma" python warpgroup_mma.py

# ncu-rep output
ncu -o ncu_prof_13 --import-source 1 --set full --kernel-name "warpgroup_mma" -f python warpgroup_mma.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_13 python warpgroup_mma.py
