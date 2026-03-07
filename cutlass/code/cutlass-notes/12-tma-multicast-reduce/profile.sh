# CSV output
# ncu --csv --log-file tma_multicast_reduce.csv  --metrics gpu__time_duration.sum --kernel-name "tma_multicast_reduce" python tma_multicast_reduce.py

# ncu-rep output
ncu -o ncu_prof_12 --import-source 1 --set full --kernel-name "tma_multicast_reduce" -f python tma_multicast_reduce.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_12 python tma_multicast_reduce.py
