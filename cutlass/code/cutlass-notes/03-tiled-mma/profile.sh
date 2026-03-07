# CSV output
# ncu --csv --log-file 03.csv  --metrics gpu__time_duration.sum --kernel-name "tiled_mma" python tiled_mma.py

# ncu-rep output
ncu -o ncu_prof_3 --import-source 1 --set full --kernel-name "tiled_mma" -f python tiled_mma.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_3 python tiled_mma.py
