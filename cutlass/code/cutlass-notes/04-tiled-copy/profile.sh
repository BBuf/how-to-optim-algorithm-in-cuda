# CSV output
# ncu --csv --log-file tiled_copy.csv  --metrics gpu__time_duration.sum --kernel-name "tiled_copy" python tiled_copy.py

# ncu-rep output
ncu -o ncu_prof_4 --import-source 1 --set full --kernel-name "tiled_copy" -f python tiled_copy.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_4 python tiled_copy.py
