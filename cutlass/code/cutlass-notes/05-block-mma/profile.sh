# CSV output
# ncu --csv --log-file block_mma.csv  --metrics gpu__time_duration.sum --kernel-name "block_mma" python block_mma.py

# ncu-rep output
ncu -o ncu_prof_5 --import-source 1 --set full --kernel-name "block_mma" -f python block_mma.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_5 python block_mma.py
