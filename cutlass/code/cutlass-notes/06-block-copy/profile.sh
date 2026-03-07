# CSV output
# ncu --csv --log-file block_copy.csv  --metrics gpu__time_duration.sum --kernel-name "block_copy" python block_copy.py

# ncu-rep output
ncu -o ncu_prof_6 --import-source 1 --set full --kernel-name "block_copy" -f python block_copy.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_6 python block_copy.py
