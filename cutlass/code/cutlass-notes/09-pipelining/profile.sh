# CSV output
# ncu --csv --log-file pipelining.csv  --metrics gpu__time_duration.sum --kernel-name "pipelining" python pipelining.py

# ncu-rep output
ncu -o ncu_prof_9 --import-source 1 --set full --kernel-name "pipelining" -f python pipelining.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_9 python pipelining.py
