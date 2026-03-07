# CSV output
# ncu --csv --log-file warp_specialization.csv  --metrics gpu__time_duration.sum --kernel-name "warp_specialization" python warp_specialization.py

# ncu-rep output
ncu -o ncu_prof_14 --import-source 1 --set full --kernel-name "warp_specialization" -f python warp_specialization.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_14 python warp_specialization.py
