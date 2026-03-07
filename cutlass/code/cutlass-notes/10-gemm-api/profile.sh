# CSV output
# ncu --csv --log-file gemm_api.csv  --metrics gpu__time_duration.sum --kernel-name "gemm_api" python gemm_api.py

# ncu-rep output
ncu -o ncu_prof_10 --import-source 1 --set full --kernel-name "gemm_api" -f python gemm_api.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_10 python gemm_api.py
