all:
	/usr/local/cuda/bin/nvcc -o bin/reduce_v0 reduce_v0_baseline.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v1 reduce_v1_interleaved_addressing.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v2 reduce_v2_bank_conflict_free.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v3 reduce_v3_idle_threads_free.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v4 reduce_v4_unroll_last_warp.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v5 reduce_v5_completely_unroll.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v6 reduce_v6_multi_add.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v7 reduce_v7_shfl_down_sync.cu
	/usr/local/cuda/bin/nvcc -o bin/reduce_v8 reduce_v8_shfl_down_sync_pack.cu


