all:
	nvcc -o bin/reduce_v0 reduce_v0_baseline.cu
	nvcc -o bin/reduce_v1 reduce_v1_interleaved_addressing.cu
	nvcc -o bin/reduce_v2 reduce_v2_bank_conflict_free.cu
	nvcc -o bin/reduce_v3 reduce_v3_idle_threads_free.cu
	nvcc -o bin/reduce_v4 reduce_v4_unroll_last_warp.cu
	
