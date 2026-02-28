all:
	/usr/local/cuda/bin/nvcc -arch=sm_80 -o bin/oneflow_softmax oneflow_softmax.cu -I/home/zhangxiaoyu/thrust/dependencies/cub/build/headers
	/usr/local/cuda/bin/nvcc -arch=sm_80 -o bin/faster_transformer_softmax faster_transformer_softmax.cu
	
