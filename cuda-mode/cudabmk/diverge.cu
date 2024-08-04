#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "repeat.h"

__global__ void kdiverge_test (unsigned int *ts, unsigned int* out, int group_size,  int p2, int its)
{
	int t1 = group_size;
	int t2 = p2;
	unsigned int start_time=0, stop_time=0;
	unsigned int tid = ((blockIdx.x * blockDim.x) + threadIdx.x) % 32;
	unsigned int tid2 = tid/group_size;

	for (int i=0;i<its;i++)
	{
		__syncthreads();

		switch (tid2)
		{
		 	case 0:
				start_time = clock();
				repeat16(t1+=t2;t2+=t1;)
				stop_time = clock();
				break;
		 	case 1:
				start_time = clock();
				repeat16(t1^=t2;t2-=t1;)
				stop_time = clock();
		 		break;
			case 2:
				start_time = clock();
				repeat16(t1<<=t2;t2&=t1;)
				stop_time = clock();
				break;
		 	case 3:
				start_time = clock();
				repeat16(t1+=t2;t2&=t1;)
				stop_time = clock();
				break;
		 	case 4:
				start_time = clock();
				repeat16(t1+=t2;t2|=t1;)
				stop_time = clock();
				break;
		 	case 5:
				start_time = clock();
				repeat16(t1+=t2;t2^=t1;)
				stop_time = clock();
		 		break;
			case 6:
				start_time = clock();
				repeat16(t1+=t2;t2<<=t1;)
				stop_time = clock();
				break;
		 	case 7:
				start_time = clock();
				repeat16(t1+=t2;t2>>=t1;)
				stop_time = clock();
				break;
		 	case 8:
				start_time = clock();
				repeat16(t1^=t2;t2+=t1;)
				stop_time = clock();
				break;
		 	case 9:
				start_time = clock();
				repeat16(t1-=t2;t2-=t1;)
				stop_time = clock();
		 		break;
			case 10:
				start_time = clock();
				repeat16(t1<<=t2;t2|=t1;)
				stop_time = clock();
				break;
		 	case 11:
				start_time = clock();
				repeat16(t1-=t2;t2&=t1;)
				stop_time = clock();
				break;
		 	case 12:
				start_time = clock();
				repeat16(t1-=t2;t2|=t1;)
				stop_time = clock();
				break;
		 	case 13:
				start_time = clock();
				repeat16(t1-=t2;t2^=t1;)
				stop_time = clock();
				break;
		 	case 14:
				start_time = clock();
				repeat16(t1-=t2;t2<<=t1;)
				stop_time = clock();
				break;
		 	case 15:
				start_time = clock();
				repeat16(t1-=t2;t2>>=t1;)
				stop_time = clock();
				break;
			case 16:
				start_time = clock();
				repeat16(t1&=t2;t2+=t1;)
				stop_time = clock();
				break;
		 	case 17:
				start_time = clock();
				repeat16(t1&=t2;t2-=t1;)
				stop_time = clock();
		 		break;
			case 18:
				start_time = clock();
				repeat16(t1^=t2;t2<<=t1;)
				stop_time = clock();
				break;
		 	case 19:
				start_time = clock();
				repeat16(t1&=t2;t2&=t1;)
				stop_time = clock();
				break;
		 	case 20:
				start_time = clock();
				repeat16(t1&=t2;t2|=t1;)
				stop_time = clock();
				break;
		 	case 21:
				start_time = clock();
				repeat16(t1&=t2;t2^=t1;)
				stop_time = clock();
		 		break;
			case 22:
				start_time = clock();
				repeat16(t1&=t2;t2<<=t1;)
				stop_time = clock();
				break;
		 	case 23:
				start_time = clock();
				repeat16(t1&=t2;t2>>=t1;)
				stop_time = clock();
				break;
			case 24:
				start_time = clock();
				repeat16(t1|=t2;t2+=t1;)
				stop_time = clock();
				break;
		 	case 25:
				start_time = clock();
				repeat16(t1|=t2;t2-=t1;)
				stop_time = clock();
		 		break;
			case 26:
				start_time = clock();
				repeat16(t1^=t2;t2>>=t1;)
				stop_time = clock();
				break;
		 	case 27:
				start_time = clock();
				repeat16(t1|=t2;t2&=t1;)
				stop_time = clock();
				break;
		 	case 28:
				start_time = clock();
				repeat16(t1|=t2;t2|=t1;)
				stop_time = clock();
				break;
		 	case 29:
				start_time = clock();
				repeat16(t1|=t2;t2^=t1;)
				stop_time = clock();
		 		break;
			case 30:
				start_time = clock();
				repeat16(t1|=t2;t2<<=t1;)
				stop_time = clock();
				break;
		 	case 31:
				start_time = clock();
				repeat16(t1|=t2;t2>>=t1;)
				stop_time = clock();
				break;

			default:
				break;		 
		}

	}

	out[0] = (unsigned int )(t1 + t2);
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;
}



//first argument is the group size, for example, 1 for 32 divergent threads 
//second argument is the total number of threads in the block
#define RUN_DIVERGE_TEST(GROUP_SIZE, THRDS)									\
do														\
{ 														\
	Db.x = THRDS;												\
														\
	printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");		\
	printf("kdiverge_test: (%d threads, each warp divided into %d groups of %d thread(s))\n", 			\
				  THRDS, 32/GROUP_SIZE, GROUP_SIZE);						\
	kdiverge_test<<<Dg, Db>>>(d_ts, d_out, GROUP_SIZE, 6, 2);					\
	cudaThreadSynchronize();										\
														\
	if ((errcode = cudaGetLastError()) != cudaSuccess)							\
	{													\
		printf("  failed. %s\n\n", cudaGetErrorString(errcode));					\
	}													\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);						\
	cudaThreadSynchronize();										\
 														\
	min = ts[0];												\
	for (int i = 0; i < 2 * Db.x; i++)									\
	{													\
		min = (ts[i] < min) ? ts[i] : min;								\
	}													\
														\
	for (int i = 0; i < Db.x; i++) 										\
	{ 													\
		if (i && i % 32 == 0)										\
			printf("---------------------------------------------------------------\n");		\
		printf("(warp: %2d, thread: %2d): \t start: %6u, stop: %6u\n", i/32, i%32, 			\
		        ts[2*i]-min, ts[2*i+1]-min); 								\
	}													\
	printf("\n");												\
														\
} while (0)



void measure_diverge()
{

	unsigned int ts[2048];		// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;		// Unused memory for storing output
	unsigned int min;

    	dim3 Db = dim3(1,1,1);
    	dim3 Dg = dim3(1,1);

	cudaError_t errcode;
	// Allocate device array.
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}

    	errcode = cudaGetLastError();
	if (errcode != cudaSuccess)
		printf("  failed. %s\n", cudaGetErrorString(errcode));    

	fprintf (stderr, "Running divergence tests...\n");



	//first argument is group size, second argument is number of threads
	RUN_DIVERGE_TEST(1, 64);
	RUN_DIVERGE_TEST(2, 32);
	RUN_DIVERGE_TEST(16, 32);

	printf("\n");

	cudaFree(d_ts);
	cudaFree(d_out);
}

// Functions defined in other .cu files.
void measure_path();
void measure_diverge2();
void measure_diverge3();

int main()
{
	measure_path();
	measure_diverge();
	measure_diverge2();
	measure_diverge3();
}
