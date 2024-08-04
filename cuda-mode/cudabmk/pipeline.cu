//#include <stdio.h>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "repeat.h"
#include "instructions.h"

#include "device_functions.h"

/* This is alternating MAD and MUL to measure the dual issue. */
__global__ void KMAD_MUL (unsigned int *ts, unsigned int* out, float p1, float  p2, int its) 	
{														
	float t1 = p1;												
	float t2 = p2;
	float t3 = p1+p2;
	float t4 = p1*p2;
	unsigned int start_time=0, stop_time=0;									
													
	for (int i=0;i<its;i++)										
	{												
		__syncthreads();									
		start_time = clock();							
		repeat128(t1+=t1*t3;t2*=t4;)
		stop_time = clock();								
	}											
														
	out[0] = (unsigned int )(t1 + t2 + t3 + t4);							
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;						
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;				
}

/* Declaring external kernel function for testing syncthreads. Needs to be compiled
   separately with custom cubin because compiler will optimize away multiple syncthreads
   calls. */
extern "C"
__global__ void K_SYNC_UINT_DEP128 (unsigned int *ts, unsigned int* out, unsigned int p1, unsigned int p2, int its);




#define MEASURE_LATENCY(FUNC)										\
do {													\
	Db.x = 1;											\
													\
	FUNC <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);							\
	cudaThreadSynchronize();									\
	cudaError_t error;										\
	printf("  %s \tlatency:    \t", #FUNC); 							\
	if ((error = cudaGetLastError()) != cudaSuccess)						\
	{												\
		printf("  failed. %s\n\n", cudaGetErrorString(error));					\
		break;											\
	}												\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);					\
	cudaThreadSynchronize();									\
	printf ("%u clk (%.3f clk/warp)\n", ts[1]-ts[0], ((double)(ts[1] - ts[0])/kernel_ops));	\
} while(0)

#define MEASURE_LATENCY2(FUNC, NUM)									\
do {													\
	Db.x = NUM;											\
													\
	FUNC <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);							\
	cudaThreadSynchronize();									\
	cudaError_t error;										\
	printf("  %s \tlatency:    \t", #FUNC); 							\
	if ((error = cudaGetLastError()) != cudaSuccess)						\
	{												\
		printf("  failed. %s\n\n", cudaGetErrorString(error));					\
		break;											\
	}												\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);					\
	cudaThreadSynchronize();									\
	printf ("%u clk (%.3f clk/warp)\n", ts[1]-ts[0], ((double)(ts[1] - ts[0])/kernel_ops));	\
} while(0)


#define MEASURE_THROUGHPUT(FUNC)									\
do {													\
	Db.x = 512; 											\
													\
	printf ("  %s \tthroughput:\t", #FUNC);								\
	FUNC<<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);								\
	cudaThreadSynchronize();									\
	cudaError_t error;										\
	if ((error = cudaGetLastError()) != cudaSuccess)						\
	{												\
		printf("  failed. %s\n\n", cudaGetErrorString(error));					\
		break;											\
	}												\
	cudaThreadSynchronize();									\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);					\
	unsigned int min_t=(unsigned)-1, max_t=0;							\
	for (int i=0;i< Db.x*2;i++)									\
	{												\
		min_t = min(min_t, ts[i]);								\
		max_t = max(max_t, ts[i]);								\
	}												\
													\
	printf ("%9u clk (%.3f ops/clk)\n", max_t-min_t, (Db.x*kernel_ops)/(double)(max_t-min_t));	\
													\
} while (0)											

#define MEASURE_THROUGHPUT2(FUNC, NUM)									\
do {													\
	Db.x = NUM; 											\
													\
	printf ("  %s \tthroughput:\t", #FUNC);								\
	FUNC<<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);								\
	cudaThreadSynchronize();									\
	cudaError_t error;										\
	if ((error = cudaGetLastError()) != cudaSuccess)						\
	{												\
		printf("  failed. %s\n\n", cudaGetErrorString(error));					\
		break;											\
	}												\
	 cudaThreadSynchronize();									\
	cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);					\
	unsigned int min_t=(unsigned)-1, max_t=0;							\
	for (int i=0;i< Db.x*2;i++)									\
	{												\
		min_t = min(min_t, ts[i]);								\
		max_t = max(max_t, ts[i]);								\
	}												\
													\
	printf ("%d thrds %9u clk (%.3f ops/clk)\n", NUM, max_t-min_t, (Db.x*kernel_ops)/(double)(max_t-min_t));\
													\
} while (0)											



/* Run the test for increasing number of warps, and generate histogram of runtime. */
#define PRINT_HISTOGRAM(FUNC)															\
do {																		\
	printf ("\nPipeline latency/throughput with multiple warps (200 iterations of %d ops)\n", kernel_ops);					\
	printf ("  %s:\n", #FUNC);														\
	for (Db.x = 1; Db.x <= 512; Db.x += (Db.x < 4)? 1 : (Db.x < 8)? 2 : (Db.x < 32) ? 8 : 32)						\
	{																	\
		unsigned int histogram[1024] = {0};												\
		unsigned int sum_time = 0;													\
		unsigned int max_time;														\
		unsigned int min_time;														\
		unsigned int sum_max_time = 0;													\
		bool failed = false;														\
																		\
		for (int i=0;i<200 && !failed ;i++)												\
		{																\
			cudaGetLastError();		/* Clear previous error code, if any */							\
			FUNC <<<Dg, Db>>>(d_ts, d_out, 4, 6, 2);										\
			if (cudaGetLastError() != cudaSuccess)											\
			{															\
				failed = true;													\
				break;														\
			}															\
																		\
			cudaThreadSynchronize();												\
			cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);								\
																		\
			max_time = 0;														\
			min_time = (unsigned)-1;												\
			/* Compute histogram.	*/												\
			for (int j=0; j< Db.x*2; j+= 64)											\
			{															\
				sum_time += (ts[j+1] - ts[j]);											\
				max_time = max(max_time, ts[j+1]);										\
				min_time = min(min_time, ts[j]);										\
				histogram[(ts[j+1] - ts[j])/kernel_ops]++;									\
			}															\
			sum_max_time += max_time-min_time;											\
		}																\
																		\
		if (failed)															\
		{																\
			printf ("    %2d warp%c (%3d thread%c)  failed.", (Db.x+31)/32, Db.x>=64 ? 's' : ' ', Db.x, Db.x > 1 ? 's' : ' ');	\
		}																\
		else																\
		{																\
			/* Compute average latency over the lifetime of each warp (sum_time), and average throughput of the kernel (sum_max_time). */ \
			printf ("    %2d warp%c (%3d thr) %9u clk (%.3f clk/warp, %.3f ops/clk) ", 						\
						(Db.x+31)/32, Db.x>=64 ? 's' : ' ', Db.x, sum_max_time,						\
						sum_time/200.0/kernel_ops/((Db.x+31)/32), kernel_ops*200.0*Db.x/sum_max_time);			\
																		\
			printf ("  Histogram { ");	/* Print a histogram of each thread's runtime for the last iteration. */		\
			for (int i=0;i<1024;i++)	/* Print the non-zero entries only */							\
			{															\
				if (histogram[i] != 0)												\
					printf ("(%d: %d) ", i, histogram[i]);									\
			}															\
			printf ("}");														\
		}																\
																		\
		printf ("\n");															\
	}																	\
																		\
    printf ("\n");																\
} while(0)

void measure_pipeline()
{
	
	const int kernel_ops = 256;		// kernels have this many operations
	unsigned int ts[2 * 1024];		// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output
	

    	dim3 Db = dim3(1,1,1);
    	dim3 Dg = dim3(1,1);
	
	// Allocate device array.
	if (cudaSuccess != cudaMalloc((void**)&d_ts, sizeof(ts)))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	if (cudaSuccess != cudaMalloc((void**)&d_out, 4))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		return;
	}
	
    	cudaGetLastError();
    	fprintf (stderr, "Running pipeline tests...\n\n");


	/* Pipeline latency/throughput for all three functional units */
	MEASURE_LATENCY(K_ADD_UINT_DEP128);	
	MEASURE_LATENCY(K_RSQRT_FLOAT_DEP128);	
	MEASURE_LATENCY(K_ADD_DOUBLE_DEP128);	
	printf("\n");

	MEASURE_THROUGHPUT(K_ADD_UINT_DEP128);
	MEASURE_THROUGHPUT(K_RSQRT_FLOAT_DEP128);
	MEASURE_THROUGHPUT(K_ADD_DOUBLE_DEP128);
	printf("\n");

	/* ARITHMETIC INSTRUCTIONS: UINT */
	MEASURE_LATENCY(K_ADD_UINT_DEP128);	
	MEASURE_LATENCY(K_SUB_UINT_DEP128);	
	MEASURE_LATENCY(K_MAD_UINT_DEP128);	
	MEASURE_LATENCY(K_MUL_UINT_DEP128);	
	MEASURE_LATENCY(K_DIV_UINT_DEP128);	
	MEASURE_LATENCY(K_REM_UINT_DEP128);	
	MEASURE_LATENCY(K_MIN_UINT_DEP128);	
	MEASURE_LATENCY(K_MAX_UINT_DEP128);	

	MEASURE_THROUGHPUT(K_ADD_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_SUB_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MAD_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MUL_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_DIV_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_REM_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MIN_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MAX_UINT_DEP128);	
	printf("\n");

	/* ARITHMETIC INSTRUCTIONS: INT */
	MEASURE_LATENCY(K_ADD_INT_DEP128);	
	MEASURE_LATENCY(K_SUB_INT_DEP128);	
	MEASURE_LATENCY(K_MAD_INT_DEP128);	
	MEASURE_LATENCY(K_MUL_INT_DEP128);	
	MEASURE_LATENCY(K_DIV_INT_DEP128);	
	MEASURE_LATENCY(K_REM_INT_DEP128);	
	MEASURE_LATENCY(K_MIN_INT_DEP128);	
	MEASURE_LATENCY(K_MAX_INT_DEP128);	
	MEASURE_LATENCY(K_ABS_INT_DEP128);	

	MEASURE_THROUGHPUT(K_ADD_INT_DEP128);	
	MEASURE_THROUGHPUT(K_SUB_INT_DEP128);	
	MEASURE_THROUGHPUT(K_MAD_INT_DEP128);	
	MEASURE_THROUGHPUT(K_MUL_INT_DEP128);	
	MEASURE_THROUGHPUT(K_DIV_INT_DEP128);	
	MEASURE_THROUGHPUT(K_REM_INT_DEP128);	
	MEASURE_THROUGHPUT(K_MIN_INT_DEP128);	
	MEASURE_THROUGHPUT(K_MAX_INT_DEP128);	
	MEASURE_THROUGHPUT(K_ABS_INT_DEP128);
	printf("\n");

	/* ARITHMETIC INSTRUCTIONS: FLOAT */
	MEASURE_LATENCY(K_ADD_FLOAT_DEP128);	
	MEASURE_LATENCY(K_SUB_FLOAT_DEP128);	
	MEASURE_LATENCY(K_MAD_FLOAT_DEP128);	
	MEASURE_LATENCY(K_MUL_FLOAT_DEP128);	
	MEASURE_LATENCY(K_DIV_FLOAT_DEP128);	
	MEASURE_LATENCY(K_MIN_FLOAT_DEP128);	
	MEASURE_LATENCY(K_MAX_FLOAT_DEP128);	

	MEASURE_THROUGHPUT(K_ADD_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_SUB_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_MAD_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_MUL_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_DIV_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_MIN_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_MAX_FLOAT_DEP128);	
	printf("\n");


	/* ARITHMETIC INSTRUCTIONS: DOUBLE */
	MEASURE_LATENCY(K_ADD_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_SUB_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_MAD_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_MUL_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_DIV_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_MIN_DOUBLE_DEP128);	
	MEASURE_LATENCY(K_MAX_DOUBLE_DEP128);	

	MEASURE_THROUGHPUT(K_ADD_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_SUB_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_MAD_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_MUL_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_DIV_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_MIN_DOUBLE_DEP128);	
	MEASURE_THROUGHPUT(K_MAX_DOUBLE_DEP128);	
	printf("\n");

	/* LOGIC */
	MEASURE_LATENCY(K_AND_UINT_DEP128);	
	MEASURE_LATENCY(K_OR_UINT_DEP128);	
	MEASURE_LATENCY(K_XOR_UINT_DEP128);	
	MEASURE_LATENCY(K_SHL_UINT_DEP128);	
	MEASURE_LATENCY(K_SHR_UINT_DEP128);	
	
	MEASURE_THROUGHPUT(K_AND_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_OR_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_XOR_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_SHL_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_SHR_UINT_DEP128);	
	printf("\n");

	/* INTRINSICS */
	/* ARITHMETIC INTRINSICS: INTEGER */
	MEASURE_LATENCY(K_UMUL24_UINT_DEP128);	
	MEASURE_LATENCY(K_MUL24_INT_DEP128);	
	MEASURE_LATENCY(K_UMULHI_UINT_DEP128);	
	MEASURE_LATENCY(K_MULHI_INT_DEP128);	
	MEASURE_LATENCY(K_USAD_UINT_DEP128);	
	MEASURE_LATENCY(K_SAD_INT_DEP128);	
	
	MEASURE_THROUGHPUT(K_UMUL24_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MUL24_INT_DEP128);
	MEASURE_THROUGHPUT(K_UMULHI_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_MULHI_INT_DEP128);	
	MEASURE_THROUGHPUT(K_USAD_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_SAD_INT_DEP128);	
	printf("\n");

	/* ARITHMETIC INTRINSICS: FLOAT */
	MEASURE_LATENCY(K_FADD_RN_FLOAT_DEP128);	
	MEASURE_LATENCY(K_FADD_RZ_FLOAT_DEP128);	
	MEASURE_LATENCY(K_FMUL_RN_FLOAT_DEP128);	
	MEASURE_LATENCY(K_FMUL_RZ_FLOAT_DEP128);	
	MEASURE_LATENCY(K_FDIVIDEF_FLOAT_DEP128);	

	MEASURE_THROUGHPUT(K_FADD_RN_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_FADD_RZ_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_FMUL_RN_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_FMUL_RZ_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_FDIVIDEF_FLOAT_DEP128);	
	printf("\n");

	/* ARITHMETIC INTRINSICS: DOUBLE */
	MEASURE_LATENCY(K_DADD_RN_DOUBLE_DEP128);	

	MEASURE_THROUGHPUT(K_DADD_RN_DOUBLE_DEP128);	
	printf("\n");
	

	/* MATH INSTRUCTIONS: FLOAT */	
	MEASURE_LATENCY(K_RCP_FLOAT_DEP128);	
	MEASURE_LATENCY(K_SQRT_FLOAT_DEP128);	
	MEASURE_LATENCY(K_RSQRT_FLOAT_DEP128);	
	
	/*
	MEASURE_LATENCY(K_SIN_FLOAT_DEP128);	
	MEASURE_LATENCY(K_COS_FLOAT_DEP128);	
	MEASURE_LATENCY(K_TAN_FLOAT_DEP128);	
	MEASURE_LATENCY(K_EXP_FLOAT_DEP128);	
	MEASURE_LATENCY(K_EXP10_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOG_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOG2_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOG10_FLOAT_DEP128);	
	MEASURE_LATENCY(K_POW_FLOAT_DEP128);	
	*/

	MEASURE_THROUGHPUT(K_RCP_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_SQRT_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_RSQRT_FLOAT_DEP128);	
	/*
	MEASURE_THROUGHPUT(K_SIN_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_COS_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_TAN_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_EXP_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_EXP10_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOG_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOG2_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOG10_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_POW_FLOAT_DEP128);	
	*/
	printf("\n");
	

	/* MATHEMATICAL INTRINSICS: FLOAT */
	MEASURE_LATENCY(K_SINF_FLOAT_DEP128);	
	MEASURE_LATENCY(K_COSF_FLOAT_DEP128);	
	MEASURE_LATENCY(K_TANF_FLOAT_DEP128);	
	MEASURE_LATENCY(K_EXPF_FLOAT_DEP128);	
	MEASURE_LATENCY(K_EXP2F_FLOAT_DEP128);	
	MEASURE_LATENCY(K_EXP10F_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOGF_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOG2F_FLOAT_DEP128);	
	MEASURE_LATENCY(K_LOG10F_FLOAT_DEP128);	
	MEASURE_LATENCY(K_POWF_FLOAT_DEP128);	
	
	MEASURE_THROUGHPUT(K_SINF_FLOAT_DEP128);
	MEASURE_THROUGHPUT(K_COSF_FLOAT_DEP128);
	MEASURE_THROUGHPUT(K_TANF_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_EXPF_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_EXP2F_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_EXP10F_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOGF_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOG2F_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_LOG10F_FLOAT_DEP128);	
	MEASURE_THROUGHPUT(K_POWF_FLOAT_DEP128);	
	printf("\n");
	
	/* CONVERSION INTRINSICS: INT/FLOAT */
	MEASURE_LATENCY(K_INTASFLOAT_UINT_DEP128);	
	MEASURE_LATENCY(K_FLOATASINT_FLOAT_DEP128);	
	
	MEASURE_THROUGHPUT(K_INTASFLOAT_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_FLOATASINT_FLOAT_DEP128);	
	printf("\n");

	/* MISC INTRINSICS: INTEGER */
	
	MEASURE_LATENCY(K_POPC_UINT_DEP128);	
	MEASURE_LATENCY(K_CLZ_UINT_DEP128);	
	printf("\n");
	
	MEASURE_THROUGHPUT(K_POPC_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_CLZ_UINT_DEP128);	
	printf("\n");
	

	/* WARP INTRINSICS */
	
	MEASURE_LATENCY(K_ALL_UINT_DEP128);	
	MEASURE_LATENCY(K_ANY_UINT_DEP128);	
	MEASURE_LATENCY(K_SYNC_UINT_DEP128);	
	printf("\n");
	
	MEASURE_THROUGHPUT(K_ALL_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_ANY_UINT_DEP128);	
	MEASURE_THROUGHPUT(K_SYNC_UINT_DEP128);	
	printf("\n");
	

	PRINT_HISTOGRAM(K_ADD_UINT_DEP128);
	printf("\n");
	
	/* DUAL ISSUE TEST */
	printf ("Trying various combinations of MUL and MAD to test dual issue:\n");
	MEASURE_THROUGHPUT(K_MUL_FLOAT_DEP128);
	MEASURE_THROUGHPUT(K_MAD_FLOAT_DEP128);
	MEASURE_THROUGHPUT(KMAD_MUL);
	printf("\n");



	printf("Measuring latency of syncthreads with multiple warps running:\n");
	for (int i = 1; i <= 512/32; i++)
	{
		printf ("%d warps: ", i);
		MEASURE_LATENCY2(K_SYNC_UINT_DEP128, i * 32);	
	}
	
	cudaFree(d_ts);
	cudaFree(d_out);
	
}

