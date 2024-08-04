
#include "repeat.h"

#define MAKE_IC_KERNEL(N) \
__global__ void kicache_test4_##N (unsigned int *ts, unsigned int* out, int p1, int p2, int its)	\
{	\
	volatile int t1 = p1;			\
	volatile int t2 = p1*p1;		\
	volatile int t3 = p1*p1+p1;		\
	volatile int t4 = p1*p1+p2;		\
	volatile int t5 = p1*p2;		\
	volatile int t6 = p1*p2+p1;		\
	volatile int t7 = p1*p2+p2;		\
	volatile int t8 = p2*p1*p2;		\
	unsigned int start_time, stop_time;	\
						\
	if (!(p1&(1<<blockIdx.x))) return;	\
	if (blockIdx.x > 0) its *= 2; 		\
						\
	if (blockIdx.x == 0)	{		\
		for (int i=0;i<its;i++)	{	\
			start_time = clock();	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)	\
 			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) \
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) \
			stop_time = clock();	\
		}				\
	}					\
	else {					\
		for (int i=0;i<its;i++)	{	\
			start_time = clock();	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)	\
 			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) \
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);)	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) \
			stop_time = clock();	\
		}				\
	}					\
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;	\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;	\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;	\
}

// This test takes too long to compile. Break up the set of kernels into multiple files so they can be compiled in parallel.

