#include <stdio.h>

#include "repeat.h"

// Define some of the kernel functions here. If all of these functions are 
// placed into the same .cu file, there is so much kernel code that the compiler
// fails due to lack of memory.


#define LTEST_FUNC(N,M,STRING)	\
__global__ void kicache_line_##N##_##M (unsigned int *ts, unsigned int* out, int p1, int p2, int its)	\
{																										\
	volatile int t1 = p1;				\
	volatile int t2 = p1*p1;				\
	volatile int t3 = p1*p1+p1;			\
	volatile int t4 = p1*p1+p2;			\
	volatile int t5 = p1*p2;				\
	volatile int t6 = p1*p2+p1;			\
	volatile int t7 = p1*p2+p2;			\
	volatile int t8 = p2*p1*p2;			\
	unsigned int start_time, stop_time;			\
		start_time = clock();	\
		for (int i=0;i<its;i++)	{	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			STRING						\
		}							\
		stop_time = clock();		\
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = start_time;		\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = stop_time;		\
}

#define LTEST0(N) LTEST_FUNC(N,0,;)
#define LTEST1(N) LTEST_FUNC(N,1,t1 = abs(t2);)
#define LTEST2(N) LTEST_FUNC(N,2,t1 = abs(t2); t2 = abs(t3);)
#define LTEST3(N) LTEST_FUNC(N,3,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); )
#define LTEST4(N) LTEST_FUNC(N,4,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); )
#define LTEST5(N) LTEST_FUNC(N,5,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); )
#define LTEST6(N) LTEST_FUNC(N,6,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); t6 = abs(t7); )
#define LTEST7(N) LTEST_FUNC(N,7,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); t6 = abs(t7); t7 = abs(t8); )

#define LTESTS(N) LTEST0(N) LTEST1(N) LTEST2(N) LTEST3(N) LTEST4(N) LTEST5(N) LTEST6(N) LTEST7(N)

LTESTS(142);
LTESTS(143);
LTESTS(144);
LTESTS(145);
LTESTS(146);
LTESTS(147);
LTESTS(148);
LTESTS(149);
LTESTS(150);
LTESTS(151);
LTESTS(152);
LTESTS(153);
LTESTS(154);
LTESTS(155);
LTESTS(156);
LTESTS(157);
LTESTS(158);
LTESTS(159);
LTESTS(160);
LTESTS(161);
LTESTS(162);
LTESTS(163);
LTESTS(164);
/*
LTESTS(165);
LTESTS(166);
LTESTS(167);
LTESTS(168);*/


