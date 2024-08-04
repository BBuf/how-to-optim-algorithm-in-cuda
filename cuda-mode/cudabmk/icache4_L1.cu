
#include "repeat.h"

#define repeat0(S)

#define LTEST_FUNC(N,M,STRING)	\
__global__ void kicache_L1_##N##_##M (unsigned int *ts, unsigned int* out, int p1, int p2, int its)	\
{																										\
	volatile int t1 = p1;				\
	volatile int t2 = p1*p1;				\
	volatile int t3 = p1*p1+p1;			\
	volatile int t4 = p1*p1+p2;			\
	volatile int t5 = p1*p2;				\
	volatile int t6 = p1*p2+p1;			\
	volatile int t7 = p1*p2+p2;			\
	volatile int t8 = p2*p1*p2;			\
	unsigned int start_time=0, stop_time=0, timer = 0;			\
	if (blockIdx.x > 0) its *= 4; \
	if (blockIdx.x == 10) {		\
		for (int i=0;i<its;i++)	{	\
			repeat39 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			t1 = abs(t2);			\
		}							\
	}								\
	else if (blockIdx.x == 20) {		\
		for (int i=0;i<its;i++)	{	\
			repeat38 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			t1 = abs(t2);			\
		}							\
	}								\
	else if (blockIdx.x == 40) {		\
		for (int i=0;i<its;i++)	{	\
			repeat39 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			t1 = abs(t2);			\
		}							\
	}								\
	else if (blockIdx.x == 50) {		\
		for (int i=0;i<its;i++)	{	\
			repeat38 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			t1 = abs(t2);			\
		}							\
	}								\
	else if (blockIdx.x == 0) {			\
		for (int i=0;i<its;i++)	{	\
			start_time = clock();	\
			repeat##N (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			STRING						\
			stop_time = clock();		\
			for (int j=0;j<24;j++) t2+=t3;\
			timer += stop_time - start_time;		\
		}							\
	}								\
	out[0] = t1+t2+t3+t4+t5+t6+t7+t8;		\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2] = 0;		\
	ts[(blockIdx.x*blockDim.x + threadIdx.x)*2 +1] = timer;		\
}

#define LTEST0(N) LTEST_FUNC(N,0,;)
#define LTEST1(N) LTEST_FUNC(N,1,t1 = abs(t2);)
#define LTEST2(N) LTEST_FUNC(N,2,t1 = abs(t2); t2 = abs(t3);)
#define LTEST3(N) LTEST_FUNC(N,3,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); )
#define LTEST4(N) LTEST_FUNC(N,4,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); )
#define LTEST5(N) LTEST_FUNC(N,5,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); )
#define LTEST6(N) LTEST_FUNC(N,6,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); t6 = abs(t7); )
#define LTEST7(N) LTEST_FUNC(N,7,t1 = abs(t2); t2 = abs(t3); t3 = abs(t4); t4 = abs(t5); t5 = abs(t6); t6 = abs(t7); t7 = abs(t8); )

#define LTESTS(N) LTEST0(N) LTEST4(N) 

/*
LTESTS(48);
LTESTS(49);
LTESTS(50);
LTESTS(51);
LTESTS(52);
LTESTS(53);
LTESTS(54);
LTESTS(55);*/
LTESTS(56);
LTESTS(57);
LTESTS(58);
LTESTS(59);
LTESTS(60);
LTESTS(61);
LTESTS(62);
LTESTS(63);
LTESTS(64);
LTESTS(65);
LTESTS(66);
LTESTS(67);
LTESTS(68);
LTESTS(69);
LTESTS(70);
LTESTS(71);
LTESTS(72);
LTESTS(73);
LTESTS(74);
LTESTS(75);
LTESTS(76);
LTESTS(77);
LTESTS(78);
LTESTS(79);
LTESTS(80);
LTESTS(81);
LTESTS(82);
LTESTS(83);
LTESTS(84);
LTESTS(85);
LTESTS(86);
LTESTS(87);
LTESTS(88);

LTESTS(89);
LTESTS(90);
LTESTS(91);
LTESTS(92);



