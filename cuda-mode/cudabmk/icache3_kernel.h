

#include "repeat.h"


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
			repeat256 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
			repeat144 (t1 = abs(t2); t2 = abs(t3); t3=abs(t4); t4=abs(t5); t5=abs(t6); t6=abs(t7); t7=abs(t8); t8=abs(t1);) 	\
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

#define LTESTS(N) LTEST0(N)


#define DECLS(N)	\
__global__ void kicache_line_##N##_0 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);

                                            DECLS( 95) DECLS( 96) DECLS( 97) DECLS( 98) DECLS( 99) DECLS(100)
DECLS(101) DECLS(102) DECLS(103) DECLS(104) DECLS(105) DECLS(106) DECLS(107) DECLS(108) DECLS(109) DECLS(110)
DECLS(111) DECLS(112) DECLS(113) DECLS(114) DECLS(115) DECLS(116) DECLS(117) DECLS(118) DECLS(119) DECLS(120)
DECLS(121) DECLS(122) DECLS(123) DECLS(124) DECLS(125) DECLS(126) DECLS(127) DECLS(128) DECLS(129) DECLS(130)
DECLS(131) DECLS(132) DECLS(133) DECLS(134) DECLS(135) DECLS(136) DECLS(137) DECLS(138) DECLS(139) DECLS(140)
DECLS(141) DECLS(142) DECLS(143) DECLS(144) DECLS(145) DECLS(146) DECLS(147) DECLS(148) DECLS(149) DECLS(150)
DECLS(151) DECLS(152) DECLS(153) DECLS(154) DECLS(155) DECLS(156) DECLS(157) DECLS(158) DECLS(159) DECLS(160)
DECLS(161) DECLS(162) DECLS(163) DECLS(164) DECLS(165) DECLS(166) DECLS(167) DECLS(168) DECLS(169) DECLS(170)
DECLS(171) DECLS(172) DECLS(173) DECLS(174) DECLS(175) DECLS(176) DECLS(177) DECLS(178) DECLS(179) DECLS(180)
DECLS(181) DECLS(182) DECLS(183) DECLS(184) DECLS(185) DECLS(186)

__global__ void kicache_line_133_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_134_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_135_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_136_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_137_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_138_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_139_4 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);

__global__ void kicache_line_134_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_135_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_136_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_137_2 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_134_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_135_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_136_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);
__global__ void kicache_line_137_6 (unsigned int *ts, unsigned int* out, int p1, int p2, int its);


