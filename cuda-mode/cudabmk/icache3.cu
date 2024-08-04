#include <stdio.h>

#include "repeat.h"
#include "icache3_kernel.h"

#define DO_LTEST(N,M)	\
	{int stime=0, max_stime=0, min_stime=0x80000000;	\
	for (int k=0;k<40;k++) {							\
		kicache_line_##N##_##M <<<Dg, Db>>>(d_ts, d_out, 4, 3, 100); cudaThreadSynchronize();	\
		cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost);			\
		stime += ts[1]-ts[0];				\
		if (max_stime < ts[1]-ts[0]) max_stime = ts[1]-ts[0];			\
		if (min_stime > ts[1]-ts[0]) min_stime = ts[1]-ts[0];			\
	}										\
	printf ("   %d: %.4f, %.4f, %.4f\n", (N+400)*64+M*8, (stime)/(4000.0*((N+400)*8+M)),		\
					(min_stime)/(100.0*((N+400)*8+M)), (max_stime)/(100.0*((N+400)*8+M))); }

#define DO_LTESTS(N) DO_LTEST(N,0)
	

void measure_icache3()
{
	
	unsigned int ts[1024];			// ts, output from kernel. Two elements used per thread.
	unsigned int *d_ts;
	unsigned int *d_out;			// Unused memory for storing output
	

    	dim3 Db = dim3(1);
    	dim3 Dg = dim3(1,1,1);
	
	// Allocate device array.
	cudaError_t errcode;
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_ts, sizeof(ts))))
	{
		printf ("cudaMalloc failed allocating %d bytes %s:%d\n", sizeof(ts), __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	if (cudaSuccess != (errcode = cudaMalloc((void**)&d_out, 4)))
	{
		printf ("cudaMalloc failed %s:%d\n", __FILE__, __LINE__);
		printf ("   %s\n", cudaGetErrorString(errcode));
		return;
	}
	
	printf ("L3 icache parameters test (~32KB of independent ops):\n");
	printf ("    [Codesize: avg, min, max runtime]\n");
	

	DO_LTESTS(95);
	DO_LTESTS(96);
	DO_LTESTS(97);
	DO_LTESTS(98);
	DO_LTESTS(99);
	DO_LTESTS(100);
	DO_LTESTS(101);
	DO_LTESTS(102);
	DO_LTESTS(103);
	DO_LTESTS(104);
	DO_LTESTS(105);
	DO_LTESTS(106);
	DO_LTESTS(107);
	DO_LTESTS(108);
	DO_LTESTS(109);
	DO_LTESTS(110);
	DO_LTESTS(111);
	DO_LTESTS(112);
	DO_LTESTS(113);
	DO_LTESTS(114);
	DO_LTESTS(115);
	DO_LTESTS(116);
	DO_LTESTS(117);
	DO_LTESTS(118);
	DO_LTESTS(119);
	DO_LTESTS(120);
	DO_LTESTS(121);
	DO_LTESTS(122);
	DO_LTESTS(123);
	DO_LTESTS(124);
	DO_LTESTS(125);
	DO_LTESTS(126);
	DO_LTESTS(127);
	DO_LTESTS(128);
	DO_LTESTS(129);	
	DO_LTESTS(130);
	DO_LTESTS(131);
	DO_LTESTS(132);
	DO_LTESTS(133); DO_LTEST(133,4);
	DO_LTESTS(134); DO_LTEST(134,2); DO_LTEST(134,4); DO_LTEST(134,6);
	DO_LTESTS(135); DO_LTEST(135,2); DO_LTEST(135,4); DO_LTEST(135,6);
	DO_LTESTS(136); DO_LTEST(136,2); DO_LTEST(136,4); DO_LTEST(136,6);
	DO_LTESTS(137); DO_LTEST(137,2); DO_LTEST(137,4); DO_LTEST(137,6);
	DO_LTESTS(138); DO_LTEST(138,4);
	DO_LTESTS(139);	DO_LTEST(139,4);
	DO_LTESTS(140);
	DO_LTESTS(141);
	DO_LTESTS(142);
	DO_LTESTS(143);
	DO_LTESTS(144);
	DO_LTESTS(145);
	DO_LTESTS(146);
	DO_LTESTS(147);
	DO_LTESTS(148);
	DO_LTESTS(149);	
	DO_LTESTS(150);
	DO_LTESTS(151);
	DO_LTESTS(152);
	DO_LTESTS(153);
	DO_LTESTS(154);
	DO_LTESTS(155);
	DO_LTESTS(156);
	DO_LTESTS(157);
	DO_LTESTS(158);
	DO_LTESTS(159);	
	DO_LTESTS(160);
	DO_LTESTS(161);
	DO_LTESTS(162);
	DO_LTESTS(163);
	DO_LTESTS(164);
	DO_LTESTS(165);
	DO_LTESTS(166);
	DO_LTESTS(167);
	DO_LTESTS(168);
	DO_LTESTS(169);		
	DO_LTESTS(170);
	DO_LTESTS(171);
	DO_LTESTS(172);
	DO_LTESTS(173);
	DO_LTESTS(174);
	DO_LTESTS(175);
	DO_LTESTS(176);
	DO_LTESTS(177);
	DO_LTESTS(178);
	DO_LTESTS(179);		
	DO_LTESTS(180);			
	DO_LTESTS(181);			
	DO_LTESTS(182);			
	DO_LTESTS(183);			
	DO_LTESTS(184);			
	DO_LTESTS(185);			
	DO_LTESTS(186);			
	
	
	cudaFree(d_ts);
	cudaFree(d_out);
}

int main()
{
	measure_icache3();
}


