#include <stdio.h>
#include <stdlib.h>

#include "repeat.h"

const int page_size = 4;

//declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref; 

__global__ void texture_latency (int * my_array, int size, unsigned long long *duration, int iter, unsigned int INNER_ITS2 ) {

   unsigned int start_time, end_time;
   int j = 0;
   unsigned long long sum_time = 0;
	unsigned int INNER_ITS = INNER_ITS2;

   duration[0] = 0;

   for (int k=0; k <= iter; k++) {
  	if (k==1) {
		sum_time = 0; // ignore the first iteration (cold misses)
	}

	start_time = clock();
	for (int cnt=0; cnt < INNER_ITS; cnt++) {
   		repeat256(j=tex1Dfetch(tex_ref, j););
	}
   	end_time = clock();

	sum_time += (end_time - start_time);
  }

  duration[0] = sum_time;
  duration[1] = j;

}


void measure_pagesize(int elems, int iterations, int stride, int offset) {

	const int N = elems * stride * 2 + offset + stride*2;
	cudaError_t error_id;

	// need to declare an array in linear memory. And initialize it. Then I'll bind a texture to it and will read it from texture memory and therefore faster.
	int * h_a, * d_a;
	unsigned long long * duration, * latency;
	double result = 0, rmax = 0, rmin = 1e99;

	h_a = (int *)malloc(sizeof(int) * N);

	latency = (unsigned long long *)malloc(2*sizeof(unsigned long long));
	latency[0] = 5;

	int size = N * sizeof(int);
	int size2 = 2*sizeof(unsigned long long);

	//initialize array
	
	for (int i=0;i<elems; i++)
		h_a[i*stride] = i*stride + stride;

	h_a[(elems-1)*stride] = elems*stride + offset;	//point last element to stride+offset

	for (int i=0;i<elems; i++)
		h_a[(i+elems)*stride+offset] = (i+elems)*stride + offset + stride;

	h_a[(2*elems-1)*stride+offset] = 0;		//wrap around.


	//int l =0;
	//repeat256(l=h_a[l]; printf("%d\n", h_a[l]););


	/* allocate array on GPU */
	cudaMalloc ((void **) &d_a, size);
	cudaMalloc ((void **) &duration, size2);

        cudaThreadSynchronize ();

        error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		printf("Error 1 is %s\n", cudaGetErrorString(error_id));
	}


	/* copy array elements from CPU to GPU */
        cudaMemcpy((void *)d_a, (void *)h_a, size, cudaMemcpyHostToDevice);

        error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
	}

        cudaMemcpy((void *)duration, (void *)latency, size2, cudaMemcpyHostToDevice);

        error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
	}


	//bind texture
	cudaBindTexture(0, tex_ref, d_a, size );

	cudaThreadSynchronize ();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 3 is %s\n", cudaGetErrorString(error_id));
	}

	unsigned int INNER_ITS = (N/stride) / (256/3.0) ;
	if (INNER_ITS < 16) INNER_ITS = 16;
	const int it = INNER_ITS*256;
	for (int l=0; l < 20; l++) {


		// launch kernel
		dim3 Db = dim3(1);
		dim3 Dg = dim3(1,1,1);
		texture_latency <<<Dg, Db>>>(d_a, size, duration, iterations, INNER_ITS);

		cudaThreadSynchronize ();

		error_id = cudaGetLastError();
		if (error_id != cudaSuccess) {
			printf("Error 4 is %s\n", cudaGetErrorString(error_id));
		}

		cudaThreadSynchronize ();

		/* copy results from GPU to CPU */
		//cudaMemcpy((void *)h_a, (void *)d_a, size, cudaMemcpyDeviceToHost);
		cudaMemcpy((void *)latency, (void *)duration, size2 , cudaMemcpyDeviceToHost);

        	cudaThreadSynchronize ();

		result += latency[0];
		if (rmax < latency[0]/(double)(it*iterations))
			rmax = latency[0]/(double)(it*iterations);
		if (rmin > latency[0]/(double)(it*iterations))
			rmin = latency[0]/(double)(it*iterations);
	}

	
	//int it = 256;
	result = result*1.0/(20*it* iterations);

	printf("   %d, %d, %f\n", (N)*sizeof(int), offset*sizeof(int), result);


	//unbind texture
	cudaUnbindTexture(tex_ref);

	//free memory on GPU
	cudaFree(d_a);
	cudaFree(duration);
        cudaThreadSynchronize ();
	
	// free memory on CPU
        free(h_a);
        free(latency);
	
}

int main() {

	int iterations;

	printf("Assuming page size is %d KB\n", page_size);
	printf("Texture2: Measuring Texture TLB page size\n");
	printf("Size(bytes), offset(bytes), latency(cycles)\n");

	iterations = 1;

	// Assume L1 TLB is set-associative, and try to find some spacing of two sets of 12 elements
	// such that they don't conflict and will fit into the 16-way L1 TLB. We find no such spacing,
	// so it appears the L1 TLB is really 512KB (128 x 4KB pages) lines, 16-way fully-associative...
	// We're kinda limited here in what to do because the 20-way TexL1 is virtually-addressed and hides
	// the 16-way L1 TLB.
	printf("\nL1 TLB page size using 2 sets of 12 elements spaced %dK+offset apart, using stride %dK.\n",
		512 * page_size/4, 512 * page_size/4);
	for (int offset = 0/4; offset <= 524288/4; offset += 2048/4 ) {
		measure_pagesize(12, iterations, 524288/4 * page_size/4, offset * page_size/4);
	}

	// 20x2 elements is enough to miss in the virtually-addressed 20-way L1 cache if all entries map to the same set. 
	// With 512K stride, we use 4 sets of the L2 TLB, so the L2 TLB can store 32 elements.
	// When first 20 elements conflicts with second set of 20 elements, 40 does not fit. 
	// When spaced just far enough apart to not conflict, 20 elements per set will fit.
	printf("\nL2 TLB page size using 2 sets of 20 elements spaced %dK+offset apart, using stride %dK.\n",
		512 * page_size/4, 512 * page_size/4);
	for (int offset = -2048/4; offset <= 6144/4; offset += 256/4 ) {
	       	measure_pagesize(20, iterations, 524288/4 *page_size/4, offset *page_size/4);
	}


	return 0;

}
