#include <stdio.h>
#include <stdlib.h>

#include "repeat.h"

const int page_size = 4;

//declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref; 

__global__ void texture_latency (int * my_array, int size, unsigned long long *duration, int iter, unsigned int INNER_ITS2 ) {

   unsigned int start_time, end_time;
   int j;
   unsigned long long sum_time = 0;
	unsigned int INNER_ITS = INNER_ITS2;

   duration[0] = 0;
	j=0;
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

int gcf(int a, int b)
{
	if (a == 0) return b;
	return gcf(b % a, a);
}



void parametric_measure_texture(int N, int iterations, int stride) {


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

	// Don't die if too much memory was requested.
	if (N > 241600000) { printf ("OOM.\n"); return; }

	//initialize array
	
	int step = gcf (stride, N-2);	// Optimization: Initialize fewer elements.
	for (int i = 0; i < (N-2); i+=step) {
		h_a[i] = (i + stride) % (N-2);
	}
	h_a[N-2] = 0;
	h_a[N-1] = 0;


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

	printf("   %d, %f, %f, %f\n", (N-2)*sizeof(int), result, rmax, rmin);


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

	int N, iterations, stride;

	printf("Assuming page size is %d KB\n", page_size);
	printf("Texture4: Measuring Texture caches and TLB\n");
	printf("array size(bytes), latency(cycles), max latency(cycles), min latency(cycles)\n");

	iterations = 1;

	printf("\n8-byte stride, 0 to 8K\n");
	for (N = (0); N <= (8192/4); N += (8/4)) {
		stride = 8/4;
		parametric_measure_texture(N+2, iterations, stride);
	}

	printf("\n64-byte stride, 0 to 320K\n");
	for (N = 0/4; N <= 327680/4; N += 64/4 ) {
		stride = 64/4;
        	parametric_measure_texture(N+2, iterations, stride);
	}
	
	printf("\n%d-KB stride, %d to %dM, for Tex L1 TLB prefetching\n", 128 * page_size/4, 5 * page_size/4, 10 * page_size/4);
	for (N = 5242880/4; N <= 1048576/4; N += 131072/4 ) {
		stride = 131072/4;
        	parametric_measure_texture((N * page_size/4)+2, iterations, stride * page_size/4);
	}


	printf("\n%d-KB stride, 0 to %dM, for Tex L2 TLB prefetching\n", 256 * page_size/4, 20 * page_size/4);
	for (N = 0; N <= 20971520/4; N += 262144/4 ) {
		stride = 262144/4;
        	parametric_measure_texture((N * page_size/4)+2, iterations, stride * page_size/4);
	}

	printf("\n%d-KB stride, 0 to %dM, for Tex L2 TLB latency\n", 512 * page_size/4, 20* page_size/4);
	for (N = 0; N <= 20971520/4; N += 524288/4 ) {
		stride = 524288/4;
        	parametric_measure_texture((N * page_size/4)+2, iterations, stride * page_size/4);
	}


	return 0;

}
