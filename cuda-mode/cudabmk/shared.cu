#include <stdio.h>

#include "repeat.h"

__global__ void shared_latency (unsigned int * my_array, int array_length, int iterations, unsigned long long * duration) {

   unsigned int start_time, end_time;
   int i, k;
   unsigned int j = 0;
   unsigned long long sum_time;

   my_array[array_length - 1] = 0;
   sum_time = 0;
   duration[0] = 0;


   // sdata[] is used to hold the data in shared memory. Dynamically allocated at launch time.
   extern __shared__ unsigned int sdata[];

   for (i=0; i < array_length; i++) {
      sdata[i] = my_array[i];
   }

   j=0;
   for (k= 0; k<= iterations; k++) {
   	  if (k==1) {
	  	sum_time = 0;
	  }

 	  start_time = clock();
	  repeat256(j=sdata[j];);
	  end_time = clock();

   	  sum_time += (end_time -start_time);
   }

   my_array[array_length - 1] = j;
   duration[0] = sum_time;

}

// Shared memory array size is N-2. Last two elements are used as dummy variables.
void parametric_measure_shared(int N, int iterations, int stride) {
	
	int i;
	unsigned int * h_a;
	unsigned int * d_a;

	unsigned long long * duration;
	unsigned long long * latency;

	cudaError_t error_id;

	/* allocate array on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * N);
	latency = (unsigned long long *)malloc(2*sizeof(unsigned long long));

   	/* initialize array elements on CPU */
	for (i = 0; i < N-2; i++) {
		h_a[i] = (i + stride) % (N-2);	
	}
	h_a[N-2] = 0;
	h_a[N-1] = 0;


	/* allocate arrays on GPU */
	cudaMalloc ((void **) &d_a, sizeof(unsigned int) * N);
	cudaMalloc ((void **) &duration, 2*sizeof(unsigned long long));

        cudaThreadSynchronize ();
	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 1 is %s\n", cudaGetErrorString(error_id));
	}

        /* copy array elements from CPU to GPU */
        cudaMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)duration, (void *)latency, 2*sizeof(unsigned long long), cudaMemcpyHostToDevice);
        
	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 2 is %s\n", cudaGetErrorString(error_id));
	}
	
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);

	//printf("Launch kernel with parameters: %d, N: %d, stride: %d\n", iterations, N, stride); 
	int sharedMemSize =  sizeof(unsigned int) * N ;

	shared_latency <<<Dg, Db, sharedMemSize>>>(d_a, N, iterations, duration);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 3 is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();

        cudaMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)latency, (void *)duration, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        cudaThreadSynchronize ();

	/* print results*/


	printf("  %d, %f\n",stride,(double)(latency[0]/(256.0*iterations)));


	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(duration);
	cudaThreadSynchronize ();

        /*free memory on CPU */
        free(h_a);
        free(latency);


}


int main() {

	int N, stride; 

	// initialize upper bounds here
	int stride_upper_bound = 1024; 

	printf("Shared memory latency for varying stride.\n");
	printf("stride (bytes), latency (clocks)\n");

	N = 256;
	stride_upper_bound = N;
	for (stride = 1; stride <= stride_upper_bound; stride += 1) {
		parametric_measure_shared(N+2, 10, stride);
	}

	return 0;

}
