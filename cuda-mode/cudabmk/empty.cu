#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

__global__ void kempty()
{
    return;
}

void measure_empty()
{ 
    struct timeval t1, t2;    


    // Start empty kernel
    dim3 Db = dim3(512);
    dim3 Dg = dim3(16,16,16);
    fprintf (stderr, "Running (%d x %d x %d) blocks of %d empty threads...", Dg.x, Dg.y, Dg.z, Db.x);

    gettimeofday(&t1, NULL);
    kempty <<<Dg, Db>>>();
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);

	fprintf (stderr, "done\n");	

    printf ("Running (%d x %d x %d) blocks of %d empty threads: %.3f ms\n", Dg.x, Dg.y, Dg.z, Db.x, 
             (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)*0.001);
    
    printf ("\n");		 
}

