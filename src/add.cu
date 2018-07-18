#include "add.cuh"

__global__ void add(int a, int b, int *c)//kernel函数，在gpu上运行。
{
    *c = a + b;
}


int add(int a,int b)
{
    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));//分配gpu的内存，第一个参数指向新分配内存的地址，第二个参数是分配内存的大小。
    add<<<1,1>>>(a, b, dev_c);//调用kernel函数，<<<1,1>>>指gpu启动1个线程块，每个线程块中有1个线程。
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);//将gpu上的数据复制到主机上，
    //即从dev_c指向的存储区域中将sizeof(int)个字节复制到&c指向的存储区域。
    cudaFree(dev_c);//释放cudaMalloc分配的内存。
    return c;
}