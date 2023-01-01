## reduce优化学习笔记

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 以及 [Liu-xiandong的cuda优化工程之reduce部分博客](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/reduce) 做的笔记。

### 问题介绍

通俗的来说，Reduce就是要对一个数组求 sum，min，max，avg 等等。Reduce又被叫作规约，意思就是递归约减，最后获得的输出相比于输入一般维度上会递减。比如 nvidia 博客上这个 Reduce Sum 问题，一个长度为 8 的数组求和之后得到的输出只有一个数，从 1 维数组变成一个标量。本文就以 Reduce Sum 为例来记录 Reduce 优化。

<img width="957" alt="图片" src="https://user-images.githubusercontent.com/35585791/210163655-52f98c65-c4d6-485d-b564-c41d27fd1043.png">

### 硬件环境

NVIDIA A100-PCIE-40GB , 峰值带宽在 1555 GB/s , CUDA版本为11.8.

### 构建BaseLine

在问题介绍一节中的 Reduce 求和图实际上就指出了 BaseLine 的执行方式，我们将以树形图的方式去执行数据累加，最终得到总和。但由于GPU没有针对 global memory 的同步操作，所以博客指出我们可以通过将计算分成多个阶段的方式来避免 global memrory 的操作。如下图所示：

<img width="860" alt="图片" src="https://user-images.githubusercontent.com/35585791/210164111-1cdc34c3-e7f5-4377-883b-8fe38b624bea.png">

接着 NVIDIA 博客给出了 BaseLine 算法的实现：


<img width="864" alt="图片" src="https://user-images.githubusercontent.com/35585791/210164190-84ce2a0e-7d9b-47a5-a20b-e28d0393ba97.png">


这里的 g_idata 表示的是输入数据的指针，而 g_odata 则表示输出数据的指针。然后首先把 global memory 数据 load 到 shared memory 中，接着在 shared memory 中对数据进行 Reduce Sum 操作，最后将 Reduce Sum 的结果写会 global memory 中。

但接下来的这页 PPT 指出了 Baseine 实现的低效之处：

<img width="897" alt="图片" src="https://user-images.githubusercontent.com/35585791/210165315-b6ff93ae-0be9-4a27-b84e-33a92c050775.png">

这里指出了2个问题，一个是warp divergent，另一个是取模这个操作很昂贵。这里的warp divergent 指的是对于启动 BaseLine Kernel 的一个 block 的 warp 来说，它所有的 thread 执行的指令都是一样的，而 BaseLine Kernel 里面存在 if 分支语句，一个 warp 的 所有 thread 都会执行存在的所有分支，但只会保留满足条件的分支产生的结果。

<img width="919" alt="图片" src="https://user-images.githubusercontent.com/35585791/210165714-93dc6ae6-fbc6-416d-8f43-4728891a9ace.png">

我们可以在第8页PPT里面看到，对于每一次迭代都会有两个分支，分别是有竖直的黑色箭头指向的小方块（有效计算的线程）以及其它没有箭头指向的方块，所以每一轮迭代实际上都有大量线程是空闲的，无法最大程度的利用GPU硬件。

接下来我们先把 BaseLine 的代码抄一下，然后我们设定好一个 GridSize 和 BlockSize 启动 Kernel 测试下性能。在PPT的代码基础上，我么补充一下内存申请以及启动 Kernel 的代码。

```c++
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v0(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *output_host = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));
    
    dim3 grid(N / BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v0<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_device, output_host, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}
```

我们这里设定输入数据的长度是 `32*1024*1024` 个float数，然后每个 block 的线程数我们设定为 256 （BLOCK_SIZE = 256， 也就是一个 Block 有 8 个 warp）并且每个 Block 要计算的元素个数也是 256 个，然后当一个 Block 的计算完成后对应一个输出元素。所以对于输出数据来说，它的长度是输入数据长度处以 256 。我们代入 kernel 加载需要的 GridSize（`N / 256`） 和 BlockSize(`256`) 再来理解一下 BaseLine 的 Reduce kernel。

首先，第 `tid` 号线程会把 global memroy 的第 i 号数据取出来，然后塞到 shared memroy 中。接下来针对已经存储到 shared memroy 中的 256 个元素展开多轮迭代，迭代的过程如 PPT 的第8页所示。完成迭代过程之后，这个 block 负责的256个元素的和都放到了 shared memrory 的0号位置，我们只需要将这个元素写回global memory就做完了。

接下来我们使用 `nvcc -o bin/reduce_v0 reduce_v0_baseline.cu` 编译一下这个源文件，并且使用nsight compute去profile一下。

<img width="1247" alt="图片" src="https://user-images.githubusercontent.com/35585791/210174419-8d28f1ff-d469-47d5-862c-1dcc332dd961.png">


性能和带宽的测试情况如下：

|优化手段|耗时(us)|带宽利用率|加速比|
|--|--|--|--|
|reduce_baseline|990.66us|39.57%|~|


