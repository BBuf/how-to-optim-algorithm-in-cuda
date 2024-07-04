> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第二课: PMPP 书的第1-3章

### PMPP 第一章

![](https://files.mdnice.com/user/59/6d74a444-8832-46a2-93db-02d09dac937a.png)

这一页没什么好说的，就是介绍一些大模型和AI的背景，CPU和GPU的架构区别，以及GPU的出现是为了解决CPU无法通过硬件技术解决的大规模计算性能问题。

![](https://files.mdnice.com/user/59/7b8b26cf-d8b2-42b0-9e04-d7415a92bc23.png)

这张图名为"The Power Wall"（功耗墙），展示了从1970年到2020年间计算机芯片技术的两个关键参数的发展趋势：

晶体管数量（紫色线）：以千为单位，呈指数增长趋势。
频率（绿色线）：以MHz为单位，显示了处理器时钟速度的变化。

"功耗墙"现象：图表底部的注释解释了为什么频率不再持续增长 —— "进一步提高频率会使芯片变得太热而无法有效散热"。

![](https://files.mdnice.com/user/59/e6b81392-0f5d-4209-82df-ad58811997a1.png)

这张slides介绍了CUDA的兴起及其关键特性：

- CUDA 专注于并行程序，适用于现代软件。
- GPU 的峰值 FLOPS 远高于多核 CPU。
- 主要原则是将工作分配给多个线程。
- GPU 注重大规模线程的执行吞吐量。
- 线程较少的程序在 GPU 上表现不佳。
- 将顺序部分放在 CPU 上，数值密集部分放在 GPU 上。
- CUDA 是 Compute Unified Device Architect（统一计算设备架构）。
- 在 CUDA 出现前，使用图形 API（如 OpenGL 或 Direct3D）进行计算。
- 由于 GPU 的广泛可用性，GPU 编程对开发者变得更具吸引力。

![](https://files.mdnice.com/user/59/fdf3c907-8e7e-4f79-a607-b11b06d39f49.png)

这张slides介绍了CUDA编程中的一些挑战：

- 如果不关注性能，并行编程很容易，但优化性能很难。
- 设计并行算法比设计顺序算法更困难，例如并行化递归计算需要非直观的思维方式（如前缀和）。
- 并行程序的速度通常受到内存延迟和吞吐量的限制（内存瓶颈，比如LLM推理的decode）。
- 并行程序的性能可能因输入数据的特性而显著变化。（比如LLM推理有不同长度的序列）。
- 并非所有应用都能轻松并行化，很多需要同步的地方会带来额外的开销（等待时间）。例如有数据依赖的情况。

![](https://files.mdnice.com/user/59/55ca7714-6dfa-4fc5-bd94-db7f92f198e1.png)

《Programming Massively Parallel Processors》这本书的三个主要目标是：

- 教大家并行编程和计算思维
- 并且以正确可靠的形式做到这一点，这包括debug和性能两方面
- 第三点指的应该是如何更好的组织书籍，让读者加深记忆之类的。

虽然这里以GPU作为例子，但这里介绍到的技术也适用于其它加速器。书中使用CUDA例子来介绍和事件相应的技术。


### PMPP 第二章

![](https://files.mdnice.com/user/59/4ff01aec-7940-44c1-ac86-9e8c392ae511.png)

题目是 CH2: 异构数据并行编程
- 异构（Heterogeneous）：结合使用CPU和GPU来进行计算，利用各自的优势来提高处理速度和效率。
- 数据并行性（Data parallelism）：通过将大任务分解为可以并行处理的小任务，实现数据的并行处理。这种方式可以显著提高处理大量数据时的效率。
- 应用示例：
    - 向量加法：这是并行计算中常见的例子，通过将向量的每个元素分别相加，可以并行处理，提高计算速度。
    - 将RGB图像转换为灰度图：这个过程通过应用一个核函数，根据每个像素的RGB值计算其灰度值。公式为 `L = r*0.21 + g*0.72 + b*0.07`，其中L代表亮度（Luminance）。这个转换是基于人眼对不同颜色的感光敏感度不同，其中绿色部分权重最高。

![](https://files.mdnice.com/user/59/9bad0b05-ba27-44e1-8109-03a10e9af79e.png)

这张Slides可以看到所有像素点的计算都是独立的。

![](https://files.mdnice.com/user/59/a5f9ceea-e103-42b4-9ab0-79e9a41e4505.png)

这张Slides介绍了CUDA C的一些特点：
- 扩展了ANSI C的语法,增加了少量的新的语法元素。
- 术语中,CPU表示主机,GPU表示设备。
- CUDA C源代码可以是主机代码和设备代码的混合。
- 设备代码函数称为内核(kernels)。
- 使用线程网格(grid of threads)来执行内核,多个线程并行运行。
- CPU和GPU代码可以并发执行(重叠)。
- 在GPU上可以大量启动多个线程,不需要担心。
- 对于输出张量的每一个元素启动一个线程是很正常的。


![](https://files.mdnice.com/user/59/0da939ef-4d56-4217-a5f5-66615a686802.png)

这张Slides给出了一个向量加法的CUDA C编程示例:

- 向量加法的并行化: 主要概念循环会被映射到多个线程进行独立计算,从而实现易于并行化。
- Naive 的GPU 向量加法步骤:
    - 为向量分配设备内存
    - 将输入从主机传输到设备
    - 启动内核(kernel)进行向量加法运算
    - 将计算结果从设备拷贝回主机
    - 释放设备内存
- 保持数据在GPU上尽可能长的时间,以支持并发的内核启动。这可以最大限度地提高性能。

![](https://files.mdnice.com/user/59/6abe12fa-9a40-406f-9aa0-9011f2064f79.png)

这张Slides展示了每个线程处理一个输出元素的计算，并且是相互独立的。

![](https://files.mdnice.com/user/59/80292cf8-90fc-4950-baf0-9df586b533e5.png)

这张Slides介绍了CUDA编程中内存分配的重要概念:

- NVIDIA设备拥有自己的DRAM(设备全局内存)。
- CUDA提供了两个重要的内存分配函数
    - cudaMalloc(): 在设备全局内存上分配内存空间。
    - cudaFree(): 释放设备全局内存上的内存空间。
- 代码示例中展示了如何使用这两个函数来动态分配和释放浮点型数组的内存空间。
    - size_t size = n * sizeof(float);//计算数组所需的字节数
    - cudaMalloc((void**)&A_d, size);//在设备上分配内存
    - cudaFree(A_d);//释放设备内存

![](https://files.mdnice.com/user/59/ae818eb4-9c6c-48a5-9d25-deb9069257d0.png)

这张Slides介绍了CUDA中内存搬运的API，包括D2H和H2D。一般来说，CUDA程序会先执行H2D的Memcpy把数据搬运到GPU上，然后kernel执行完之后再把结果通过D2H的Memcpy搬运回主机端。


![](https://files.mdnice.com/user/59/88dceae7-b037-49a5-bf7c-85e40e685ba8.png)

这张Slides介绍了CUDA编程中的错误处理机制:

- CUDA函数如果出现错误,会返回一个特殊的错误代码 cudaError_t。如果不是 cudaSuccess，则表示发生了问题。也可以通过这个错误代码获得它的字符串表示形式。
- 编程时,我们需要始终检查 CUDA 函数的返回值,并处理可能出现的错误。

我们在 https://github.com/cuda-mode/lectures/blob/main/lecture_002/vector_addition/vector_addition.cu 这里可以到如何处理错误码：

```c++
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

void vecAdd(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  size_t size = n * sizeof(float);

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  const unsigned int numThreads = 256;
  unsigned int numBlocks = cdiv(n, numThreads);

  vecAddKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
```

![](https://files.mdnice.com/user/59/5b14e221-ef96-42a0-8857-b0952043ab2d.png)

这张Slides介绍了CUDA编程中内核函数(kernel)的基本特点:

- 启动内核函数相当于启动一个由多个线程组成的网格(grid of threads)。
- 所有的线程执行同样的代码,实现了单程序多数据(SPMD)的并行模式。
- 线程以分层的方式组织,分为网格块(grid blocks)和线程块(thread blocks)。
- 每个线程块最多可以包含1024个线程。

![](https://files.mdnice.com/user/59/1c2b0000-aa4a-4da2-8fc1-7a16d750d951.png)

这张Slides讲解了Kernel坐标的几个点：
- 内核中可用的内置变量：blockIdx, threadIdx：这些是CUDA编程中用来标识线程位置的内置变量。blockIdx表示当前线程块的索引，而threadIdx表示当前线程在其所在块中的索引。
- 这些“坐标”允许所有执行相同代码的线程识别要处理的数据部分：通过使用blockIdx和threadIdx，每个线程可以确定它应该处理数据的哪一部分。这对于并行处理非常重要，因为不同的线程可以同时处理不同的数据片段。
- 每个线程可以通过threadIdx和blockIdx唯一标识：threadIdx和blockIdx的组合可以唯一确定一个线程的位置，从而避免不同线程处理相同的数据片段。
- 电话系统类比：将blockIdx视为区号，将threadIdx视为本地电话号码：这种类比帮助理解：blockIdx相当于更大的区域（类似于区号），而threadIdx是在这个区域内的具体线程（类似于本地电话号码）。
- 内置的blockDim告诉我们块中的线程数：blockDim表示每个线程块中包含的线程数。这个变量对于计算每个线程的全局索引是必要的。
- 对于向量加法，我们可以计算线程的数组索引：示例代码：int i = blockIdx.x * blockDim.x + threadIdx.x; 这行代码展示了如何计算每个线程在整个数据数组中的位置。blockIdx.x * blockDim.x计算的是当前块之前所有线程的总数，加上threadIdx.x得到当前线程的全局索引。

![](https://files.mdnice.com/user/59/d52dfd78-ebc6-417e-a835-c67ebc50cb84.png)

这张Slides是对Kernel坐标定位的可视化。我们可以看到每个线程执行相同的代码，仅仅是数据的位置不同。



