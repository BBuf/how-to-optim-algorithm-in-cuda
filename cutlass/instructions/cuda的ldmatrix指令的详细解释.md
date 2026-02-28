> 来自：https://zhuanlan.zhihu.com/p/697228676 ,我做了一些修改和注释

## cuda的ldmatrix指令的详细解释

ldmatrix指令是 PTX 级别的指令，它是个warp级别的数据加载指令，当然数据是从shared memory中加载到32个cuda thread中的寄存器中。

### 1. ldmatrix指令的使用格式例子: ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0 }, [ %1 ];

- 直接看例子吧，例如这个指令`ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0 }, [ %1 ];`
- 他的意思就是让一个warp中的32个线程，从shared memory中加载1(`x1`)个`8*8(m8n8)`的矩阵，这个矩阵有8行！
- 每行的8个元素必须连续存放，不同行之间可以不连续
- 这个矩阵的元素粒度是b16，也就是一个元素占两个字节。
- 那也就是说这个warp读取了8*8=64个元素，warp中的每个cuda thread占据了两个元素，因此正好是一个32位的寄存器！
- 由于每行的位置可以不连续，因此用户需要指定8行的开始地址，这个需要用户确保`thread0-thread7`的`%1`寄存器填充的是这8个地址即可
- 至于`thread 8-thread 31`的`%1`寄存器数据，那就可以随便设置了！
- 返回的32位数据，也就是在`0%`寄存器里了。
- 我们好奇的第一个问题就是，这64个元素是如何分布在32个cuda thread之间的呢？
- 答案就是看下面这个图即可，红色的表示线程id，每个线程被占据了两个数据，也就是正好是1个32位寄存器！
- 图上只给出了thread0-thread7的情况，其他线程的占据的数据以此类推即可，我这里省略了。
- 图中一共是8行，注意每行的首地址必须由`thread0-thread7`的`%1`寄存器指定!

![](https://files.mdnice.com/user/59/d4b35fc7-0ee5-47af-96e0-7bc96ba03508.png)

### 2. ldmatrix指令的使用格式例子：ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ]

- 这个例子使用更广泛，他的意思就是让一个warp中的32个线程，从shared memory中加载4个8*8的矩阵，矩阵的元素仍然是16位！
- 因此这个时候需要指定32个地址了，也就是thread0-thread31的`%4`寄存器需要指定这32个地址！
- 同时每个cuda thread应该瓜分到`4*8*8/32=8`个bf16元素，也就是4个32位寄存器啦！
- 所以他的返回值有四个寄存器了！

#### ldmatrix.sync.aligned指令的例子

- 下面的代码非常简单，值得一看。编译命令是 `nvcc A.cu -arch sm_80`
- 下面的代码就是让一个warp中的32个线程，从shared memory中加载4个`8*8`的矩阵，但是我这个里面写的是`uint32_t`元素类型，因此其实是4个`8*4`的矩阵！
- 下面代码打印出线程1占据的四个32位寄存器的值！

```c
#include <stdio.h>
#include <iostream>

__global__ void helloFromGPU (void)
{
  __shared__ uint32_t aTile[4*8*4];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的4*8*4的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 4*8*4; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx % 16 * 8 + tidx / 16 * 4;
  uint32_t a[4];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
  : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) 
  : "r"(smem)
  );

  if (tidx == 1) {
    printf("%d \n", (a[0])); printf("%d \n", (a[1]));
    printf("%d \n", (a[2])); printf("%d \n", (a[3]));
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, block>>>();

cudaDeviceReset();
return 0;
}
```

- 上面代码中`int aTile_index = tidx % 16 * 8 + tidx / 16 * 4;`这个代码是让
每个线程分别指向每行的首地址！
其中`tidx % 16` 表示的是行id，
`tidx / 16 * 4` 表示的是列id，
- 从按照b16的角度来看，其实列id应该是`tidx / 16 * 8`，
- 但是我代码里面写的是uint32类型，所以就是tidx / 16 * 4了！
- 最后打印出来的四个数字如下图所示哦！


![](https://files.mdnice.com/user/59/e07a31bd-b985-40d6-b6e9-f7cc784bd1ba.png)


- 上面的使用例子我的数据单元设置成`uint32_t`了，用户可能不太好直接理解，下面我补充一个例子，直接针对`fp16`元素进行`ldmatrix`指令的例子。
- 下面例子的代码我起名字叫`B.cu`，编译命令是 `nvcc B.cu -arch sm_80`

```c
#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"

__global__ void helloFromGPU (void) {
  __shared__ half aTile[4*8*8];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的4*8*8的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 4*8*8; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx % 16 * 16 + tidx / 16 * 8;
  uint32_t my_register[4];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
  : "=r"(my_register[0]), "=r"(my_register[1]), "=r"(my_register[2]), "=r"(my_register[3]) 
  : "r"(smem)
  );

  if (tidx == 1) {
    for (int i = 0; i < 4; i++) {
        half * tmp = (half*)(&(my_register[i]));
        printf("%f\n", (float)(tmp[0]));
        printf("%f\n", (float)(tmp[1]));
    }
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, block>>>();

cudaDeviceReset();
return 0;
}

```

![](https://files.mdnice.com/user/59/4fd73336-4318-4325-93b0-31c9b72a9610.png)


#### 为什么索引计算方式是：`int aTile_index = tidx % 16 * 16 + tidx / 16 * 8;`，这里的`*8`是什么意思?

- 这个公式是用来计算每个线程在共享内存中应该访问的起始位置。让我们逐步分析：
1. 首先要理解数据布局：
   - 我们有4个8×8的矩阵（使用half类型）
   - 每个矩阵有8行，每行8个half元素
   - 一个warp（32个线程）需要协同工作来加载这些数据
2. 公式拆解：
   - tidx % 16：将32个线程分成两组，得到0-15的值，表示在前16个线程中的位置
   - tidx / 16：将32个线程分成两组，得到0或1，表示线程在前半部分还是后半部分
   - 16：每行有8个half元素，所以16字节（因为每个half是2字节）
   - 8：每个线程组负责的行偏移
3. 为什么是`* 8`：
   - 每个half元素占2字节
   - 一行有8个half元素
   - 所以当我们要移动到下一组行时，需要跳过8个元素
   - 这个`* 8`确保了第二组线程（`tidx/16 == 1`）会访问正确的行

让我们用一个具体的例子来说明：
假设我们有一个8×8的矩阵（用half类型），数据布局如下：

```shell
[0,  1,  2,  3,  4,  5,  6,  7]    // 第0行
[8,  9,  10, 11, 12, 13, 14, 15]   // 第1行
[16, 17, 18, 19, 20, 21, 22, 23]   // 第2行
...
```

- 对于线程0（tidx=0）：
  - `tidx % 16 * 16 = 0 * 16 = 0`
  - `tidx / 16 * 8 = 0 * 8 = 0`
  - 最终索引 = 0，指向第一行的开始
- 对于线程1（tidx=1）：
  - `tidx % 16 * 16 = 1 * 16 = 16`
  - `tidx / 16 * 8 = 0 * 8 = 0`
  - 最终索引 = 16，指向第二行的开始
- 对于线程16（tidx=16）：
  - `tidx % 16 * 16 = 0 * 16 = 0`
  - `tidx / 16 * 8 = 1 * 8 = 8`
  - 最终索引 = 8，指向另一个区域的开始

这样的索引计算方式确保了：
- 每个线程都能访问到正确的数据位置
- 数据访问是连续的（对于每一行）
- 满足了`ldmatrix`指令对数据布局的要求
- 能够高效地将数据从共享内存加载到寄存器中

这种计算方式是为了配合`ldmatrix`指令的工作方式，使得warp中的32个线程能够协同工作，高效地完成矩阵数据的加载操作。


#### 为什么需要这样的指令呢？

- 为啥需要`ldmatrix`指令呢？是因为这个指令主要和`mma`指令搭配使用的！
- 也就是先用`ldmatrix`指令将数据从shared memory中加载到寄存器，然后调用 `mma` 指令计算！
- 请我们来看一下这个链接 9.7.13.4.8. Matrix Fragments for mma.m16n8k16 with floating point type(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)，这个链接上展示了`mma.m16n8k16`指令。
- 这个指令的功能是计算A矩阵`16*16`和B矩阵`16*8`，然后得到一个`16*8`的矩阵C。
- 其中A矩阵`16*16`的元素分布在32个cuda thread的寄存器中，每个线程占据着8个元素，那么这么多元素是如何分布在32个线程中的呢？
- 就是下面这个图片所示哦


![](https://files.mdnice.com/user/59/1d5299c3-07be-4595-a668-972ea4638606.jpg)

- 从上图可以看出，假设你把上面的16 * 16 初始化成0-255，当然啦，是先行后列，mma指令对于输入在32个cuda thread之间的分布恰好就是B.cu那个例子似的！
- B矩阵16*8的元素分布在32个cuda thread的寄存器中，那么这么多元素是如何分布在32个cuda thread的寄存器中的呢？
- 就是下面这个图片所示哦

![](https://files.mdnice.com/user/59/7652e737-15ea-4b1f-881d-967a65c107f7.png)

- 从上图可以看出，mma指令对于输入在32个cuda thread之间的分布恰好就是 ldmatrix指令那样！
- 当然啦，这个要求smem中B矩阵必须是col major的哦！，否则就不能调用ldmatrix指令哦！
- 其实也是可以调用ldmatrix指令的啦！只不过要加上一个trans，也就是 ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
- 具体看我的图例！

![](https://files.mdnice.com/user/59/b8bda19f-9808-49d6-b919-77e699a8d5cd.png)


- 加了`trans`后，数据在每个线程上的划分就像B矩阵那样了！
- 代码如下，我起名字叫C.cu，编译命令是 `nvcc C.cu -arch sm_80`

```c
#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"

__global__ void helloFromGPU (void) {
  __shared__ half aTile[2*8*8];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的2*8*8的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 2*8*8; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx * 8;
  uint32_t my_register[2];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [ %2 ];\n"
  : "=r"(my_register[0]), "=r"(my_register[1])
  : "r"(smem)
  );

  if (tidx == 4) {
    for (int i = 0; i < 2; i++) {
        half * tmp = (half*)(&(my_register[i]));
        printf("%f\n", (float)(tmp[0]));
        printf("%f\n", (float)(tmp[1]));
    }
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, block>>>();

cudaDeviceReset();
return 0;
}

```

ldmatrix指令的官方文档如下：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

#### 为什么int8矩阵当A是rowmajor B是col major性能最好呢？

- 我个人觉得是因为ldmatrix指令只支持16byte的转置，不支持8byte的转置，所以int8矩阵要求A是rowmajor B是col major。

#### 为什么sm75以上的架构的mma指令都只支持A和B为一种layout呢？

![](https://files.mdnice.com/user/59/9c7e70c7-93a4-451f-9560-69e19ec27eb4.jpg)

因为有`ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`指令啦！



















