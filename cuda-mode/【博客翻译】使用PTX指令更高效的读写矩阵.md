> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/

> 本文实验cuda代码见：https://github.com/simveit/load_and_store

# 使用PTX指令更高效地加载和存储矩阵

2025年5月14日

## `ldmatrix`

从PTX文档(https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)中我们可以看到,`ldmatrix`可以用于从共享内存中集体加载一个或多个矩阵,以供`mma`指令使用。

指令格式如下

```
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];

ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];

.shape   = {.m8n8, .m16n16};
.num     = {.x1, .x2, .x4};
.ss      = {.shared{::cta}};
.type    = {.b16, .b8};
.dst_fmt = { .b8x16 };
.src_fmt = { .b6x16_p32, .b4x16_p64 };
```

该指令将从`.shared`空间集体加载一个或多个矩阵到寄存器中。

- `p`: 在`.shared`空间中的地址操作数
- `r`: 目标寄存器
- `shape`: 加载矩阵的形状
- `num`: 加载1、2或4个矩阵

可能的数据类型如下：


|**.shape**| 	**矩阵形状**| 	**元素大小**|
|:---:|:---:|:---:|
|`.m8n8`| 	8x8| 	16-bit|
|`.m16n16`| 	16x16| 	8-bit 或 6-bit 或 4-bit|
|`.m8n16`| 	8x16| 	6-bit or 4-bit|

注意，目前只有sm_100及更高版本的GPU支持`m16n16`和`m8n16`的形状。由于我目前没有访问权限，我们将专注于`m8n8`指令。

下表显示了每个线程组负责哪些矩阵。每个地址对应矩阵中的一行。每个"线程组"（即0-7、8-15、16-23和24-31）加载一个单独的矩阵。连续的行应该在内存中连续存储。

|**.num**| 	**Threads 0–7**| 	**Threads 8–15**| 	**Threads 16–23**| 	**Threads 24–31**|
|:---:|:---:|:---:|:---:|:---:|
|`.x1`| 	addr0–addr7| 	–| 	–| 	–|
|`.x2`| 	addr0–addr7| 	addr8–addr15| 	–| 	–|
|`.x4`| 	addr0–addr7| 	addr8–addr15| 	addr16–addr23| 	addr24–addr31|


下图展示了使用`ldmatrix`加载的`8x8`矩阵的fragment布局：

![](https://files.mdnice.com/user/59/f5fbc9ae-36a9-4691-b38e-5cf1f36cda03.png)

```shell
// 使用64位地址加载一个8x8矩阵
.reg .b64 addr;
.reg .b32 d;
ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {d}, [addr];

// 加载两个8x8矩阵，以列主格式
.reg .b64 addr;
.reg .b32 d<2>;
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {d0, d1}, [addr];

// 加载四个8x8矩阵
.reg .b64 addr;
.reg .b32 d<4>;
ldmatrix.sync.aligned.m8n8.x4.b16 {d0, d1, d2, d3}, [addr];
```

## 实现

如上所述，指针应该位于`.shared`空间中。有多种方法将通用指针转换为`.shared`空间。最简单的方法如下(https://forums.developer.nvidia.com/t/problem-about-ptx-instruction-cp-async-ca-shared-global/224219/2):

```c++
size_t asl = __cvta_generic_to_shared(smem+threadIdx.x);
```

我们也可以使用内联汇编：

```c++
asm volatile(".reg .u64 smem_ptr64; cvta.to.shared.u64 smem_ptr64, %0;\n" :: "l"(smem+threadIdx.x));
```

或者像这样：

```c++
asm volatile(".reg .u64 smem_ptr64; cvta.to.shared.u64 smem_ptr64, %0;\n" :: "l"(smem+threadIdx.x)); 
asm volatile(".reg .u32 smem_ptr32; cvt.u32.u64 smem_ptr32, smem_ptr64;\n" ::);
```

我们也可以参考CUTLASS库(https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cute/arch/copy_sm75.hpp#L39)来获取实现思路。

从这里开始,实现就比较直接了:

```c++
#include <cstdint>
#include <iostream>

// 定义一个设备端内联函数，用于从共享内存加载8x8矩阵
// d0: 输出参数，用于存储加载的数据
// address: 输入参数，共享内存中的地址
__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x1_b16(
    uint32_t &d0, const uint32_t &address) {
  // 使用内联PTX汇编指令加载矩阵
  // ldmatrix.sync.aligned.m8n8.x1.shared.b16: 同步加载8x8矩阵，每个元素16位
  // {%0}: 输出寄存器，存储加载的数据
  // [%1]: 输入寄存器，指定共享内存地址
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
               : "=r"(d0)    // 输出约束，表示d0是一个输出寄存器
               : "r"(address)); // 输入约束，表示address是一个输入寄存器
}

// CUDA核函数，用于演示矩阵加载
__global__ void ldmatrix(uint16_t *value) {
  // 定义共享内存大小
  constexpr int N = 64;
  // 声明共享内存数组
  __shared__ uint16_t smem[N];
  // 获取当前线程ID
  auto tid = threadIdx.x;

  // 计算行偏移量：每个线程负责8个元素，所以乘以8
  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  // 计算最终地址：共享内存基址 + 行偏移
  const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;

  // 初始化共享内存
  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
  }
  // 同步所有线程，确保共享内存初始化完成
  __syncthreads();

  // 声明用于存储加载数据的变量
  uint32_t frag;
  // 调用矩阵加载函数
  ldmatrix_sync_aligned_m8n8_x1_b16(frag, address);

  // 再次同步，确保所有线程都完成加载
  __syncthreads();

  // 从32位数据中提取两个16位值
  // 提取低16位
  uint16_t number1 = static_cast<uint16_t>(frag & 0xFFFF);
  // 提取高16位
  uint16_t number2 = static_cast<uint16_t>((frag >> 16) & 0xFFFF);
  // 打印结果
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
}

// 主函数
int main() {
  // 声明设备端指针
  uint16_t *d_value;
  // 分配设备内存
  cudaMalloc(&d_value, sizeof(uint16_t));
  // 启动核函数，使用1个块，32个线程
  ldmatrix<<<1, 32>>>(d_value);
  // 等待设备完成
  cudaDeviceSynchronize();
  // 释放设备内存
  cudaFree(d_value);
  return 0;
}
```

注意，根据上表，线程0-7需要对应于前8行的地址：

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;
```

我们将在加载时传递地址和fragment。注意，每个fragment有`32bit`，我们可以通过先使用全16位掩码来提取最后16位，然后右移并再次执行相同的操作来提取前16位来输出加载的fragment。

```shell
0 -> 0  0   1   
1 -> 2  2   3   
2 -> 4  4   5   
3 -> 6  6   7   
4 -> 8  8   9   
5 -> 10  10   11   
6 -> 12  12   13   
7 -> 14  14   15   
8 -> 16  16   17   
9 -> 18  18   19   
10 -> 20  20   21   
11 -> 22  22   23   
12 -> 24  24   25   
13 -> 26  26   27   
14 -> 28  28   29   
15 -> 30  30   31   
16 -> 32  32   33   
17 -> 34  34   35   
18 -> 36  36   37   
19 -> 38  38   39   
20 -> 40  40   41   
21 -> 42  42   43   
22 -> 44  44   45   
23 -> 46  46   47   
24 -> 48  48   49   
25 -> 50  50   51   
26 -> 52  52   53   
27 -> 54  54   55   
28 -> 56  56   57   
29 -> 58  58   59   
30 -> 60  60   61   
31 -> 62  62   63
```

我们可以看到每个寄存器包含两个值。

我们可以在一个warp中同时写入两个矩阵。我们需要考虑到地址是按线程组提供的:

|**.num**| 	**线程 0-7**| 	**线程 8-15**| 	**线程 16-23**| 	**线程 24-31**|
|:---:|:---:|:---:|:---:|:---:|
|.x1| 	addr0–addr7| 	–| 	–| 	–|
|.x2| 	addr0–addr7| 	addr8–addr15| 	–| 	–|
|.x4| 	addr0–addr7| 	addr8–addr15| 	addr16–addr23| 	addr24–addr31|


使用`x2`的`ldmatrix`语法如下所示

```c++
__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, const uint32_t &address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(d0), "=r"(d1)
               : "r"(address));
}
```

注意，现在我们写入`32bit` fragment。

我们可以将这个包装成一个kernel:

```c++
__global__ void ldmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[2 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
    smem[i + 64] = i + 64;
  }
  __syncthreads();

  uint32_t frag1;
  uint32_t frag2;
  ldmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
}
```

计算地址的逻辑如下:

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
const uint32_t offset = offset_rows + offset_matrix;
const uint32_t address = __cvta_generic_to_shared(smem) + offset;
```

我们需要计算行偏移和矩阵偏移。前8个线程提供第一个矩阵的地址。接下来的8个线程提供第二个矩阵的地址。

我们可以非常相似地加载4个`8x8`矩阵。语法如下:

```c++
__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(address));
}
```

完整的kernel如下所示:

```c++
__global__ void ldmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[4 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
    smem[i + 64] = i + 64;
    smem[i + 128] = i + 128;
    smem[i + 192] = i + 192;
  }
  __syncthreads();

  uint32_t frag1;
  uint32_t frag2;
  uint32_t frag3;
  uint32_t frag4;
  ldmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, frag3, frag4, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
  uint16_t number5 = static_cast<uint16_t>(frag3 & 0xFFFF);
  uint16_t number6 = static_cast<uint16_t>((frag3 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 128]),
         (int)number5, (int)number6);
  uint16_t number7 = static_cast<uint16_t>(frag4 & 0xFFFF);
  uint16_t number8 = static_cast<uint16_t>((frag4 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 192]),
         (int)number7, (int)number8);
}
```



地址计算类似。我们再次有8个线程组，每个线程组提供4个矩阵的8行地址，因此总共`32`个线程在warp中提供地址。

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
const uint32_t offset = offset_rows + offset_matrix;
const uint32_t address = __cvta_generic_to_shared(smem) + offset;
```

每个kernel可以像这样调用:

```c++
int main() {
  uint16_t *d_value;
  cudaMalloc(&d_value, sizeof(uint16_t));
  ldmatrix<<<1, 32>>>(d_value);
  cudaDeviceSynchronize();
  cudaFree(d_value);
  return 0;
}
```

## stmatrix

`stmatrix`是一个PTX指令，用于将一个或多个矩阵集体存储到共享内存中。

```c++
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;

.shape  = {.m8n8, .m16n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```



如你所见，指令与`ldmatrix`类似。`.m8n8`在Hopper中可用，`m16n8`在Blackwell GPU中可用。

地址的提供方式与上面相同。这次我们提供地址来知道提供寄存器(s)的内容存储到哪个位置。

|**.num**|**Threads 0–7**|**Threads 8–15**|**Threads 16–23**|**Threads 24–31**|
|--------|-----------|------------|-------------|--------------|
|`.x1`|addr0–addr7|–|–|–|
|`.x2`|addr0–addr7|addr8–addr15|–|–| 
|`.x4`|addr0–addr7|addr8–addr15|addr16–addr23|addr24–addr31|

### 实现

一旦我们正确理解了上面的`ldmatrix`指令，实现并不困难。请确保您理解了上面的代码，然后再继续阅读。

下面的代码给出了一个简单的PTX指令包装器，并存储一个`8x8`矩阵。

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x1_b16(
    uint32_t &d0, const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(address),
      "r"(d0));
}
```

我们可以将这个包装成一个kernel:

```c++
__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;

  uint32_t frag = 0x00000000;
  frag |= (tid * 2 + 0);
  frag |= (tid * 2 + 1) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x1_b16(frag, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
}
```

大部分代码与上面类似。但这次我们定义一个fragment，并将其存储到共享内存中的地址。

下面的代码初始化一个32位无符号整数。我们将首先初始化前16位为`2 * tid + 0`，然后初始化最后16位为`2 * tid + 1`。这主要是为了与`ldmatrix`示例中的相同结果。

```c++
uint32_t frag = 0x00000000;
frag |= (tid * 2 + 0);
frag |= (tid * 2 + 1) << 16;
```

我们将片段存储到给定的地址。这将输出:

```shell
0 -> 0  0   1   
1 -> 2  2   3   
2 -> 4  4   5   
3 -> 6  6   7   
4 -> 8  8   9   
5 -> 10  10   11   
6 -> 12  12   13   
7 -> 14  14   15   
8 -> 16  16   17   
9 -> 18  18   19   
10 -> 20  20   21   
11 -> 22  22   23   
12 -> 24  24   25   
13 -> 26  26   27   
14 -> 28  28   29   
15 -> 30  30   31   
16 -> 32  32   33   
17 -> 34  34   35   
18 -> 36  36   37   
19 -> 38  38   39   
20 -> 40  40   41   
21 -> 42  42   43   
22 -> 44  44   45   
23 -> 46  46   47   
24 -> 48  48   49   
25 -> 50  50   51   
26 -> 52  52   53   
27 -> 54  54   55   
28 -> 56  56   57   
29 -> 58  58   59   
30 -> 60  60   61   
31 -> 62  62   63   
```

这证实了我们的实现与上面的`ldmatrix`操作相反。

存储到2或4个矩阵的实现非常相似:

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::"r"(address),
      "r"(d0), "r"(d1));
}

// CUDA核函数，用于演示矩阵存储
__global__ void stmatrix(uint16_t *value) {
  // 定义共享内存大小
  constexpr int N = 64;
  // 声明共享内存数组，大小为2*N以存储两个矩阵
  __shared__ uint16_t smem[2 * N];
  // 获取当前线程ID
  auto tid = threadIdx.x;

  // 计算行偏移量：每个线程负责8个元素，所以乘以8
  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  // 计算矩阵偏移量：根据线程组(0-7或8-15)选择第一个或第二个矩阵
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
  // 计算总偏移量
  const uint32_t offset = offset_rows + offset_matrix;
  // 计算最终地址：共享内存基址 + 总偏移
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  // 初始化第一个数据片段
  uint32_t frag1 = 0x00000000;
  // 设置低16位为 2*tid
  frag1 |= (tid * 2 + 0);
  // 设置高16位为 2*tid+1
  frag1 |= (tid * 2 + 1) << 16;
  
  // 初始化第二个数据片段
  uint32_t frag2 = 0x00000000;
  // 设置低16位为 2*tid+64
  frag2 |= (tid * 2 + 0 + 64);
  // 设置高16位为 2*tid+65
  frag2 |= (tid * 2 + 1 + 64) << 16;
  
  // 同步所有线程，确保数据准备完成
  __syncthreads();

  // 调用矩阵存储函数，将两个片段写入共享内存
  stmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, address);

  // 再次同步，确保所有线程都完成存储
  __syncthreads();

  // 从第一个32位片段中提取两个16位值
  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);  // 提取低16位
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);  // 提取高16位
  // 打印第一个矩阵的结果
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
         
  // 从第二个32位片段中提取两个16位值
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);  // 提取低16位
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);  // 提取高16位
  // 打印第二个矩阵的结果
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
}
```

对于四个矩阵的情况

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(
          address),
      "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[4 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  uint32_t frag1 = 0x00000000;
  frag1 |= (tid * 2 + 0);
  frag1 |= (tid * 2 + 1) << 16;
  uint32_t frag2 = 0x00000000;
  frag2 |= (tid * 2 + 0 + 64);
  frag2 |= (tid * 2 + 1 + 64) << 16;
  uint32_t frag3 = 0x00000000;
  frag3 |= (tid * 2 + 0 + 128);
  frag3 |= (tid * 2 + 1 + 128) << 16;
  uint32_t frag4 = 0x00000000;
  frag4 |= (tid * 2 + 0 + 192);
  frag4 |= (tid * 2 + 1 + 192) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x4_b16(frag1, frag2, frag3, frag4, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
  uint16_t number5 = static_cast<uint16_t>(frag3 & 0xFFFF);
  uint16_t number6 = static_cast<uint16_t>((frag3 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 128]),
         (int)number5, (int)number6);
  uint16_t number7 = static_cast<uint16_t>(frag4 & 0xFFFF);
  uint16_t number8 = static_cast<uint16_t>((frag4 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 192]),
         (int)number7, (int)number8);
}
```

我们需要做的就是初始化更多的fragments。当存储到2个矩阵时我们需要提供2个fragments,当存储到4个矩阵时我们需要提供4个fragments。

## 结论

我希望这篇博客对以下方面有所帮助:
- 详细理解ldmatrix和stmatrix指令
- 通过观察这两条指令之间的对称性来加深理解。如果你想了解更多,可以参考PTX文档。我很乐意收到你对这篇博客的反馈,并讨论CUDA及相关话题。如果你想联系我,可以在LinkedIn上添加我(https://www.linkedin.com/in/simon-veitner-174a681b6/)。代码可以在我的代码仓库(https://github.com/simveit/load_and_store)中找到。














