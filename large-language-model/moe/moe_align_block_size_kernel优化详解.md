# sgl-kernel MoE Align Block Size Kernel 优化过程解析

## 0x0. 前言

这篇文章记录了SGLang的sgl-kernel中优化 `moe_align_kernel.cu` 的过程（https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu）。MoE模型里面有个很关键的kernel就是 `moe_align_block_size`,它的作用是把tokens按照expert分组对齐,为后面的expert计算做准备。

这个kernel从最初的baseline版本一路优化到现在,经历了几个版本的迭代:
- 0x1 Baseline: 最初的实现，小expert（num_expert <= 64 && token <= 1024）的时候我基本沿用了vLLM的实现，并做了一些访存合并访问的调整。其它情况下新起了一个kernel，以warp为单位来处理。
- 0x2: 加了向量化的padding操作
- 0x3: 用Blelloch Scan算法把前缀和计算并行化了
- 0x4: 进一步用Warp Scan减少同步开销,这是目前性能最好的版本

**需要指出的是Baseline版本是我完成的。然后0x2和0x3的关键优化是 https://github.com/ispobock 完成的。0x4的关键优化是 https://github.com/yuan-luo 完成的。**

下面会详细讲讲每个版本的优化思路和实现细节。

## 0x1. Baseline Kernel 详解

### 这个kernel到底在干啥

简单来说,这个kernel要做4件事:
1. 统计每个expert有多少个token
2. 计算对齐后的前缀和(要按block_size对齐)
3. 生成expert_ids数组,记录每个block对应哪个expert
4. 把tokens按expert分组排序

Baseline版本用了3个kernel来完成这些工作:

#### 1. `moe_align_block_size_kernel` - 主 kernel

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,      // 输入: 每个token对应的expert id
    int32_t* __restrict__ sorted_token_ids,     // 输出: 排序后的token索引
    int32_t* __restrict__ expert_ids,           // 输出: 每个block对应的expert id
    int32_t* __restrict__ total_tokens_post_pad,// 输出: 对齐后的总token数
    int32_t num_experts,                        // expert总数
    int32_t padded_num_experts,                 // 对齐到warp_size的expert数
    int32_t experts_per_warp,                   // 每个warp处理的expert数量
    int32_t block_size,                         // 对齐的block大小
    size_t numel,                               // 输入token总数
    int32_t* __restrict__ cumsum) {             // 输出: 前缀和数组
  
  extern __shared__ int32_t shared_counts[];
  
  // 步骤1: 初始化共享内存计数器
  // 每个warp负责experts_per_warp个expert的计数
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  
  // 初始化当前warp负责的expert计数为0
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  
  __syncthreads();
  
  // 步骤2: 统计每个expert的token数量
  // 所有线程协作,遍历所有tokens,使用原子加统计
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];  // 获取当前token的expert id
    // 计算该expert在shared_counts中的位置
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    // 原子加操作,统计该expert的token数量
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }
  
  __syncthreads();
  
  // 步骤3: 计算前缀和(只用thread 0执行)
  // 前缀和用于确定每个expert在输出中的起始位置
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      
      // 按block_size对齐: CEILDIV(count, block_size) * block_size
      // 这样可以保证每个expert的token数是block_size的倍数
      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }
  
  __syncthreads();
  
  // 步骤4: 填充expert_ids数组
  // expert_ids[i]表示第i个block对应的expert编号
  if (threadIdx.x < num_experts) {
    // 每个线程负责一个expert
    // 从cumsum[threadIdx.x]到cumsum[threadIdx.x+1]的所有block都属于expert threadIdx.x
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}
```

这个kernel的几个关键设计:
- 用共享内存存每个expert的token计数,减少全局内存访问
- 统计的时候用原子操作,避免多个线程同时写导致的问题
- 前缀和计算是串行的,只有thread 0在干活,这是后面优化的重点
- expert_ids填充是并行的,每个线程负责一个expert

#### 2. `count_and_sort_expert_tokens_kernel` - 排序 kernel

```cpp
template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  
  // 遍历所有tokens
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // 使用原子加获取当前token在该expert中的位置
    // cumsum_buffer[expert_id]记录了该expert已经放置的token数量
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    // 将token索引i放到对应位置
    sorted_token_ids[rank_post_pad] = i;
  }
}
```

这个kernel就是把tokens按expert分组排序,用原子操作保证线程安全。

#### 3. `moe_align_block_size_small_batch_expert_kernel` - 小规模优化版本

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
  
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 前缀和数组
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);  // token计数数组
  
  // 步骤1: 初始化每个线程的局部计数器
  // tokens_cnts布局: [blockDim.x+1][num_experts]
  // tokens_cnts[(threadIdx.x + 1) * num_experts + i] 存储线程threadIdx.x对expert i的计数
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }
  
  // 步骤2: 每个线程统计自己负责的tokens
  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }
  
  __syncthreads();
  
  // 步骤3: 对每个expert,累加所有线程的计数(前缀和)
  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      // 累加前缀和,tokens_cnts[i * num_experts + threadIdx.x]最终存储
      // 前i个线程对expert threadIdx.x的总计数
      tokens_cnts[i * num_experts + threadIdx.x] += 
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }
  
  __syncthreads();
  
  // 步骤4: 计算对齐后的前缀和
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + 
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }
  
  __syncthreads();
  
  // 步骤5: 填充expert_ids
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  
  // 步骤6: 排序tokens(直接在kernel中完成,避免额外的kernel调用)
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // 计算当前token在输出中的位置
    // tokens_cnts[threadIdx.x * num_experts + expert_id]: 当前线程之前的线程对该expert的计数
    // cumsum[expert_id]: 该expert在输出中的起始位置
    int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + 
                            cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    // 更新计数,为下一个token做准备
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}
```

这个版本适合小规模场景,也就是 `numel < 1024 && num_experts <= 64` 的情况。

主要优化点:
- 把统计、前缀和、排序全融合到一个kernel里
- 用线程局部计数,避免原子操作的开销
- 减少kernel启动次数

### Baseline版本的性能瓶颈在哪

分析下来主要有2个问题:
1. 前缀和计算是串行的,只有thread 0在干活,其他线程都在摸鱼,完全没利用并行性
2. 统计阶段用了大量atomicAdd,原子操作开销不小

---

## 0x2. 加上向量化Padding

PR: https://github.com/sgl-project/sglang/pull/7437

### 这个版本改了啥

Baseline版本还有个问题,就是`sorted_token_ids`是在python层初始化为numbel的,相当于会多调用一个fill的kernel。0x2版本加了向量化的padding操作,在kernel中直接padding,减少了fill的开销:

```cpp
#define VEC_SIZE 4
using Vec = AlignedArray<int32_t, VEC_SIZE>;

// 在moe_align_block_size_kernel中新增的padding代码
if (pad_sorted_token_ids) {
    int32_t fill_val = static_cast<int32_t>(numel);  // 使用numel作为填充值
    int32_t total = *total_tokens_post_pad;
    
    // 准备向量化的填充值
    Vec fill_vec;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      fill_vec.data[i] = fill_val;
    }
    
    // 向量化写入,一次写4个int32_t
    int32_t total_vec_count = (total + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    
    for (int32_t idx = tid; idx < total_vec_count; idx += stride) {
      out_ptr[idx] = fill_vec;  // 一次写入16字节
    }
  }
```

为啥要这么做:
- 向量化访存: 用`int4`一次写4个int32_t,内存带宽利用率提升4倍
- 合并内存事务: 向量化写入可以合并多个内存事务,减少延迟
- 填充值用`numel`,后面可以识别哪些是padding

`AlignedArray`这个模板类的作用就是保证数组按16字节对齐,和int4其实等价,编译器可以生成更高效的向量化指令。

对两个kernel都加了padding支持:主kernel在计算完expert_ids后用向量化方式填充整个sorted_token_ids数组,后面的排序kernel会覆盖有效位置。小批量kernel在排序前先padding,然后排序操作覆盖有效的token位置。

这个优化的好处是向量化写入提高内存带宽利用率,保证输出数据确定性,padding操作和其他计算并行,几乎没额外开销。

---

## 0x3. 用Blelloch Scan并行化前缀和

PR: https://github.com/sgl-project/sglang/pull/7794

### 这版本解决了啥问题

0x2版本前缀和还是串行的,只有thread 0在干活,这是最大的性能瓶颈。0x3版本引入了Blelloch Scan算法,把前缀和计算并行化了。

### Blelloch Scan算法怎么工作的

Blelloch Scan是个很经典的并行前缀和算法,分两个阶段:

- 阶段1: Up-Sweep (Reduce Phase)

构建求和树,自底向上计算部分和:

```
原始数据: [3, 1, 7, 0, 4, 1, 6, 3]
         
Step 1:   [3, 4, 7, 7, 4, 5, 6, 9]  // 相邻元素两两相加
Step 2:   [3, 4, 7, 11, 4, 5, 6, 14] // 间隔2相加
Step 3:   [3, 4, 7, 11, 4, 5, 6, 25] // 间隔4相加,得到总和
```

- 阶段2: Down-Sweep (Distribution Phase)

自顶向下分配前缀和。这个阶段的核心思想是: 将总和分配到各个位置,计算每个位置之前所有元素的和 。

详细步骤说明:

```
Up-sweep结束后: [3, 4, 7, 11, 4, 5, 6, 25]  // 最后元素是总和25

Step 0: 将最后元素置0,开始down-sweep
        [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep的操作: 对于索引对(ai, bi),执行:
  temp = arr[ai]
  arr[ai] = arr[bi]      // ai位置接收bi的值
  arr[bi] = arr[bi] + temp  // bi位置累加原ai的值

Step 1: stride=4, 处理间隔为8的元素对
        索引对: (3, 7)
        temp = 11, arr[3] = 0, arr[7] = 0 + 11 = 11
        结果: [3, 4, 7, 0, 4, 5, 6, 11]
        
        解释: 索引7之前有11个元素(索引0-3的总和)

Step 2: stride=2, 处理间隔为4的元素对
        索引对: (1, 3), (5, 7)
        
        对(1, 3): temp = 4, arr[1] = 0, arr[3] = 0 + 4 = 4
        对(5, 7): temp = 5, arr[5] = 11, arr[7] = 11 + 5 = 16
        结果: [3, 0, 7, 4, 4, 11, 6, 16]
        
        解释: 
        - 索引3之前有4个元素(索引0-1的总和)
        - 索引7之前有16个元素(索引0-5的总和)

Step 3: stride=1, 处理间隔为2的元素对
        索引对: (0, 1), (2, 3), (4, 5), (6, 7)
        
        对(0, 1): temp = 3, arr[0] = 0, arr[1] = 0 + 3 = 3
        对(2, 3): temp = 7, arr[2] = 4, arr[3] = 4 + 7 = 11
        对(4, 5): temp = 4, arr[4] = 11, arr[5] = 11 + 4 = 15
        对(6, 7): temp = 6, arr[6] = 16, arr[7] = 16 + 6 = 22
        
最终结果: [0, 3, 4, 11, 11, 15, 16, 22]  // Exclusive prefix sum!

验证:
- arr[0] = 0 (前面没有元素)
- arr[1] = 3 (索引0的值)
- arr[2] = 3+1 = 4 (索引0-1的和)
- arr[3] = 3+1+7 = 11 (索引0-2的和)
- arr[4] = 3+1+7+0 = 11 (索引0-3的和)
- ...
```

关键理解:
- Down-sweep 是 Up-sweep 的"逆过程"
- 每一步都在"分配"前面累积的和
- 通过交换和累加,巧妙地计算出每个位置的前缀和

时间复杂度: O(n) 工作量, O(log n) 深度 (并行)

### 代码实现详解

```cpp
// Up-Sweep Phase: 构建求和树
int offset = 1;
#pragma unroll
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  if (tid < d) {
    int ai = offset * (2 * tid + 1) - 1;
    int bi = offset * (2 * tid + 2) - 1;
    scan_buf[bi] += scan_buf[ai];  // 累加求和
  }
  offset <<= 1;
  __syncthreads();
}

// 保存总和并置0
if (tid == 0) {
  prefix[num_experts] = scan_buf[scan_size - 1];
  scan_buf[scan_size - 1] = 0;
}
__syncthreads();

// Down-Sweep Phase: 分配前缀和
#pragma unroll
for (int d = 1; d < scan_size; d <<= 1) {
  offset >>= 1;
  if (tid < d) {
    int ai = offset * (2 * tid + 1) - 1;
    int bi = offset * (2 * tid + 2) - 1;
    if (bi < scan_size) {
      int temp = scan_buf[ai];
      scan_buf[ai] = scan_buf[bi];
      scan_buf[bi] += temp;
    }
  }
  __syncthreads();
}
```

### 关键优化点

#### 1. 并行前缀和计算

0x2版本 (串行):
```cpp
if (threadIdx.x == 0) {
  cumsum[0] = 0;
  for (int i = 1; i <= num_experts; ++i) {
    cumsum[i] = cumsum[i - 1] + padded_count[i-1];
  }
}
```

0x3版本 (并行):
- Up-sweep: O(log n) 步
- Down-sweep: O(log n) 步
- 所有线程参与计算

性能提升: 对于 num_experts=128, 从 O(128) 降低到 O(log 128) = O(7)

#### 2. expert_ids 填充优化

0x2版本:
```cpp
// 每个线程负责一个expert,负载不均衡
if (threadIdx.x < num_experts) {
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
    expert_ids[i / block_size] = threadIdx.x;
  }
}
```

0x3版本 (使用二分查找):
```cpp
// 所有线程并行处理所有blocks
const int32_t num_blocks = s_total_tokens_post_pad / block_size;
for (int32_t i = tid; i < num_blocks; i += stride) {
  int32_t block_start = i * block_size;
  // 二分查找找到对应的expert
  int left = 0, right = num_experts;
  while (left < right) {
    int mid = (left + right) >> 1;
    if (prefix[mid] <= block_start) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  expert_ids[i] = left - 1;
}
```

优势:
- 所有线程参与,负载均衡
- 二分查找复杂度 O(log num_experts)
- 适合 expert 分布不均匀的场景

#### 3. 共享内存布局

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
```

scan_size 必须是2的幂次:
```cpp
const size_t scan_size = next_pow2(num_experts);
```

### 性能分析

对比下时间复杂度:
- 前缀和计算: 0x2是O(n)串行,0x3是O(log n)并行
- expert_ids填充: 0x2是O(blocks/experts)不均衡,0x3是O(blocks)均衡

共享内存开销:
```
shared_mem_size = (num_experts + (num_experts + 1) + scan_size) * 4 bytes
```

对于 num_experts=128: (128 + 129 + 128) * 4 = 1540 bytes

---

## 0x4. Block/Warp Scan 算法优化

**PR:** https://github.com/sgl-project/sglang/pull/7884

### 相比 0x3 的核心改进

0x3版本使用 Blelloch Scan 算法实现了并行前缀和,但 Blelloch Scan 需要多次 `__syncthreads()`,同步开销较大。0x4版本引入了**两层 Warp Scan 算法**,利用 warp 内的 shuffle 指令减少同步开销。

### Warp Scan 算法原理

Warp Scan 利用 warp 内线程可以无需同步直接通信的特性(通过 shuffle 指令),实现高效的前缀和计算。

#### Warp-Level Exclusive Scan

```cpp
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = SHFL_UP(mask, v, offset);  // 从前面的线程获取值
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;  // 返回exclusive scan结果
}
```

工作原理:
```
线程ID:  0   1   2   3   4   5   6   7
输入:    3   1   7   0   4   1   6   3

offset=1: 每个线程从前1个线程获取值
         -   3   1   7   0   4   1   6
结果:    3   4   8   7   4   5   7   9

offset=2: 每个线程从前2个线程获取值
         -   -   3   4   8   7   4   5
结果:    3   4  11  11  12  12  11  14

offset=4: 每个线程从前4个线程获取值
         -   -   -   -   3   4  11  11
结果:    3   4  11  11  15  16  22  25

Exclusive: 减去原始值
结果:    0   3   4  11  11  15  16  22
```

优势:
- 无需同步: warp 内线程天然同步
- 低延迟: shuffle 指令延迟很低
- 高效: O(log 32) = 5 次迭代

### 两层 Scan 架构

0x4版本使用**两层扫描**策略:

1. 第一层: 每个 warp 内部进行 scan
2. 第二层: warp0 对所有 warp 的和进行 scan
3. 合并: 每个线程加上前面所有 warp 的总和

```cpp
// 第一层: Intra-warp scan
const int warp_id = tid / WARP_SIZE;
const int lane_id = tid & (WARP_SIZE - 1);
const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;

// 每个warp内部进行inclusive scan
const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;  // 保存warp总和
__syncthreads();

// 第二层: warp0对所有warp的和进行scan
if (tid < WARP_SIZE) {
  int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
  int incl = warp_exclusive_scan(val) + val;  // inclusive scan
  warp_sums[tid] = incl;  // 保存累积和
}
__syncthreads();

// 获取整个block的总和
if (tid == 0) {
  prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
  s_total_tokens_post_pad = prefix[num_experts];
  *total_tokens_post_pad = s_total_tokens_post_pad;
}
__syncthreads();
```

### 完整的前缀和计算流程

```cpp
// 步骤1: 准备scan_buf (与0x3相同)
if (tid < num_experts) {
  int32_t count = shared_counts[tid];
  padded_count = (count + block_size - 1) / block_size * block_size;
  scan_buf[tid] = padded_count;
}
if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
__syncthreads();

// 步骤2: 两层warp scan计算exclusive prefix sum
int v = (tid < scan_size) ? scan_buf[tid] : 0;
int pre = warp_exclusive_scan(v);  // warp内exclusive scan
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;  // 保存warp总和
__syncthreads();

// warp0对所有warp总和进行scan
if (warp_id == 0) {
  int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
  warp_sums[lane_id] = warp_exclusive_scan(val);  // exclusive scan
}
__syncthreads();

// 步骤3: 合并结果
int offset = warp_sums[warp_id];  // 前面所有warp的总和
if (tid < scan_size) scan_buf[tid] = pre + offset;  // 最终的exclusive prefix sum
__syncthreads();

// 步骤4: 写回结果
if (tid < num_experts) prefix[tid] = scan_buf[tid];
if (tid <= num_experts) {
  cumsum[tid] = prefix[tid];
}
```

### 关键优化点对比

#### 1. 前缀和计算

0x3版本 (Blelloch Scan):
```cpp
// Up-sweep: log(n) 次循环,每次都需要 __syncthreads()
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  // ... 计算 ...
  __syncthreads();  // 全局同步
}

// Down-sweep: log(n) 次循环,每次都需要 __syncthreads()
for (int d = 1; d < scan_size; d <<= 1) {
  // ... 计算 ...
  __syncthreads();  // 全局同步
}
```

同步次数: 2 * log(scan_size) 次 `__syncthreads()`

0x4版本 (Warp Scan):
```cpp
// Warp内scan: 无需同步
int pre = warp_exclusive_scan(v);

// 只需3次全局同步
__syncthreads();  // 1. 等待warp_sums写入
// warp0 scan
__syncthreads();  // 2. 等待warp0完成
// 合并结果
__syncthreads();  // 3. 等待写入完成
```

同步次数: 3 次 `__syncthreads()`

性能提升: 对于 scan_size=128, 从 2*log(128)=14 次同步降低到 3 次

#### 2. 共享内存布局

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
int32_t* warp_sums = scan_buf + scan_size;      // [<= 32] - 新增!
```

新增 warp_sums 数组:
- 存储每个 warp 的累积和
- 最多 32 个元素 (1024 threads / 32 = 32 warps)
- 额外开销: 32 * 4 = 128 bytes

#### 3. Shuffle 指令优势

SHFL_UP 指令:
```cpp
#ifndef __CUDA_ARCH__  // HIP
#define SHFL_UP(mask, val, delta) __shfl_up((val), (delta))
#else  // CUDA
#define SHFL_UP(mask, val, delta) __shfl_up_sync((mask), (val), (delta))
#endif
```

特点:
- 延迟低: 通常只需几个时钟周期
- 无内存访问: 直接在寄存器间传输

### 性能分析

时间复杂度对比:
- 前缀和计算: 两个版本都是O(log n)并行
- 同步次数: 0x3需要2*log(scan_size)次,0x4只需要3次
- Shuffle指令: 0x3不用,0x4每个warp用O(log WARP_SIZE)次

实际性能提升:
- 减少同步开销带来~10%性能提升
- 指令级并行更好
- 适合num_experts >= 128的场景

共享内存开销:
```
0x3: (num_experts + (num_experts + 1) + scan_size) * 4 bytes
0x4: (num_experts + (num_experts + 1) + scan_size + 32) * 4 bytes
```

额外开销: 128 bytes (可忽略)

0x3和0x4的主要区别:
- 前缀和算法: 0x3用Blelloch Scan,0x4用两层Warp Scan
- 同步次数: 0x3需要2*log(n)次,0x4只需要3次
- Shuffle指令: 0x3不用,0x4大量使用
- 共享内存: 0x4略多一点(+128B),可以忽略
- 适用场景: 0x3通用,0x4更适合num_experts >= 128
- 性能提升: 0x4相比0x3有~10%提升

---

## 0x5. 进一步优化:动态padding和并行填充

### 优化背景

在0x4版本的基础上,社区又提出了两个方面的优化:

1. **优化小batch场景的max_num_tokens_padded计算**:对于小batch,使用更小的padding值
2. **并行填充sorted_token_ids**:利用额外的线程资源并行填充,而不是串行执行

### 优化1: 动态调整max_num_tokens_padded

**原始逻辑:**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
```

**优化后:**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
# 新增:对于小batch,使用更小的padding
if topk_ids.numel() < num_experts:
    max_num_tokens_padded = topk_ids.numel() * block_size
```

**优化原理:**
- 当token数量很少时(少于expert数量),原始公式会分配过多内存
- 新逻辑保证每个token最多占用一个block,避免内存浪费
- 例如:8个token,256个expert,block_size=128
  - 原始: 8 + 256 * 127 = 32520
  - 优化: 8 * 128 = 1024 (节省96.8%内存)

### 优化2: 并行填充sorted_token_ids

#### 主kernel的优化

**0x4版本(串行填充):**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 开始时串行填充
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }
  
  // 然后进行统计和前缀和计算
  // ...
}
```

**优化版本(使用额外的thread block并行):**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 使用单独的thread block来填充
  if (blockIdx.x == 1) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
      sorted_token_ids[it] = numel;
    }
    return;  // 填充完成后直接返回
  }
  
  // blockIdx.x == 0 的block执行原有逻辑
  // 统计、前缀和、expert_ids填充等
  // ...
}
```

**关键改动:**
- kernel启动从`<<<1, threads>>>`改为`<<<2, threads>>>`
- blockIdx.x == 1专门负责填充sorted_token_ids
- blockIdx.x == 0执行原有的统计和前缀和逻辑
- 两个block完全并行执行,无需同步

**性能提升:**
- 填充操作和统计操作完全并行
- 减少了主计算路径的延迟
- 特别适合大规模场景(max_num_tokens_padded很大时)

#### 小batch kernel的优化

**0x4版本(串行填充):**
```cpp
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  // 所有线程先填充
  for (size_t it = tid; it < max_num_tokens_padded; it += stride) {
    sorted_token_ids[it] = numel;
  }
  
  // 然后统计、前缀和、排序
  // ...
}
```

**优化版本(使用额外的线程组):**
```cpp
template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  // 前fill_threads个线程专门负责填充
  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
      sorted_token_ids[it] = numel;
    }
    // 等待其他线程完成计算(3次同步)
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }
  
  // 其余线程执行原有逻辑
  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;
  // ...
}
```

**关键改动:**
- 模板参数`fill_threads`指定填充线程数(例如256)
- kernel启动从`<<<1, threads>>>`改为`<<<1, fill_threads + threads>>>`
- 前256个线程专门填充,后面的线程做计算
- 需要3次`__syncthreads()`保证填充线程等待计算完成

**为什么需要3次同步:**
```cpp
// 计算线程的同步点:
for (int i = 0; i < num_experts; ++i) {
  tokens_cnts[(tid + 1) * num_experts + i] = 0;
}
for (size_t i = tid; i < numel; i += stride) {
  ++tokens_cnts[(tid + 1) * num_experts + topk_ids[i]];
}
__syncthreads();  // 同步点1

if (tid < num_experts) {
  // 计算前缀和
}
__syncthreads();  // 同步点2

if (tid == 0) {
  // 计算cumsum
}
__syncthreads();  // 同步点3

// 填充线程需要等待这3个同步点
```

### 性能分析

两个优化的性能影响:

1. **动态max_num_tokens_padded**
   - 内存节省:小batch场景下节省90%+内存
   - 性能提升:减少填充开销,约5-10%提升

2. **并行填充**
   - 主kernel:填充和计算完全并行,延迟降低20-30%
   - 小batch kernel:填充和计算部分并行,延迟降低10-15%

综合性能提升:
- 小batch场景:15-30%
- 大batch场景:10-20%

---

## 0x6. 总结

从Baseline到0x5,这个kernel的优化过程其实就是典型的CUDA性能优化路径:

1. Baseline: 功能正确但性能一般,前缀和是串行的
2. 0x2: 加了向量化padding,提升内存带宽利用率
3. 0x3: 用Blelloch Scan并行化前缀和,性能提升明显
4. 0x4: 用Warp Scan减少同步开销,进一步优化性能
5. 0x5: 并行填充+动态内存分配,全方位优化

对于num_experts >= 128的大规模场景,0x4版本的Warp Scan优势明显,能带来~10%的性能提升。0x5版本在此基础上,通过并行填充和动态内存分配,能带来额外10-30%的性能提升。对于小规模场景,用`small_batch_expert_kernel`融合所有操作更合适。
