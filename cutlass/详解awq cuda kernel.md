# 0x0. 前言

上篇博客[详解vLLM和SGLang awq dequantize kernel的魔法](https://mp.weixin.qq.com/s/X9AOH1HGXJ3t0jZ5_hd7Ew)在开头的时候写把int4的awq权重使用`awq_dequantize`反量化回float16的条件写错了，应该是在token<256的时候会直接使用vllm `ops.awq_gemm` 这个算子直接执行awq的矩阵乘法。代码位置见： https://github.com/vllm-project/vllm/blob/b82662d9523d9aa1386d8d1de410426781a1fa3b/vllm/model_executor/layers/quantization/awq.py#L162-L184

```python
def apply(self,
          layer: torch.nn.Module,
          x: torch.Tensor,
          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    qweight = layer.qweight
    scales = layer.scales
    qzeros = layer.qzeros
    pack_factor = self.quant_config.pack_factor
    out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
    reshaped_x = x.reshape(-1, x.shape[-1])

    # num_tokens >= threshold
    FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

    if FP16_MATMUL_HEURISTIC_CONDITION:
        out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        out = torch.matmul(reshaped_x, out)
    else:
        out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                           pack_factor)
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)
```

这篇博客就继续完成`ops.awq_gemm`的代码解析。

# 0x1. kernel启动逻辑

```c++
// in_feats: M, IC [float16] // 输入特征: M行, IC列 [float16类型]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b] // 权重: IC行, OC/8列 [int32类型] -> 转换为IC行, OC列 [uint4b类型]
// scaling_factors: IC // G, OC [float16] // 缩放因子: IC/G行, OC列 [float16类型]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b] // 零点值: IC/G行, OC/8列 [int32类型] -> 转换为IC/G行, OC列 [uint4b类型]
// assume that batch_size < 16 for now // 假设批处理大小当前小于16

/**
 * @brief AWQ量化GEMM计算函数
 * 
 * AWQ (Activation-aware Weight Quantization) 是一种模型量化方法，将FP16权重量化为INT4权重，
 * 同时保持模型性能。该函数实现了AWQ量化后的GEMM (General Matrix Multiplication) 操作。
 * 
 * 计算公式: out_feats = in_feats * (kernel * scaling_factors - zeros)
 * 
 * @param _in_feats 输入特征张量 [M, IC] (float16)
 * @param _kernel 量化后的权重张量 [IC, OC/8] (int32，每个int32包含8个int4值)
 * @param _scaling_factors 缩放因子张量 [IC/G, OC] (float16)，G为分组大小
 * @param _zeros 零点值张量 [IC/G, OC/8] (int32，每个int32包含8个int4零点值)
 * @param split_k_iters 分割K维度的迭代次数，用于并行计算
 * @return 输出特征张量 [M, OC] (float16)
 */
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters) {
  // Get input feature dimensions // 获取输入特征的维度信息
  int num_in_feats = _in_feats.size(0);    // 输入特征的行数M
  int num_in_channels = _in_feats.size(1); // 输入特征的列数IC
  // Set CUDA device guard to ensure correct device execution // 设置CUDA设备保护，确保在正确的设备上执行操作
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

  // Create output tensor options with the same data type and device as input // 创建输出张量的选项，使用与输入相同的数据类型和设备
  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
  // Create output tensor with shape [split_k_iters, M, OC] to store intermediate results // 创建输出张量，形状为[split_k_iters, M, OC]，用于存储分割K维度后的中间结果
  at::Tensor _out_feats =
      torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
  int num_out_feats = _out_feats.size(-2);     // 输出特征的行数M
  int num_out_channels = _out_feats.size(-1);  // 输出特征的列数OC

  // Get raw pointers to input and output tensors // 获取各个张量的原始数据指针并转换为适当的类型
  auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());          // 输入特征指针
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());                    // 量化权重指针
  auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());        // 输出特征指针
  auto scaling_factors =
      reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());               // 缩放因子指针
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());                      // 零点值指针
  // Calculate group size, which is the number of input channels per group // 计算分组大小，每组包含的输入通道数
  int group_size = num_in_channels / _scaling_factors.size(0);

  // Check parameter validity, ensuring dimensions meet requirements // 检查参数有效性，确保维度满足要求
  if (num_out_channels % 64 != 0)
    throw std::invalid_argument("OC is not multiple of cta_N = 64");  // 输出通道数必须是64的倍数
  if (num_out_channels % 8 != 0)
    throw std::invalid_argument("OC is not multiple of pack_num = 8"); // 输出通道数必须是8的倍数（因为每个int32存储8个int4值）
  if (group_size % 32 != 0)
    throw std::invalid_argument("Group size should be a multiple of 32"); // 分组大小必须是32的倍数
  if (num_out_channels % group_size != 0)
    throw std::invalid_argument("OC is not multiple of Group size"); // 输出通道数必须是分组大小的倍数

  // Get current CUDA stream // 获取当前CUDA流
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // Choose kernel configuration based on output channel count // 根据输出通道数选择不同的kernel配置
  if (num_out_channels % 128 == 0) {
    // When output channel count is a multiple of 128, use a more efficient kernel // 当输出通道数是128的倍数时，使用更高效的kernel
    int j_factors1 = num_out_channels / 128 / 1;  // 计算输出通道维度的分块因子
    // Calculate number of CUDA blocks to launch // 计算需要启动的CUDA块数量
    dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
    // Set threads per block // 设置每个块的线程数
    // threadIdx.x: 32 - 每个warp有32个线程
    // threadIdx.y: 这里是2，表示每个块有2个warp
    dim3 threads_per_block(32, 2);
    // Launch kernel with 128 output channels per block // 调用模板函数执行GEMM计算，模板参数128表示每个线程块处理128个输出通道
    vllm::awq::gemm_forward_4bit_cuda_m16nXk32<128>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  } else if (num_out_channels % 64 == 0) {
    // When output channel count is a multiple of 64, use another kernel configuration // 当输出通道数是64的倍数时，使用另一个kernel配置
    int j_factors1 = num_out_channels / 64 / 1;  // 计算输出通道维度的分块因子
    // Calculate number of CUDA blocks to launch // 计算需要启动的CUDA块数量
    dim3 num_blocks(1 * (num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);

    // Set threads per block // 设置每个块的线程数
    // threadIdx.x: 32 - 每个warp有32个线程
    // threadIdx.y:  这里是2，表示每个块有2个warp
    dim3 threads_per_block(32, 2);
    // Launch kernel with 64 output channels per block // 调用模板函数执行GEMM计算，模板参数64表示每个线程块处理64个输出通道
    vllm::awq::gemm_forward_4bit_cuda_m16nXk32<64>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros,
            num_in_feats, num_in_channels, num_out_channels, out_feats);
  }
  // Sum split_k_iters intermediate results along dimension 0 to get final output // 将split_k_iters个中间结果沿着第0维相加，得到最终输出
  return _out_feats.sum(0);
}
```

注意到输入的Tensor shape是`[M, IC]`，输入的int4量化权重为`[IC, OC // 8]`，然后int4量化权重被反量化为shape为 `[IC, OC]`且 `dtype=fp16`的Tensor。在kernel dispatch处有2个逻辑，根据`num_out_channels`是否整除128和64 dispatch到`gemm_forward_4bit_cuda_m16nXk32` kernel，`gemm_forward_4bit_cuda_m16nXk32` kernel的模板参数就是128或者64也就是每个线程块处理的输出通道数。把输出看成一个`[M, OC]` shape的矩阵Tensor，我们在行方向上以16大小作为分块大小，在列方向上使用64或者128作为分块大小，所以一个block就要负责`[16x128]`或者`[16x64]`大小的输出Tensor区域元素的计算。另外为了提升并行度，我们在矩阵的K纬度，也就是`IC`纬度引入一个splitk优化，所以最终的block大小是这三者的乘积。对于每个Block使用固定的64个线程，并且使用`dim3 threads_per_block(32, 2)`的二维线程网格。

# 0x2. `gemm_forward_4bit_cuda_m16nXk32` kernel实现细节

魔鬼在细节中，这里读细一点。

## 0x2.1 定位每个线程处理的A和B的global memory位置

```c++
/**
 * @brief AWQ量化GEMM计算的CUDA kernel函数
 * 
 * 该函数实现了AWQ量化后的GEMM (General Matrix Multiplication) 操作的核心计算。
 * 使用Tensor Core加速矩阵乘法，支持INT4量化权重的高效计算。
 * 
 * 计算流程：
 * 1. 从全局内存加载输入数据到共享内存
 * 2. 对量化权重进行反量化(dequantize)：权重值 = 量化值 * 缩放因子 - 零点值 * 缩放因子
 * 3. 使用Tensor Core执行矩阵乘法
 * 4. 将结果写回全局内存
 * 
 * @tparam N 每个线程块处理的输出通道数，支持64或128
 * @param G 分组大小，用于分组量化
 * @param split_k_iters 分割K维度的迭代次数，用于并行计算
 * @param A 输入特征指针 [M, IC] (float16)
 * @param B 量化权重指针 [IC, OC/8] (int32，每个int32包含8个int4值)
 * @param scaling_factors 缩放因子指针 [IC/G, OC] (float16)
 * @param zeros 零点值指针 [IC/G, OC/8] (int32，每个int32包含8个int4零点值)
 * @param M 输入特征的行数
 * @param IC 输入特征的列数
 * @param OC 输出特征的列数
 * @param C 输出特征指针 [M, OC] (float16)
 */
template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
```

这里是函数入口没什么好说的，看上面的注释即可。

```c++
// Only support matrix n = 64 or 128 // 仅支持矩阵 n = 64 或 128
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);  // Require CUDA architecture >= 750 (Turing+) // 要求CUDA架构 >= 750 (Turing及以上)
#else
  static constexpr uint32_t ZERO = 0x0;  // Zero constant for FMA operations // 用于FMA操作的零常量
  float C_warp[32];  // Register array to store computation results // 存储计算结果的寄存器数组
  __shared__ half A_shared[16 * (32 + 8)];  // A matrix in shared memory, with padding to avoid bank conflicts // 共享内存中的A矩阵，包含填充以避免bank冲突
  __shared__ half B_shared[32 * (N + 8)];   // B matrix in shared memory, with padding to avoid bank conflicts // 共享内存中的B矩阵，包含填充以避免bank冲突
```

- 这里的N代表的是输出矩阵的Tile的列方向大小，然后A_shared 的大小 `[16 * (32 + 8)]`，这个大小表示每个线程块处理输入/输出矩阵的16行数据，然后每次迭代处理32个输入通道（也就是K/IC）纬度。额外的8列填充是为了避免共享内存的bank冲突。

- B_shared的大小` [32 * (N + 8)]`，32对应每次迭代处理的32个输入通道（K/IC维度），N输出矩阵的Tile的列方向大小，+8的填充同样是为了避免共享内存的bank冲突。

- C_warp是存储计算结果的寄存器数组。

```c++
// Calculate output channel block factor // 计算输出通道的分块因子
  int j_factors1 = ((OC + N - 1) / N);
  // Calculate block index, blockIdx_y represents the position of the output block // 计算块索引，blockIdx_y表示处理的输出块位置
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  // Calculate block index in split_k dimension // 计算在split_k维度上的块索引
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  // Data loaded by each warp for A and B matrices // 每个warp加载的A和B矩阵数据
  half A_shared_warp[8];
  half B_shared_warp[N / 4];
```

OC是输出通道数，N是模板参数（64或128），表示每个线程块处理的输出通道数，`(OC + N - 1) / N`是向上取整的整数除法，表示需要多少块来覆盖所有输出通道。这个分块因子用于后续的线程块索引计算，确保所有输出通道都被处理。

`int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);` 这行代码将一维的blockIdx.x映射到二维空间的y坐标，`(M + 16 - 1) / 16`计算输入矩阵M行需要多少个16行的块，乘以j_factors1得到总的块数，对blockIdx.x取模，得到在y维度上的位置。`int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);` 这行代码计算在split_k维度上的索引。

```c++
half A_shared_warp[8];
half B_shared_warp[N / 4];
```

`half A_shared_warp[8]` 表示每个线程用8个half类型的寄存器存储A矩阵的数据，这些数据将从共享内存加载，用于Tensor Core计算，8个元素对应一个线程在矩阵乘法中处理的一部分输入数据。`B_shared_warp[N / 4]`，当N=64时，每个线程有16个寄存器存储B矩阵数据，当N=128时，每个线程有32个寄存器存储B矩阵数据，这些寄存器数组的大小设计考虑了Tensor Core指令的限制。在后续的代码中，这些寄存器将被用于存储从共享内存加载的数据，然后传递给Tensor Core指令进行矩阵乘法计算。

```c++
static constexpr int row_stride_warp = 32 * 8 / 32;  // 每个warp内的行步长
static constexpr int row_stride = 2 * 32 * 8 / N;    // 整个线程块的行步长
```

`row_stride_warp`表示每个warp在处理矩阵行时的步长为8，从上面的warp 共享内存定义也可以看出来。然后`row_stride`表示线程块在处理矩阵时的stride，当N=64时，row_stride=8；当N=128时，row_stride=4。

```c++
// Check if A matrix data needs to be loaded (boundary check) // 检查是否需要加载A矩阵数据（边界检查）
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;  // threadIdx.y is warp_id // threadIdx.y是warp_id
```

- 这行代码检查当前线程是否需要加载A矩阵的数据（是否在有效边界内）。其中`blockIdx_y / j_factors1 * 16`确定当前线程块处理的输入矩阵起始行。

```shell
int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
blockIdx_y / j_factors1 * 16 = blockIdx.x
```

- `threadIdx.y * row_stride_warp` 表示warp内的行偏移，
- `threadIdx.x * 8 / 32`：线程在warp内的行偏移（每32个线程共享8行，即每4个线程处理1行）
- 连到一起就是计算出当前线程要处理的行索引，如果小于M（输入矩阵的总行数），则需要加载数据


```c++
half* A_ptr =
    A +
    (((int)blockIdx_y) / j_factors1 * 16 +
     (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
        IC +
    (((int)threadIdx.x) % (32 / 8)) * 8;
```

同样这行代码计算每个线程访问A矩阵的起始位置，具体来说：

- 矩阵行偏移部分
  - `blockIdx_y / j_factors1 * 16`：线程块的起始行
  - `threadIdx.y * row_stride_warp`：warp内的行偏移
  - `threadIdx.x / (32/8)`：线程在warp内的行偏移（等价于上面的`threadIdx.x * 8 / 32`）
  - 这三项相加得到行索引，乘以IC（输入通道数）得到行的起始位置
- 矩阵列偏移部分
  - `threadIdx.x % (32/8) * 8`：线程在行内的列偏移（等价于`threadIdx.x % 4 * 8`）, %4是因为每4个线程处理一行
  - 每个线程负责加载8个连续的half元素


画个图，假设忽略splitk参数（有splitk参数不好画，但是它比较好理解，这里就不画了）：

```shell
矩阵A [M, IC]:
+------------------------------------------------------------------+
|                                                                  |
|  线程块(0,0) (16行×IC列)          线程块(0,1) (16行×IC列)          |
|  +----------------------------+  +----------------------------+  |
|  | Warp0 (8行)               |  | Warp0 (8行)               |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行0, 列0-7      ||  | | 线程0: 行0, 列0-7      ||  |
|  | | 线程1: 行0, 列8-15     ||  | | 线程1: 行0, 列8-15     ||  |
|  | | 线程2: 行0, 列16-23    ||  | | 线程2: 行0, 列16-23    ||  |
|  | | 线程3: 行0, 列24-31    ||  | | 线程3: 行0, 列24-31    ||  |
|  | | 线程4: 行1, 列0-7      ||  | | 线程4: 行1, 列0-7      ||  |
|  | | 线程5: 行1, 列8-15     ||  | | 线程5: 行1, 列8-15     ||  |
|  | | 线程6: 行1, 列16-23    ||  | | 线程6: 行1, 列16-23    ||  |
|  | | 线程7: 行1, 列24-31    ||  | | 线程7: 行1, 列24-31    ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程28: 行7, 列0-7     ||  | | 线程28: 行7, 列0-7     ||  |
|  | | 线程29: 行7, 列8-15    ||  | | 线程29: 行7, 列8-15    ||  |
|  | | 线程30: 行7, 列16-23   ||  | | 线程30: 行7, 列16-23   ||  |
|  | | 线程31: 行7, 列24-31   ||  | | 线程31: 行7, 列24-31   ||  |
|  | +------------------------+|  | +------------------------+|  |
|  |                           |  |                           |  |
|  | Warp1 (8行)               |  | Warp1 (8行)               |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行8, 列0-7      ||  | | 线程0: 行8, 列0-7      ||  |
|  | | 线程1: 行8, 列8-15     ||  | | 线程1: 行8, 列8-15     ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程31: 行15, 列24-31  ||  | | 线程31: 行15, 列24-31  ||  |
|  | +------------------------+|  | +------------------------+|  |
|  +----------------------------+  +----------------------------+  |
|                                                                  |
|  线程块(1,0) (16行×IC列)          线程块(1,1) (16行×IC列)          |
|  +----------------------------+  +----------------------------+  |
|  | ...                       |  | ...                       |  |
|  +----------------------------+  +----------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

接下来看看B矩阵：

```c++
// Calculate B matrix (weight) pointer in global memory // 计算B矩阵(权重)在全局内存中的指针位置
  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;
```

B矩阵（量化权重）的指针计算更为复杂，因为它是INT4格式，每个int32包含8个INT4值，可以阅读[详解vLLM和SGLang awq dequantize kernel的魔法](https://mp.weixin.qq.com/s/X9AOH1HGXJ3t0jZ5_hd7Ew)作为铺垫。下面分析下下标的具体细节：
- `threadIdx.y * (OC / 8) * (256 / N)`：确定warp在输入通道方向上的偏移。
- `threadIdx.x / (N / 8) * (OC / 8)`：确定线程在输入通道方向上的偏移。
- `blockIdx_y % j_factors1 * (N / 8)`：确定线程块在输出通道方向上的偏移。
- `threadIdx.x % (N / 8) * 1`：确定线程在输出通道方向上的偏移。

假设N=128，画一下B矩阵的block，warp，线程布局：

```shell
矩阵B [IC, OC/8] (量化权重):
+------------------------------------------------------------------+
|                                                                  |
|  线程块(0,0) (IC行×16列)          线程块(0,1) (IC行×16列)          |
|  +----------------------------+  +----------------------------+  |
|  | Warp0 (行0-255)           |  | Warp0 (行0-255)           |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行0, 列0        ||  | | 线程0: 行0, 列16       ||  |
|  | | 线程1: 行0, 列1        ||  | | 线程1: 行0, 列17       ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程15: 行0, 列15      ||  | | 线程15: 行0, 列31      ||  |
|  | | 线程16: 行128, 列0     ||  | | 线程16: 行128, 列16    ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程31: 行128, 列15    ||  | | 线程31: 行128, 列31    ||  |
|  | +------------------------+|  | +------------------------+|  |
|  |                           |  |                           |  |
|  | Warp1 (行256-511)         |  | Warp1 (行256-511)         |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行256, 列0      ||  | | 线程0: 行256, 列16     ||  |
|  | | 线程1: 行256, 列1      ||  | | 线程1: 行256, 列17     ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程31: 行384, 列15    ||  | | 线程31: 行384, 列31    ||  |
|  | +------------------------+|  | +------------------------+|  |
|  +----------------------------+  +----------------------------+  |
```

再结合B的下标映射看一下：

- 输入通道（行）方向的分配：
    - `threadIdx.y * (OC / 8) * (256 / N)`：Warp0处理行0-255，Warp1处理行256-511
    - `(threadIdx.x / (N / 8)) * (OC / 8)`：当N=128时，N/8=16，所以每16个线程处理不同的行
- 输出通道（列）方向的分配：
    - `(blockIdx_y % j_factors1) * (N / 8)`：Block(0,0)处理列0-15，Block(0,1)处理列16-31
    - (((int)threadIdx.x) % (N / 8)) * 1：线程在列方向的偏移

小结一下，A矩阵大小为[M, IC]，每个Block处理16行输入，每个Warp处理8行，每个线程负责加载8个连续的half元素。B矩阵大小为 [IC, OC/8]，每个Block处理N个输出通道(N=64或128)，每个线程负责加载N/4个int32元素，如果N=128就是每个线程加载32个int4元素，也就是加载原始的B矩阵的4个元素再做反量化。

还需要在这里指出的一点是，示意图里面只画了行0、行128、行256和行384的处理情况，但没有明确说明其他行（如行1、2、3等）是如何处理的。这里先解释一下，从上面的B矩阵的index计算流程，我们可以分析出B矩阵行的处理方式：

- Warp级别的行分配
    - threadIdx.y * (OC / 8) * (256 / N)：当N=128时，这部分为threadIdx.y * (OC / 8) * 2
        - Warp0 (threadIdx.y=0) 处理行0到行255
        - Warp1 (threadIdx.y=1) 处理行256到行511
- 线程级别的行分配
    - (threadIdx.x / (N / 8)) * (OC / 8)：当N=128时，N/8=16
        - 线程0-15 (threadIdx.x / 16 = 0) 处理基础行（行0或行256）
        - 线程16-31 (threadIdx.x / 16 = 1) 处理基础行+128（行128或行384）

实际上，其他行（如行1、2、3等）是通过K维度的迭代来处理的。在kernel的主循环中：

```c++
for (int k_0 = 0; k_0 < IC; k_0 += 32) {
    // 加载数据到共享内存
    // ...
    
    // 计算矩阵乘法
    // ...
}
```

kernel的实际代码是

```c++
// Main loop: iterate over K dimension blocks // 主循环：遍历K维度的分块
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
```

## B矩阵的布局和加载详细解释

A的布局和加载比较好理解，B矩阵的布局和加载更困难一些，这里再详细读一下。

在main loop中，B矩阵的处理方式如下：

```c++
// 获取当前处理的B矩阵(权重)指针
int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

// 遍历并加载B矩阵数据到共享内存，同时进行反量化操作
for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
  // 从全局内存加载量化的权重值(INT4)
  uint32_t B_loaded =
      *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
  // ...反量化操作...
}
```

然后B矩阵指针的计算方式：

```c++
int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
             (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
             (((int)blockIdx_y) % j_factors1) * (N / 8) +
             (((int)threadIdx.x) % (N / 8)) * 1;
```

假设 N=128 ,这里来算一下B矩阵的布局。先把上一节画出来的布局图再贴一下：


```shell
矩阵B [IC, OC/8] (量化权重):
+------------------------------------------------------------------+
|                                                                  |
|  线程块(0,0) (IC行×16列)          线程块(0,1) (IC行×16列)          |
|  +----------------------------+  +----------------------------+  |
|  | Warp0 (行0-255)           |  | Warp0 (行0-255)           |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行0, 列0        ||  | | 线程0: 行0, 列16       ||  |
|  | | 线程1: 行0, 列1        ||  | | 线程1: 行0, 列17       ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程15: 行0, 列15      ||  | | 线程15: 行0, 列31      ||  |
|  | | 线程16: 行128, 列0     ||  | | 线程16: 行128, 列16    ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程31: 行128, 列15    ||  | | 线程31: 行128, 列31    ||  |
|  | +------------------------+|  | +------------------------+|  |
|  |                           |  |                           |  |
|  | Warp1 (行256-511)         |  | Warp1 (行256-511)         |  |
|  | +------------------------+|  | +------------------------+|  |
|  | | 线程0: 行256, 列0      ||  | | 线程0: 行256, 列16     ||  |
|  | | 线程1: 行256, 列1      ||  | | 线程1: 行256, 列17     ||  |
|  | | ...                    ||  | | ...                    ||  |
|  | | 线程31: 行384, 列15    ||  | | 线程31: 行384, 列31    ||  |
|  | +------------------------+|  | +------------------------+|  |
|  +----------------------------+  +----------------------------+  |
```

- Block级别的划分:
    - 每个Block处理N=128个输出通道
    - `blockIdx_y % j_factors1 * (N / 8)`确定了Block在输出通道方向上的起始位置
    - 例如：Block(0,0)处理列0-15，Block(0,1)处理列16-31
- Warp级别的划分:
    - `threadIdx.y * (OC / 8) * (256 / N)`表示Warp在输入通道方向的偏移
    - 当N=128时，每个Warp处理256/128=2行的输入通道
    - Warp0处理行0-255，Warp1处理行256-511（假设每个Warp有32个线程）
- 线程级别的划分:
    - `threadIdx.x / (N / 8)`：当N=128时，N/8=16，表示每16个线程处理不同的输入通道行
    - `threadIdx.x % (N / 8)`：线程在输出通道方向上的偏移，每个线程处理一个输出通道
- 数据加载:
    - 在main loop中，`B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8)`计算当前迭代的B矩阵指针
    - `ax0_ax1_fused_0`循环遍历`N/16`次，加载B矩阵的数据
    - row_stride定义为`2 * 32 * 8 / N`，当N=128时，`row_stride=4`

然后这个数据和上面的布局图是对应的，当N=128时，每个线程实际上处理的是量化后的权重，每个int32包含8个INT4值。所以每个线程实际上负责处理8个输出通道的量化值，然后每个线程在每次迭代中会加载多个int32值，这些值会被反量化成多个half值。

## main loop的加载和计算逻辑

我们注意到每个线程分别加载的A和B数据为：

```c++
half A_shared_warp[8];
half B_shared_warp[N / 4];
```

为什么warp中每个线程的A和B共享内存大小分别是8和N/4？

首先来看A_shared_warp 的大小 [8]，Tensor Core 在 CUDA 中执行 8x8x8 的矩阵乘法操作（对于 half 精度数据），每个 warp (32线程) 协作处理一个 8x8 的矩阵块，然后`ldmatrix.sync.aligned.m8n8.x4.shared.b16`指令从共享内存加载一个 8x8 的矩阵到寄存器中，其中`.x4`表示加载 4 个 16 位元素，总共 8 个 half 元素。8 个 half 元素正好对应 A_shared_warp[8] 的大小。

```c++
// Use ldmatrix instruction to load A matrix data from shared memory to registers // 使用ldmatrix指令从共享内存加载A矩阵数据到寄存器
__asm__ __volatile__(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
        "=r"(((unsigned*)(A_shared_warp + 0))[1]),
        "=r"(((unsigned*)(A_shared_warp + 0))[2]),
        "=r"(((unsigned*)(A_shared_warp + 0))[3])
    : "r"(addr));
```

