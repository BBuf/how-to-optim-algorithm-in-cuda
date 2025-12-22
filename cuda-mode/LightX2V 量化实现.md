# 0x0. 前言

最近在研究 LightX2V 项目时,发现它的 lightx2v_kernel 是一个专门为 Blackwell 架构(SM120)优化的低精度量化 GEMM 库。这个库支持 NVFP4(NVIDIA 自定义的 4-bit 浮点格式 E2M1)和 MXFP4/6/8(基于 MX-Formats 标准)等多种低精度格式,底层基于 CUTLASS 3.x 实现,充分利用了 Blackwell 的 Block Scaled Tensor Core 和 PTX 指令。

本文会从接口使用、量化原理、到底层 CUDA 实现进行详细解读,特别是量化和 GEMM kernel 的线程模型部分。代码实现部分会添加详细的注释和讲解,帮助理解这个高性能 kernel 是如何工作的。

项目地址: https://github.com/LightX2V/LightX2V

# 0x1. 接口使用

## 0x1.1 Python 接口

LightX2V kernel 提供了简洁的 Python 接口，主要就是量化函数和矩阵乘法函数。

### NVFP4 接口

```python
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

# 量化函数
def scaled_nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    将输入张量量化为 FP4 格式
    
    Args:
        input: 输入张量，shape 为 (m, n)，dtype 为 fp16/bf16
        input_global_scale: 全局缩放因子，标量张量
    
    Returns:
        output: 量化后的张量，shape 为 (m, n//2)，dtype 为 uint8（两个 fp4 打包）
        output_scale: 量化因子，shape 为 (rounded_m, rounded_k)，dtype 为 float8_e4m3fn
                     其中 rounded_m = ((m + 128 - 1) // 128) * 128
                          rounded_k = (n // 16 + 4 - 1) // 4
    """

# 矩阵乘法函数
def cutlass_scaled_nvfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """
    执行 FP4 矩阵乘法：D = alpha * A @ B^T + bias
    
    Args:
        mat_a: 矩阵 A，shape 为 (m, k//2)，已量化
        mat_b: 矩阵 B，shape 为 (n, k//2)，已量化
        scales_a: A 的量化因子
        scales_b: B 的量化因子
        alpha: 缩放因子
        bias: 可选的偏置项，shape 为 (1, n)
    
    Returns:
        out: 输出张量，shape 为 (m, n)，dtype 为 bfloat16
    """
```

### MXFP4 接口

```python
from lightx2v_kernel.gemm import scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm

# 量化函数
def scaled_mxfp4_quant(input: torch.Tensor):
    """
    将输入张量量化为 MXFP4 格式
    
    Args:
        input: 输入张量，shape 为 (m, n)，dtype 为 fp16/bf16
    
    Returns:
        output: 量化后的张量，shape 为 (m, n//2)，dtype 为 uint8
        output_scale: 量化因子，dtype 为 float8_e8m0fnu
    """

# 矩阵乘法函数（接口与 NVFP4 类似）
def cutlass_scaled_mxfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """MXFP4 矩阵乘法"""
```

## 0x1.2 `scaled_nvfp4_quant` 函数解析

这是 NVFP4 量化的核心 Python 接口，下面逐行看看它是怎么实现的：

```python
def scaled_nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """
```

### 函数签名和参数

先看参数：
- `input`：待量化的输入张量
  - 形状：`(m, n)`，其中 m 是行数，n 是列数
  - 数据类型：`torch.float16` 或 `torch.bfloat16`
  - 约束：n 必须是 16 的倍数（因为每 16 个元素共享一个 scale factor）

- `input_global_scale`：全局缩放因子
  - 形状：标量张量 `(1,)` 或 `()`
  - 数据类型：`torch.float32`
  - 作用：用于将 per-group 的 scale factor 缩放到 FP8(E4M3) 的范围
  - 计算公式：`global_scale = 2688.0 / max(abs(input))`，其中 2688.0 = 6.0 × 448.0

返回值：
- `output`：量化后的数据
  - 形状：`(m, n//2)`
  - 数据类型：`torch.uint8`
  - 原因：两个 FP4 值（每个 4 bits）打包成一个 uint8（8 bits）

- `output_scale`：量化因子
  - 形状：`(rounded_m, rounded_k)`
  - 数据类型：`torch.float8_e4m3fn`
  - 布局：swizzled layout（优化 Tensor Core 访问）

### 获取输入维度

```python
    m, n = input.shape
    block_size = 16
    device = input.device
```

**代码解析**：
- `m, n = input.shape`：解包输入张量的形状
  - `m`：行数（batch size 或 sequence length）
  - `n`：列数（hidden dimension 或 feature dimension）
  
- `block_size = 16`：定义量化粒度
  - NVFP4 使用 per-group 量化，每 16 个元素共享一个 scale factor
  - 这是硬件优化的结果，与 Tensor Core 的数据处理单元对齐
  
- `device = input.device`：获取输入张量所在的设备
  - 确保输出张量在同一设备上创建
  - 避免不必要的设备间数据传输

### 注释掉的断言检查

```python
    # assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    # assert input.dtype in (
    #     torch.float16,
    #     torch.bfloat16,
    # ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
```

为什么注释掉？
- 这些检查在 CUDA kernel 内部已经有了
- Python 层面的检查会增加额外的开销
- 在生产环境中，通常假设输入已经经过验证

实际使用建议：
- 在开发和调试阶段，建议取消注释这些断言
- 在生产环境中，可以保持注释以提高性能

### 创建量化数据输出张量

```python
    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
```

**代码解析**：
- `torch.empty`：创建未初始化的张量（比 `torch.zeros` 更快）
  - 数据会被 CUDA kernel 完全覆盖，无需初始化
  
- `(m, n // 2)`：输出形状
  - 行数保持不变：`m`
  - 列数减半：`n // 2`
  - 原因：每两个 FP4 值（4 bits × 2 = 8 bits）打包成一个 uint8
  
- `dtype=torch.uint8`：使用无符号 8 位整数
  - 每个 uint8 存储两个 FP4 值
  - 低 4 位存储第一个 FP4 值
  - 高 4 位存储第二个 FP4 值

内存布局：
```
原始数据 (FP16): [v0, v1, v2, v3, v4, v5, ...]  # n 个元素
量化数据 (uint8): [v0|v1, v2|v3, v4|v5, ...]    # n/2 个元素
                   └─┬─┘  └─┬─┘  └─┬─┘
                   1 byte 1 byte 1 byte
```

### 创建量化因子输出张量

```python
    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    # rounded_m = ((m + 128 - 1) // 128) * 128
    # scale_n = n // block_size
    # rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_scale = torch.zeros(
        (((m + 128 - 1) // 128) * 128, (n // block_size + 4 - 1) // 4), 
        device=device, 
        dtype=torch.int32
    )
```

为什么用 `torch.zeros` 而不是 `torch.empty`？
- Swizzled layout 可能需要 padding
- Padding 区域需要初始化为 0，避免未定义行为

第一维度 `((m + 128 - 1) // 128) * 128`：
- 目的：将 m 向上舍入到 128 的倍数
- 原因：Tensor Core 的 M 维度 tile 大小是 128
- 示例：
  - `m = 100` → `rounded_m = 128`
  - `m = 200` → `rounded_m = 256`
  - `m = 1000` → `rounded_m = 1024`

第二维度 `(n // block_size + 4 - 1) // 4`：
- `n // block_size`：scale factor 的数量
  - 每 16 个元素一个 scale factor
  - 例如：`n = 4096` → `4096 // 16 = 256` 个 scale factors
  
- `(... + 4 - 1) // 4`：向上舍入到 4 的倍数
  - 原因：4 个 FP8(E4M3) 值（每个 8 bits）打包成一个 int32（32 bits）
  - 这是 swizzled layout 的要求
  
- 示例：
  - `n = 4096` → `scale_n = 256` → `rounded_k = 64`
  - `n = 5120` → `scale_n = 320` → `rounded_k = 80`

为什么用 `dtype=torch.int32`？
- 4 个 FP8 值打包成 1 个 int32
- 这是 CUDA kernel 的输出格式
- 后续会通过 `view` 操作转换为 `float8_e4m3fn`

Swizzled Layout 的作用：
```
常规布局：
[SF_0, SF_1, SF_2, SF_3, SF_4, SF_5, ...]  # 线性排列

Swizzled 布局：
[SF_0, SF_32, SF_64, SF_96,   # 第一组
 SF_1, SF_33, SF_65, SF_97,   # 第二组
 ...]                          # 优化 Tensor Core 访问模式
```

### 调用 CUDA Kernel

```python
    torch.ops.lightx2v_kernel.scaled_nvfp4_quant_sm120.default(
        output, input, output_scale, input_global_scale
    )
```

调用路径：
- `torch.ops`：PyTorch 的自定义算子命名空间
- `lightx2v_kernel`：库名称（在 `common_extension.cc` 中注册）
- `scaled_nvfp4_quant_sm120`：算子名称
- `.default`：默认的调度版本（对应 CUDA 实现）

参数传递：
1. `output`：输出张量（in-place 修改）
   - 形状：`(m, n//2)`
   - 类型：`torch.uint8`
   
2. `input`：输入张量（只读）
   - 形状：`(m, n)`
   - 类型：`torch.float16` 或 `torch.bfloat16`
   
3. `output_scale`：量化因子输出（in-place 修改）
   - 形状：`(rounded_m, rounded_k)`
   - 类型：`torch.int32`
   
4. `input_global_scale`：全局缩放因子（只读）
   - 形状：标量
   - 类型：`torch.float32`

CUDA Kernel 内部做了什么：
1. 每个线程处理 8 个元素
2. 每 2 个线程（16 个元素）计算一个 scale factor
3. Scale factor 量化到 FP8(E4M3)
4. 数据缩放并量化到 FP4(E2M1)
5. 两个 FP4 值打包成一个 uint8
6. Scale factors 写入 swizzled layout

### 转换量化因子数据类型

```python
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale
```

`view(torch.float8_e4m3fn)` 的作用：
- 将 `int32` 张量重新解释为 `float8_e4m3fn` 张量
- **不进行数据拷贝**，只改变类型解释
- 形状保持不变：`(rounded_m, rounded_k)`

为什么先用 int32 再转换？
1. CUDA kernel 输出时，4 个 FP8 值打包成 1 个 int32
2. PyTorch 的 `float8_e4m3fn` 类型在 Python 层面更易用
3. `view` 操作是零开销的，只改变元数据

内存布局：
```
int32 视图:
[0xAABBCCDD, 0xEEFF0011, ...]
 └─┬──┬──┬──┬┘
   4个FP8值

float8_e4m3fn 视图:
[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, ...]
 └─┬┘  └─┬┘  └─┬┘  └─┬┘  └─┬┘  └─┬┘  └─┬┘  └─┬┘
  SF0   SF1   SF2   SF3   SF4   SF5   SF6   SF7
```

返回值：
- `output`：量化后的数据，`(m, n//2)` 的 uint8 张量
- `output_scale`：量化因子，`(rounded_m, rounded_k)` 的 float8_e4m3fn 张量

### 完整数据流示例

假设输入：
- `input.shape = (1024, 4096)`
- `input.dtype = torch.bfloat16`
- `input_global_scale = 2688.0 / max(abs(input))`

处理过程：
1. **输入维度**：`m=1024, n=4096`
2. **量化数据**：`output.shape = (1024, 2048)`，`dtype=uint8`
3. **Scale factor 数量**：`4096 // 16 = 256` 个
4. **Rounded 维度**：
   - `rounded_m = 1024`（已经是 128 的倍数）
   - `rounded_k = 256 // 4 = 64`
5. **量化因子**：`output_scale.shape = (1024, 64)`，`dtype=float8_e4m3fn`

内存占用对比：
- **原始数据**：`1024 × 4096 × 2 bytes = 8 MB`（bfloat16）
- **量化数据**：`1024 × 2048 × 1 byte = 2 MB`（uint8）
- **量化因子**：`1024 × 64 × 1 byte = 64 KB`（float8_e4m3fn）
- **总计**：`2.06 MB`（压缩比约 3.88×）

---

## 0x1.3 使用示例

下面是一个完整的使用例子，看看如何用 NVFP4 进行权重量化和推理：

```python
import torch
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

class MMWeightFp4:
    """使用 FP4 量化的矩阵乘法封装类"""
    
    def __init__(self, weight, bias):
        # 加载并量化权重
        self.load_fp4_weight(weight, bias)
        # 校准激活值的最大值
        self.calibrate_x_absmax()

    @torch.no_grad()
    def apply(self, input_tensor):
        """执行量化矩阵乘法"""
        # 量化输入
        input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
            input_tensor, self.input_global_scale
        )
        # 执行矩阵乘法
        output_tensor = cutlass_scaled_nvfp4_mm(
            input_tensor_quant, 
            self.weight, 
            input_tensor_scale, 
            self.weight_scale, 
            alpha=self.alpha, 
            bias=self.bias
        )
        return output_tensor

    @torch.no_grad()
    def load_fp4_weight(self, weight, bias):
        """量化权重"""
        # 计算权重的全局缩放因子
        # 2688.0 = 6.0 * 448.0，其中 6.0 是 FP4 最大值，448.0 是 FP8(E4M3) 最大值
        self.weight_global_scale = (
            2688.0 / torch.max(torch.abs(weight))
        ).to(torch.float32)
        
        # 量化权重
        self.weight, self.weight_scale = scaled_nvfp4_quant(
            weight, self.weight_global_scale
        )
        self.bias = bias

    def calibrate_x_absmax(self):
        """校准输入激活值的最大值"""
        # 这个值需要通过校准数据集来确定
        self.x_absmax = torch.tensor(
            5.0, dtype=torch.float32, device=self.weight.device
        )
        # 计算输入的全局缩放因子
        self.input_global_scale = (
            2688.0 / self.x_absmax
        ).to(torch.float32)
        # 计算最终的 alpha 值
        self.alpha = 1.0 / (
            self.input_global_scale * self.weight_global_scale
        )

# 使用示例
m, k, n = 1024, 4096, 4096
input_tensor = torch.randn(m, k, dtype=torch.bfloat16).cuda()
weight = torch.randn(n, k, dtype=torch.bfloat16).cuda()
bias = None

# 创建量化矩阵乘法对象
mm = MMWeightFp4(weight, bias)

# 执行推理
output = mm.apply(input_tensor)
print(f"Output shape: {output.shape}")  # (1024, 4096)
```

---

# 0x2. 量化原理

## 0x2.1 NVFP4 量化原理

#### 3.1.1 数据格式

NVFP4 使用 E2M1 格式（1 位符号位 + 2 位指数位 + 1 位尾数位），浮点数的计算公式为：

```
ans = (-1)^s * 2^(p-b) * (1 + d1/2)
```

其中：
- `s`：符号位
- `p`：指数位的值（0-3）
- `b = 2^(e-1) - 1 = 2^(2-1) - 1 = 1`（偏置值）
- `d1`：尾数位的值（0 或 1）

**NVFP4 的特殊之处**：
- 取消了 inf 和 nan 的表示
- 最大值可以表示到 ±6.0（而标准 E2M1 只能表示到 ±3.0）
- 0000 表示 +0，1000 表示 -0
- 0001 表示 0.5，1001 表示 -0.5

完整的 E2M1 值表：

| E2M1 | 0000 | 0001 | 0010 | 0011 | 0100 | 0101 | 0110 | 0111 | 1000 | 1001 | 1010 | 1011 | 1100 | 1101 | 1110 | 1111 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 值   | +0   | 0.5  | 1.0  | 1.5  | 2.0  | 3.0  | 4.0  | 6.0  | -0   | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |

#### 3.1.2 量化过程

NVFP4 采用 **Per-Group 量化**，量化粒度为 16 个元素一组。量化因子使用 FP8(E4M3) 格式存储。

**量化步骤**：

给定一组数据 `X`，假设 `Xg` 表示一个 group 的数据（16 个元素）。

1. **计算 scale1**（每个 group 的原始 scale）：
   ```
   scale1 = max(abs(Xg)) / 6.0
   ```
   其中 6.0 是 NVFP4 的最大值。

2. **量化 scale**（将 scale 量化到 FP8）：
   ```
   global_scale = 6.0 * 448.0 / max(abs(X))
   scale2 = global_scale * scale1
   scale2 = max(abs(Xg)) / max(abs(X)) * 448.0
   ```
   此时 scale2 被缩放到 FP8(E4M3) 的范围（最大值 448.0），然后量化到 FP8：
   ```
   scale2_fp8 = quant_fp8(scale2)
   ```

3. **量化数据 X**：
   ```
   scale2_fp32 = cvt2fp32(scale2_fp8)
   Xquant = quant_fp4(X * global_scale / scale2_fp32)
   ```
   近似等于：
   ```
   Xquant ≈ quant_fp4(X / scale1)
   ```

4. **FP4 矩阵乘法**：
   ```
   ans = Aquant * Bquant * Ascale2 * Bscale2 / Aglobal_scale / Bglobal_scale
   ```
   简化为：
   ```
   ans ≈ Aquant * Bquant * Ascale1 * Bscale1
   ```

**关键点**：
- Weight 和 Activation 都使用 Per-Group 量化，group size 为 16
- 量化 scale 使用 FP8(E4M3) 格式存储
- 需要对 scale 本身进行量化，这是与常见 W8A8-INT8 量化的主要区别

## 0x2.2 MX-Formats 量化原理

#### 3.2.1 数据格式与量化因子

MX-Formats（Microscaling Formats）是 OCP（Open Compute Project）定义的标准化微缩放浮点格式。

**源数据格式**：fp16/bf16

**目标数据格式**：mxfp4/6/8

**量化因子数据格式**：E8M0
- E8M0 与 fp32 数值范围一致
- 经过 rounding 后可直接存储量化因子
- 缺点：尾数的丢失会影响精度

**量化粒度**：[1×32]
- 每 32 个元素共享一个量化因子

**量化维度**：
- 沿着 K 维度量化（GEMM 的 K 维度）

#### 3.2.2 Rounding 与 Clamp

CUDA 通过 PTX 指令或内置函数高效地完成 Rounding 和 Clamp 操作。

例如，`cvt.rn.satfinite.e2m1x2.f32` 可以将两个 fp32 类型的输入转换为两个 fp4 类型的输出：
- **Rounding 模式**：`rn`（round-to-nearest-even）
- **Clamp 模式**：`satfinite`（钳制到目标范围内的最大有限值，排除无穷和 NaN）

#### 3.2.3 数据布局与量化因子布局

**数据布局**：
- MXFP4：两个 fp4 值打包为一个 uint8
- MXFP6：每 4 个 fp6 值打包为 3 个 uint8
- MXFP8：直接使用 uint8 存储

**量化因子布局**：
Cutlass Block Scaled GEMMs 对量化因子布局有特殊的 swizzle 要求，以满足矩阵运算加速。布局格式为：
```
[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
```

#### 3.2.4 MX-Formats 与 NVFP4 的区别

| 特性 | NVFP4 | MX-Formats |
|------|-------|------------|
| 量化粒度 | 16 个元素 | 32 个元素 |
| 量化因子格式 | FP8(E4M3) | FP8(E8M0) |
| 需要量化 scale | 是 | 否 |
| 全局缩放因子 | 需要 | 不需要 |

---

# 0x3. 代码实现

## 0x3.1 NVFP4 量化实现

### 核心数据结构

```cpp
// 类型转换器：用于在 Type 和 Type2 之间转换（half <-> half2, bfloat16 <-> bfloat162）
template <typename T>
struct TypeConverter {
  using Type = half2;  // 默认
};

template <>
struct TypeConverter<half> {
  using Type = half2;  // half 对应 half2
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;  // bfloat16 对应 bfloat162
};

// 打包向量：16 字节的打包数据类型
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // 4 个 Type2，共 8 个元素
};
```

#### 4.1.2 FP32 到 E2M1 的转换

这是量化的核心操作，使用 PTX 内联汇编实现高效转换：

```cpp
// 将 4 个 float2 值（共 8 个 float）转换为 8 个 e2m1 值（打包为 1 个 uint32_t）
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{"
      ".reg .b8 byte0;"           // 定义 4 个 8-bit 寄存器
      ".reg .b8 byte1;"
      ".reg .b8 byte2;"
      ".reg .b8 byte3;"
      // 每条指令将 2 个 float32 转换为 2 个 e2m1（共 1 个字节）
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;"  // array[0].y, array[0].x -> byte0
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;"  // array[1].y, array[1].x -> byte1
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;"  // array[2].y, array[2].x -> byte2
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;"  // array[3].y, array[3].x -> byte3
      // 将 4 个字节打包为 1 个 uint32_t
      "mov.b32 %0, {byte0, byte1, byte2, byte3};"
      "}"
      : "=r"(val)  // 输出：val
      : "f"(array[0].x), "f"(array[0].y),  // 输入：8 个 float
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
}
```

**关键点**：
- `cvt.rn.satfinite.e2m1x2.f32`：PTX 指令，将 2 个 float32 转换为 2 个 e2m1
- `rn`：round-to-nearest-even（最近偶数舍入）
- `satfinite`：饱和到有限值范围，排除 inf 和 nan
- 4 条转换指令 + 1 条打包指令，高效完成 8 个值的转换

### 快速倒数计算

```cpp
// 使用 PTX 指令实现快速近似倒数
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  // rcp.approx.ftz.f32：快速近似倒数，flush-to-zero
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
```

**优势**：
- 比标准的 `1.0f / a` 快得多
- `ftz`（flush-to-zero）：将非规格化数刷新为零，提高性能

### 量化因子布局计算

Cutlass Block Scaled GEMM 要求量化因子使用特殊的 swizzled 布局：

```cpp
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int numCols, SFType* SFout) {
  
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 只有特定线程写入 SF（每 CVT_FP4_NUM_THREADS_PER_SF 个线程写一个 SF）
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF 向量索引（K 维度每 16 个元素共享一个 SF）
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF 布局：[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 索引：[mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // 计算 M 维度的 tile 索引
    int32_t mTileIdx = mIdx / (32 * 4);  // 每个 M tile 包含 128 行
    int factor = CVT_FP4_SF_VEC_SIZE * 4;  // 16 * 4 = 64
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;  // M tile 的步长

    // 计算 K 维度的 tile 索引
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;  // K tile 的步长

    // M tile 内部布局是列主序 [32, 4]
    int32_t outerMIdx = (mIdx % 32);  // 外层 M 索引（0-31）
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;  // 内层 M 索引（0-3）
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);  // 内层 K 索引（0-3）
    int64_t innerKStride = 1;

    // 计算全局偏移
    int64_t SFOffset = mTileIdx * mTileStride + 
                       kTileIdx * kTileStride + 
                       outerMIdx * outerMStride +
                       innerMIdx * innerMStride + 
                       innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}
```

布局说明：
- SF 布局采用 5 维结构: `[numMTiles, numKTiles, 32, 4, 4]`
- M 维度被分为 128 行一个 tile(32×4)
- K 维度被分为 64 个元素一个 tile(16×4)
- M tile 内部采用列主序布局
- 这种布局优化了 Tensor Core 的访问模式

### 核心量化 Kernel

这是执行量化的主要 kernel 函数：

```cpp
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(256, 6) cvt_fp16_to_fp4(
    int32_t numRows, int32_t numCols, Type const* in, 
    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = 
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);  // 16 / 8 = 2
  
  // 获取全局缩放因子
  // SFScale 与下一个 GEMM 的 alpha 相同，即 (448.0 / (Alpha_A / 6.0))
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // 输入张量的行/列循环
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; 
         colIdx += blockDim.x) {
      
      // 读取输入数据（16 字节，8 个元素）
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 输出偏移（8 个元素打包为 1 个 uint32_t）
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // 获取 SF 输出地址
      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
          rowIdx, colIdx, numCols, SFout);

      // 执行量化
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
}
```

Kernel 配置:
- `__launch_bounds__(256, 6)`: 每个 block 256个线程,每个 SM 最多 6个 block
- 每个线程处理 8个元素
- 使用 grid-stride loop 处理所有行

### Warp 级量化函数

这是在 warp 内执行量化的核心函数：

```cpp
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  
  // 1. 计算局部最大值（每个线程处理 8 个元素）
  auto localMax = __habs2(vec.elts[0]);  // 取绝对值
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 逐对比较
  }

  // 2. Warp 内规约，获取 16 个元素的最大值（2 个线程）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 计算量化因子（SF）
  // vecMax / 6.0 是原始 scale，乘以 SFScaleVal 后量化到 FP8
  float SFValue = SFScaleVal * (vecMax * 0.16666666666666666f);  // 0.1666... = 1/6
  
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
    // 使用 E8M0 格式
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // 使用 E4M3 格式（默认）
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }

  // 4. 计算输出缩放因子
  // 最终的数据 = 原始数据 * outputScale，然后量化到 FP4
  float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;

  // 5. 写入量化因子到全局内存
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6. 转换输入数据到 float 并缩放
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 7. 转换为 e2m1 值
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  return e2m1Vec;
}
```

关键步骤:
1. **局部最大值计算**：使用 `__habs2` 和 `__hmax2` 进行向量化操作
2. **Warp 规约**：使用 `__shfl_xor_sync` 在 warp 内交换数据，获取 16 个元素的最大值
3. **量化因子计算**：`vecMax / 6.0 * SFScaleVal`，然后量化到 FP8
4. **输出缩放**：计算 `outputScale = SFScaleVal / SFValue`，用于缩放原始数据
5. **数据转换**：将 fp16/bf16 转换为 float，乘以 outputScale，再转换为 e2m1

### 主机端调用接口

```cpp
void scaled_nvfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, 
    torch::Tensor& output_sf, torch::Tensor const& input_sf) {
  
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  // 检查 N 维度必须是 16 的倍数
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount = getMultiProcessorCount();

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  bool useUE8M0 = false;  // 默认使用 E4M3

  // 根据输入类型分发
  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, 
                           sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, 
                           sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
}
```

---

## 0x3.2 量化 Kernel 线程模型

在看 GEMM 实现之前,先了解一下量化 kernel 的线程模型,这对理解性能很重要。

### NVFP4 量化 Kernel 线程模型

#### Grid 和 Block 配置

```cpp
// 每个线程处理 8 个元素
dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));

// 每个 SM 的 block 数量
int const numBlocksPerSM = 1536 / block.x;

// Grid 大小
dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));
```

配置解析:

1. **Block 大小计算**：
   - `block.x = min(n / 8, 256)`
   - 每个线程处理 8 个元素，所以需要 `n / 8` 个线程
   - 最多 256 个线程/block（受 `__launch_bounds__(256, 6)` 限制）
   - 示例：
     - `n = 4096` → `block.x = min(512, 256) = 256`
     - `n = 1024` → `block.x = min(128, 256) = 128`

2. **每个 SM 的 Block 数量**：
   - `numBlocksPerSM = 1536 / block.x`
   - 1536 是每个 SM 的最大线程数（Blackwell 架构）
   - 目的：充分利用 SM 资源
   - 示例：
     - `block.x = 256` → `numBlocksPerSM = 6`
     - `block.x = 128` → `numBlocksPerSM = 12`

3. **Grid 大小**：
   - `grid.x = min(m, multiProcessorCount * numBlocksPerSM)`
   - 每个 block 处理一行数据
   - 使用 grid-stride loop 处理所有行
   - 示例（假设 144 个 SM）：
     - `m = 1024, block.x = 256` → `grid.x = min(1024, 144 × 6) = 864`
     - `m = 100, block.x = 256` → `grid.x = min(100, 864) = 100`

#### Kernel 执行模型

```cpp
__launch_bounds__(256, 6)  // 每个 block 256 线程，每个 SM 最多 6 个 block
cvt_fp16_to_fp4(...) {
  // Grid-stride loop 处理行
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    // Block-stride loop 处理列
    for (int colIdx = threadIdx.x; colIdx < numCols / 8; colIdx += blockDim.x) {
      // 每个线程处理 8 个元素
      // ...
    }
  }
}
```

执行流程:

1. **行级并行**（Grid 维度）：
   - 每个 block 负责一行或多行
   - 使用 grid-stride loop：`rowIdx += gridDim.x`
   - 保证所有行都被处理

2. **列级并行**（Block 维度）：
   - 每个线程负责 8 个连续元素
   - 使用 block-stride loop：`colIdx += blockDim.x`
   - 线程 0 处理元素 [0-7], [256×8-256×8+7], ...
   - 线程 1 处理元素 [8-15], [256×8+8-256×8+15], ...

3. **Warp 级协作**：
   - 每 2 个线程（16 个元素）计算一个 scale factor
   - 使用 `__shfl_xor_sync` 进行 warp 内规约
   - 32 个线程（一个 warp）处理 256 个元素，生成 16 个 scale factors

#### 线程到数据的映射

假设 `m=1024, n=4096, block.x=256, grid.x=864`：

```
Block 0:
  Thread 0:  处理 row 0, cols [0-7], [2048-2055], ...
  Thread 1:  处理 row 0, cols [8-15], [2056-2063], ...
  ...
  Thread 255: 处理 row 0, cols [2040-2047], [4088-4095]

Block 1:
  Thread 0:  处理 row 1, cols [0-7], [2048-2055], ...
  ...

Block 863:
  Thread 0:  处理 row 863, cols [0-7], [2048-2055], ...
  ...

Block 0 (第二轮):
  Thread 0:  处理 row 864, cols [0-7], [2048-2055], ...
  ...
```

#### 内存访问模式

全局内存读取(Coalesced):
```
Warp 0 (Threads 0-31):
  Thread 0:  读取 input[row][0:8]     → 16 bytes
  Thread 1:  读取 input[row][8:16]    → 16 bytes
  ...
  Thread 31: 读取 input[row][248:256] → 16 bytes
  
总计：32 个线程，32 × 16 = 512 bytes，完美 coalesced
```

全局内存写入:
```
量化数据（每 2 个 FP4 打包成 1 个 uint8）：
  Thread 0:  写入 output[row][0]     → 4 bytes (8个FP4)
  Thread 1:  写入 output[row][1]     → 4 bytes
  ...

Scale factors（swizzled layout）：
  Thread 0:  写入 SF[swizzled_offset] → 1 byte (FP8)
  Thread 2:  写入 SF[swizzled_offset] → 1 byte
  ...（每 2 个线程写入一个 SF）
```

### MXFP4 量化 Kernel 线程模型

MXFP4 的线程模型和 NVFP4 差不多,主要差别在量化粒度。

#### 关键差异

```cpp
// NVFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 16;  // 16 个元素/组
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 16 / 8 = 2;  // 2 个线程/SF

// MXFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 32;  // 32 个元素/组
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 32 / 8 = 4;  // 4 个线程/SF
```

影响:

1. **Warp 规约次数**：
   - NVFP4：1 次 `__shfl_xor_sync(mask, val, 1)`（2 个线程规约）
   - MXFP4：2 次 `__shfl_xor_sync`（4 个线程规约）
   ```cpp
   // MXFP4 需要额外的规约步骤
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
   ```

2. **Scale Factor 密度**：
   - NVFP4：每 16 个元素 1 个 SF → 每 256 个元素 16 个 SF
   - MXFP4：每 32 个元素 1 个 SF → 每 256 个元素 8 个 SF

3. **内存访问模式**：
   - 相同的 coalesced 读取模式
   - 不同的 SF 写入模式（更稀疏）

### 性能分析

#### 占用率(Occupancy)

NVFP4/MXFP4 量化 Kernel:
```
__launch_bounds__(256, 6)
- 每个 block：256 个线程
- 每个 SM：最多 6 个 block
- 理论占用率：(256 × 6) / 1536 = 100%
```

资源使用:
- **寄存器**：每个线程约 40-50 个寄存器
- **共享内存**：0 bytes（不使用共享内存）
- **Warp 数量**：256 / 32 = 8 个 warp/block

#### 吞吐量估算

假设输入：`m=1024, n=4096, dtype=bfloat16`

NVFP4:
```
数据量：1024 × 4096 × 2 bytes = 8 MB

每个线程处理：8 个元素 = 16 bytes
每个 block 处理：256 × 16 = 4 KB（一行的一半）
需要 2 轮才能处理完一行

总 block 数：1024 行 × 2 轮 = 2048 block-iterations
实际 block 数：min(1024, 144 × 6) = 864 blocks
Grid-stride 轮数：⌈1024 / 864⌉ = 2 轮

内存带宽需求：
- 读取：8 MB
- 写入：2 MB (量化数据) + 64 KB (SF) ≈ 2.06 MB
- 总计：≈ 10 MB

假设带宽 2 TB/s，理论时间：10 MB / 2000 GB/s ≈ 5 μs
实际时间（考虑计算）：约 10-20 μs
```

#### 性能瓶颈

1. **内存带宽受限**：
   - 量化操作相对简单
   - 主要时间花在内存读写上
   - Coalesced 访问模式优化了带宽利用

2. **计算开销**：
   - PTX 指令 `cvt.rn.satfinite.e2m1x2.f32` 非常快
   - Warp 规约开销很小
   - FP8 转换（查表或内置指令）

3. **优化空间**：
   - 已经接近最优（100% 占用率）
   - Coalesced 内存访问
   - 向量化加载（16 bytes/线程）

---

## 0x3.3 NVFP4 矩阵乘法实现

NVFP4 的矩阵乘法基于 CUTLASS 3.x 的 Block Scaled GEMM。

### GEMM 配置结构

```cpp
struct Fp4GemmSm120 {
    // A 矩阵配置
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4 类型
    using LayoutATag = cutlass::layout::RowMajor;                   // 行主序
    static constexpr int AlignmentA = 32;                           // 对齐要求：32 个元素

    // B 矩阵配置
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;                // 列主序
    static constexpr int AlignmentB = 32;

    // C/D 矩阵配置
    using ElementD = cutlass::bfloat16_t;                           // 输出类型
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    
    // 累加器配置
    using ElementAccumulator = float;                               // 内部累加使用 float
    using ArchTag = cutlass::arch::Sm120;                           // Blackwell 架构
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; // Block Scaled Tensor Op

    // 性能配置
    using ThreadBlockShape = Shape<_128,_128,_128>;                 // Tile 大小：128×128×128
    using ClusterShape = Shape<_1,_1,_1>;                           // Cluster 大小

    // Epilogue 配置：支持 per-column bias
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    // 构建 Collective Epilogue
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        EVTOp
    >::CollectiveOp;

    // 构建 Collective Mainloop
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // GEMM Kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
```

配置说明:
- **ElementA/B**：使用 `nv_float4_t<float_e2m1_t>` 表示 NVFP4 类型
- **AlignmentA/B**：32 个元素对齐，确保高效的内存访问
- **ThreadBlockShape**：128×128×128 的 tile 大小，平衡寄存器使用和共享内存
- **OpClassBlockScaledTensorOp**：使用 Block Scaled Tensor Core 操作
- **EVTOp**：支持 per-column bias 的 epilogue 融合操作

### GEMM Kernel 线程模型

CUTLASS 3.x 的 GEMM kernel 使用了层次化的线程模型,下面看看它是怎么工作的。

#### Tile 层次结构

```
问题规模: M × N × K

CTA (Cooperative Thread Array) Tile: 128 × 128 × 128
  └─ Warp Tile: 根据 warp 数量自动划分
      └─ Thread Tile: 每个线程处理的数据块
          └─ Instruction Tile: Tensor Core 指令处理的最小单元
```

层次详解:

1. **CTA Tile (ThreadBlockShape)**：
   - 大小：`128 × 128 × 128` (M × N × K)
   - 含义：一个 thread block 处理的矩阵块
   - 选择原因：
     - 128×128 的输出 tile 适合 Tensor Core
     - K=128 提供足够的数据重用
     - 平衡寄存器和共享内存使用

2. **Cluster (ClusterShape)**：
   - 大小：`1 × 1 × 1`
   - 含义：不使用 thread block cluster（Hopper+ 特性）
   - Blackwell 可以使用，但这里选择单 CTA 模式

3. **Warp 分配**：
   - 每个 CTA 有多个 warp（通常 4-8 个）
   - 每个 warp 负责 CTA tile 的一部分
   - 使用 CUTLASS 的 warp 专门化（warp specialization）

#### 线程组织

假设一个典型的配置：

```cpp
// CTA 配置
ThreadBlockShape = 128 × 128 × 128
每个 CTA 的线程数 = 128 或 256（由 CUTLASS 自动确定）
Warp 数量 = 线程数 / 32 = 4 或 8 个 warp

// Warp 到输出 tile 的映射（示例：4 个 warp）
Warp 0: 处理输出 tile [0:64, 0:64]
Warp 1: 处理输出 tile [0:64, 64:128]
Warp 2: 处理输出 tile [64:128, 0:64]
Warp 3: 处理输出 tile [64:128, 64:128]
```

#### Mainloop 执行流程

```cpp
// 伪代码展示 GEMM mainloop
for (int k_tile = 0; k_tile < K; k_tile += 128) {
  // 1. 从全局内存加载 A 和 B tile 到共享内存
  // 使用 cp.async 异步加载
  load_A_tile_to_smem(A, k_tile);  // 128×128 的 A tile
  load_B_tile_to_smem(B, k_tile);  // 128×128 的 B tile
  load_SFA_to_smem(SFA, k_tile);   // Scale factors for A
  load_SFB_to_smem(SFB, k_tile);   // Scale factors for B
  
  // 2. 同步等待数据加载完成
  cp.async.wait_group<0>();
  __syncthreads();
  
  // 3. 从共享内存加载到寄存器，执行 Tensor Core 运算
  for (int k_frag = 0; k_frag < 128; k_frag += 16) {
    // 加载 A 和 B 的 fragment 到寄存器
    load_A_fragment(A_frag, k_frag);
    load_B_fragment(B_frag, k_frag);
    
    // 加载对应的 scale factors
    load_SFA_fragment(SFA_frag, k_frag);
    load_SFB_fragment(SFB_frag, k_frag);
    
    // 执行 Block Scaled MMA (Matrix Multiply-Accumulate)
    // 使用 Tensor Core 指令：mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32
    mma_with_scaling(C_frag, A_frag, B_frag, SFA_frag, SFB_frag);
  }
  
  __syncthreads();
}
```

**关键点**：

1. **异步加载（cp.async）**：
   - 使用 CUDA 的异步拷贝指令
   - 隐藏全局内存访问延迟
   - 支持 software pipelining

2. **共享内存布局**：
   - A tile: `128 × 128` 的 FP4 数据（打包后 8 KB）
   - B tile: `128 × 128` 的 FP4 数据（打包后 8 KB）
   - Scale factors: swizzled layout，优化 bank conflict
   - 总共约 16-20 KB 共享内存/CTA

3. **Tensor Core 指令**：
   - `mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32`
   - 输入：E2M1 (FP4)
   - 输出：FP32 累加器
   - 形状：16×8×32 (M×N×K)

#### Epilogue 执行流程

```cpp
// Epilogue: 将累加器写回全局内存
// 支持融合操作：D = alpha * C + bias

// 1. 从寄存器读取累加器
for each thread's output elements {
  // 2. 应用 scaling (alpha)
  output = accumulator * alpha;
  
  // 3. 添加 bias（如果有）
  if (bias) {
    output += bias[col];  // per-column bias
  }
  
  // 4. 类型转换（FP32 → BF16）
  output_bf16 = convert_to_bf16(output);
  
  // 5. 写回全局内存
  D[row][col] = output_bf16;
}
```

Epilogue 优化:
- **向量化写入**：一次写入多个元素
- **Coalesced 访问**：warp 内的线程访问连续地址
- **寄存器重用**：累加器直接用于 epilogue，无需额外共享内存

#### 内存访问模式

全局内存 → 共享内存(Mainloop):
```
A 矩阵加载（每个 k_tile）：
  - 大小：128 × 128 × 0.5 bytes = 8 KB (FP4 打包)
  - 模式：Coalesced，使用 cp.async
  - 带宽利用：接近峰值

B 矩阵加载（每个 k_tile）：
  - 大小：128 × 128 × 0.5 bytes = 8 KB
  - 模式：Coalesced，转置访问优化
  
Scale Factors 加载：
  - A_SF: (128/16) × (128/16) = 8 × 8 = 64 个 FP8 = 64 bytes
  - B_SF: 同上
  - 模式：Swizzled layout，避免 bank conflict
```

共享内存 → 寄存器(Tensor Core):
```
每个 warp 每次迭代：
  - 加载 A fragment: 16 × 32 × 0.5 bytes = 256 bits
  - 加载 B fragment: 8 × 32 × 0.5 bytes = 128 bits
  - 加载 SF: 少量 FP8 值
  
使用 ldmatrix 指令：
  - 高效的矩阵加载指令
  - 直接从共享内存加载到 Tensor Core 寄存器
```

寄存器 → 全局内存(Epilogue):
```
每个线程写入：
  - 通常 4-8 个 BF16 值
  - 向量化写入：128-bit store
  - Coalesced 模式
```

#### 性能特征

计算吞吐量:
```
Tensor Core 峰值性能（Blackwell）：
  - FP4 × FP4 → FP32: ~2000 TFLOPS

实际性能（GEMM）：
  - 取决于矩阵大小和 K 维度
  - 大矩阵（M, N, K > 1024）：可达 80-90% 峰值
  - 小矩阵：受限于启动开销和占用率
```

内存带宽需求:
```
假设 M=N=K=4096：
  - A 矩阵：4096 × 4096 × 0.5 bytes = 8 MB
  - B 矩阵：4096 × 4096 × 0.5 bytes = 8 MB
  - Scale factors：约 128 KB
  - 输出 D：4096 × 4096 × 2 bytes = 32 MB
  - 总计：≈ 48 MB

计算量：2 × 4096³ ≈ 137 GFLOPS
计算强度：137 GFLOPS / 48 MB ≈ 2854 FLOPS/byte

结论：计算密集型（compute-bound），内存带宽不是瓶颈
```

#### 与量化 Kernel 的对比

| 特性 | 量化 Kernel | GEMM Kernel |
|------|-------------|-------------|
| **瓶颈** | 内存带宽 | 计算（Tensor Core） |
| **共享内存** | 0 bytes | 16-20 KB/CTA |
| **寄存器压力** | 低（40-50/线程） | 高（100+/线程） |
| **占用率** | 100% | 50-75%（受寄存器限制） |
| **主要优化** | Coalesced 访问 | Tile 大小、Pipeline |
| **执行时间** | 微秒级 | 毫秒级（大矩阵） |

### 参数构建函数

```cpp
typename Fp4GemmSm120::Gemm::Arguments args_from_options_nvfp4_nvfp4(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    at::Tensor const& alpha, c10::optional<torch::Tensor> const& bias,
    int64_t M, int64_t N, int64_t K) {
  
  using Sm1xxBlkScaledConfig = 
      typename Fp4GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  
  // 计算 stride
  auto stride_A = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideD{}, {m, n, 1});

  // 计算 scale factor 的 layout
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  if (bias) {
    // 带 bias 的情况
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {// Epilogue arguments
       {},
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    
    // 设置 fusion 参数
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Fp4GemmSm120::Gemm::ElementC const*>(
        bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    
    return arguments;
  } else {
    // 不带 bias 的情况（类似，省略 bias 设置）
    // ...
  }
}
```

关键点:
- 计算 stride 和 scale factor layout
- 构建 CUTLASS Arguments 结构
- 支持带 bias 和不带 bias 两种情况
- 使用 epilogue fusion 将 bias 添加融合到 GEMM 中

### GEMM 执行函数

```cpp
void runGemmNvfp4Sm120(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    at::Tensor const& alpha, c10::optional<torch::Tensor> const& bias,
    int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  
  typename Fp4GemmSm120::Gemm gemm;

  // 构建参数
  auto arguments = args_from_options_nvfp4_nvfp4(
      D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  
  // 分配 workspace
  size_t workspace_size = Fp4GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  // 检查是否可以执行
  CUTLASS_CHECK(gemm.can_implement(arguments));
  
  // 初始化
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  
  // 执行
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
```

### 主机端接口

```cpp
void cutlass_scaled_nvfp4_mm_sm120(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, c10::optional<torch::Tensor> const& bias) {

  // 输入检查
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1], "a and b shapes cannot be multiplied");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;  // 因为两个 FP4 打包为一个 uint8

  // 对齐检查
  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // 计算 rounded 尺寸
  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);  // k/16 是 scale factor 的数量

  // 检查 scale factor 的尺寸
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to shape (", rounded_m, "x", rounded_k, ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to shape (", rounded_n, "x", rounded_k, ")");

  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  runGemmNvfp4Sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
}
```

---

## 0x3.4 MXFP4 量化实现

MXFP4 的量化实现和 NVFP4 差不多,主要差别在于:

### 主要差异

```cpp
// NVFP4 vs MXFP4 的关键差异

// 1. 量化粒度
constexpr int CVT_FP4_SF_VEC_SIZE_NVFP4 = 16;  // NVFP4: 16 个元素一组
constexpr int CVT_FP4_SF_VEC_SIZE_MXFP4 = 32;  // MXFP4: 32 个元素一组

// 2. 量化因子格式
// NVFP4: 使用 E4M3，需要 global_scale
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);

// MXFP4: 使用 E8M0，不需要 global_scale
__nv_fp8_e8m0 tmp;
tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);

// 3. 输出缩放计算
// NVFP4: 需要考虑 global_scale
float outputScale = SFScaleVal * reciprocal_approximate_ftz(SFValue);

// MXFP4: 直接使用 SF 的倒数
float outputScale = reciprocal_approximate_ftz(SFValue);
```

### MXFP4 Warp 量化函数

```cpp
template <class Type>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, uint8_t* SFout) {
  
  // 1. 计算局部最大值（每个线程 8 个元素）
  auto localMax = __habs2(vec.elts[0]);
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // 2. Warp 内规约，获取 32 个元素的最大值（4 个线程）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);  // 额外一次规约
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 计算量化因子（直接除以 6.0，不需要 global_scale）
  float SFValue = vecMax * 0.16666666666666666f;
  
  // 4. 量化到 E8M0
  uint8_t fp8SFVal;
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);
  fp8SFVal = tmp.__x;

  // 5. 计算输出缩放（不需要 global_scale）
  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6-7. 转换和量化（与 NVFP4 相同）
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
  return e2m1Vec;
}
```

**关键差异**：
- **规约次数**：MXFP4 需要两次 `__shfl_xor_sync`（32 个元素需要 4 个线程）
- **无 global_scale**：MXFP4 直接使用 `vecMax / 6.0` 作为 SF
- **E8M0 格式**：使用 `__nv_cvt_float_to_e8m0` 转换

---

## 0x3.5 MXFP4 矩阵乘法实现

MXFP4 的矩阵乘法配置和 NVFP4 差不多,主要差别:

### GEMM 配置差异

```cpp
struct Mxfp4GemmSm120 {
    // A 矩阵配置
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  // 使用 mx_float4_t
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128;  // 更大的对齐要求：128 个元素

    // B 矩阵配置
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128;

    // 其他配置与 NVFP4 相同
    // ...
};
```

**关键差异**：
- **ElementA/B**：使用 `mx_float4_t` 而不是 `nv_float4_t`
- **AlignmentA/B**：128 个元素对齐（MXFP4 的 group size 是 32，4 个 group）
- **Scale Factor 类型**：使用 `float_ue8m0_t` 而不是 `float_ue4m3_t`

### 对齐检查差异

```cpp
void cutlass_scaled_mxfp4_mm_sm120(...) {
  // ...
  
  auto const k = A.sizes()[1] * 2;
  
  // MXFP4 需要更严格的对齐
  constexpr int alignment = 128;  // NVFP4 是 32
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // Scale factor 的计算也不同
  int rounded_k = round_up(k / 32, 4);  // MXFP4: k/32，NVFP4: k/16
  
  // ...
}
```

---

# 0x4. LightX2V 项目中的实际使用

## 0x4.1 集成方式

LightX2V 项目通过 `MMWeight` 类体系来集成 lightx2v_kernel,实现了模型权重的量化和推理加速。

### 导入量化算子

```python
# lightx2v/common/ops/mm/mm_weight.py
try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_mxfp4_mm,
        cutlass_scaled_mxfp6_mxfp8_mm,
        cutlass_scaled_mxfp8_mm,
        cutlass_scaled_nvfp4_mm,
        scaled_mxfp4_quant,
        scaled_mxfp6_quant,
        scaled_mxfp8_quant,
        scaled_nvfp4_quant,
    )
except ImportError:
    # 如果没有安装 lightx2v_kernel,使用 None
    scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm = None, None
    scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm = None, None
    scaled_mxfp6_quant, cutlass_scaled_mxfp6_mxfp8_mm = None, None
    scaled_mxfp8_quant, cutlass_scaled_mxfp8_mm = None, None
```

## 0x4.2 NVFP4 量化权重类

LightX2V 实现了 `MMWeightNvfp4` 类来管理 NVFP4 量化的权重。

### 类定义

```python
@MM_WEIGHT_REGISTER("nvfp4")
class MMWeightNvfp4(MMWeightQuantNvfp4Template):
    """
    NVFP4 量化权重类
    - Weight: NVFP4 格式
    - Act: NVFP4 动态量化
    - Kernel: lightx2v_kernel
    """
    
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
        )
        # 设置量化函数
        self.load_func = self.load_nvfp4
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_nvfp4
```

### 权重加载

权重加载时需要加载以下数据:

```python
def _get_cuda_tensor_pair(self, source, is_lazy):
    # 1. 加载量化后的权重
    weight = source.get_tensor(self.weight_name).to(AI_DEVICE)
    
    # 2. 加载权重的 scale factors
    scale = source.get_tensor(self.weight_scale_name).to(AI_DEVICE)
    
    # 3. 计算或加载 input_global_scale
    if self.input_absmax_name in source:
        # 从校准数据计算
        input_absmax = source.get_tensor(self.input_absmax_name)
        input_global_scale = (2688.0 / input_absmax).to(torch.float32)
        weight_global_scale = source.get_tensor(self.weight_global_scale_name)
        alpha = 1.0 / (input_global_scale * weight_global_scale)
    else:
        # 直接加载
        input_global_scale = source.get_tensor(self.input_global_scale_name)
        alpha = source.get_tensor(self.alpha_name)
    
    return weight, scale, input_global_scale, alpha
```

关键参数说明:
- `weight`: 量化后的权重,shape 为 `(out_features, in_features//2)`,dtype 为 `uint8`
- `scale`: 权重的 scale factors,dtype 为 `float8_e4m3fn`
- `input_global_scale`: 输入的全局缩放因子,用于量化激活
- `alpha`: 输出缩放因子,`alpha = 1.0 / (input_global_scale * weight_global_scale)`

### 推理过程

```python
def apply(self, input_tensor):
    # 1. 量化输入激活
    # input_tensor: (batch_size, in_features), dtype=bfloat16
    input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
    # input_tensor_quant: (batch_size, in_features//2), dtype=uint8
    # input_tensor_scale: (batch_size, in_features//16), dtype=float8_e4m3fn
    
    # 2. 执行量化矩阵乘法
    output_tensor = cutlass_scaled_nvfp4_mm(
        input_tensor_quant,      # 量化后的输入
        self.weight,             # 量化后的权重
        input_tensor_scale,      # 输入的 scale factors
        self.weight_scale,       # 权重的 scale factors
        alpha=self.alpha,        # 输出缩放因子
        bias=self.bias,          # 可选的 bias
    )
    # output_tensor: (batch_size, out_features), dtype=bfloat16
    
    return output_tensor
```

### 激活量化函数

```python
def act_quant_nvfp4(self, x):
    """
    对输入激活进行 NVFP4 量化
    
    Args:
        x: 输入张量,shape=(batch_size, in_features), dtype=bfloat16
    
    Returns:
        input_tensor_quant: 量化后的张量,shape=(batch_size, in_features//2)
        input_tensor_scale: scale factors,shape=(batch_size, in_features//16)
    """
    input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
        x, 
        self.input_global_scale
    )
    return input_tensor_quant, input_tensor_scale
```

## 0x4.3 完整的推理流程

下面是一个完整的推理流程示例:

```python
# 1. 创建量化权重对象
mm_weight = MMWeightNvfp4(
    weight_name="transformer.blocks.0.attn.qkv.weight",
    bias_name="transformer.blocks.0.attn.qkv.bias",
    lazy_load=True,
    lazy_load_file="/path/to/quantized_model",
)

# 2. 加载量化权重
mm_weight.load(weight_dict)

# 3. 将权重加载到 GPU
mm_weight.to_cuda()

# 4. 推理
input_tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
output_tensor = mm_weight.apply(input_tensor)
# 输出: (1024, 12288), dtype=bfloat16

# 5. 推理完成后可以卸载到 CPU
mm_weight.to_cpu()
```

## 0x4.4 量化模型转换

LightX2V 提供了模型量化转换工具,可以将 FP16/BF16 模型转换为 NVFP4 量化模型。

### 权重量化

```python
# tools/convert/quant/quant.py
def quantize_weight_nvfp4(weight, calib_data):
    """
    将权重量化为 NVFP4 格式
    
    Args:
        weight: 原始权重,shape=(out_features, in_features), dtype=bfloat16
        calib_data: 校准数据,用于计算 input_global_scale
    
    Returns:
        quantized_weight: 量化后的权重
        weight_scale: 权重的 scale factors
        input_global_scale: 输入的全局缩放因子
        weight_global_scale: 权重的全局缩放因子
    """
    # 1. 计算 input_global_scale
    input_absmax = calib_data.abs().max()
    input_global_scale = 2688.0 / input_absmax
    
    # 2. 量化权重
    weight = weight.to("cuda").to(torch.bfloat16)
    quantized_weight, weight_scale = scaled_nvfp4_quant(
        weight, 
        torch.tensor(input_global_scale, device="cuda")
    )
    
    # 3. 计算 weight_global_scale
    weight_absmax = weight.abs().max()
    weight_global_scale = 2688.0 / weight_absmax
    
    return quantized_weight, weight_scale, input_global_scale, weight_global_scale
```

### 保存量化模型

```python
def save_quantized_model(model, output_path):
    """保存量化后的模型"""
    state_dict = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'mm_weight') and isinstance(module.mm_weight, MMWeightNvfp4):
            # 保存量化权重
            state_dict[f"{name}.weight"] = module.mm_weight.weight
            state_dict[f"{name}.weight_scale"] = module.mm_weight.weight_scale
            state_dict[f"{name}.input_global_scale"] = module.mm_weight.input_global_scale
            state_dict[f"{name}.weight_global_scale"] = module.mm_weight.weight_global_scale
            
            if module.mm_weight.bias is not None:
                state_dict[f"{name}.bias"] = module.mm_weight.bias
    
    # 使用 safetensors 保存
    from safetensors.torch import save_file
    save_file(state_dict, output_path)
```

## 0x4.5 性能优化技巧

### CPU Offload 支持

LightX2V 支持将量化权重放在 CPU 上,推理时动态加载到 GPU:

```python
# 创建 CPU pin memory buffer
mm_weight = MMWeightNvfp4(
    weight_name="...",
    bias_name="...",
    create_cpu_buffer=True,  # 使用 CPU pin memory
    lazy_load=True,
)

# 推理时异步加载到 GPU
mm_weight.to_cuda(non_blocking=True)
output = mm_weight.apply(input_tensor)

# 推理完成后卸载回 CPU
mm_weight.to_cpu(non_blocking=True)
```

### Lazy Load 支持

对于大模型,可以使用 lazy load 按需加载权重:

```python
mm_weight = MMWeightNvfp4(
    weight_name="transformer.blocks.0.attn.qkv.weight",
    bias_name="transformer.blocks.0.attn.qkv.bias",
    lazy_load=True,
    lazy_load_file="/path/to/model/block_0.safetensors",
)

# 权重会在第一次使用时才从磁盘加载
```

### 多层权重管理

LightX2V 使用 `load_state_dict` 机制管理多层权重:

```python
# 预先创建 CUDA buffer
for layer_idx in range(num_layers):
    mm_weight = MMWeightNvfp4(
        weight_name=f"transformer.blocks.{layer_idx}.attn.qkv.weight",
        bias_name=f"transformer.blocks.{layer_idx}.attn.qkv.bias",
        create_cuda_buffer=True,  # 预分配 GPU 显存
    )

# 推理时动态加载不同层的权重
for layer_idx in range(num_layers):
    mm_weight.load_state_dict(weight_dict, layer_idx)
    output = mm_weight.apply(input_tensor)
```

## 0x4.6 实际应用场景

### 视频生成模型加速

LightX2V 项目主要用于视频生成模型(如 Wan2.2, HunyuanVideo)的推理加速:

```python
# 示例: Wan2.2 模型的 Transformer block
class TransformerBlock:
    def __init__(self):
        # QKV projection 使用 NVFP4 量化
        self.qkv = MMWeightNvfp4(
            weight_name="transformer.blocks.0.attn.qkv.weight",
            bias_name="transformer.blocks.0.attn.qkv.bias",
        )
        
        # MLP 使用 NVFP4 量化
        self.mlp_fc1 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc1.weight",
            bias_name="transformer.blocks.0.mlp.fc1.bias",
        )
        self.mlp_fc2 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc2.weight",
            bias_name="transformer.blocks.0.mlp.fc2.bias",
        )
    
    def forward(self, x):
        # 1. QKV projection (量化加速)
        qkv = self.qkv.apply(x)  # (B, L, 3*D)
        
        # 2. Attention (FP16/BF16)
        attn_out = self.attention(qkv)
        
        # 3. MLP (量化加速)
        mlp_out = self.mlp_fc2.apply(
            F.gelu(self.mlp_fc1.apply(attn_out))
        )
        
        return mlp_out
```

### 性能提升

在 Blackwell GPU (如 B200) 上,使用 NVFP4 量化可以获得:
- **2-3x** 的推理加速(相比 BF16)
- **4x** 的显存节省(权重从 BF16 压缩到 FP4)
- **接近 BF16** 的精度(通过 per-group 量化和校准)

### 适用模型

LightX2V 的量化方案特别适合:
- **DiT (Diffusion Transformer)** 模型: Wan2.2, HunyuanVideo
- **大规模 Transformer**: 参数量 > 10B
- **推理密集型应用**: 视频生成、图像生成

---

# 0x5. 总结

LightX2V kernel 是一个高性能的低精度量化 GEMM 库,主要特点:

**核心特性**: 支持 NVFP4 和 MXFP4/6/8 多种格式; 充分利用 Blackwell 架构的 Block Scaled Tensor Core 和 PTX 指令; Python 接口简洁易用; 通过向量化、warp 规约、epilogue fusion 等技术优化性能。

**适用场景**: 大模型推理加速; 视频生成模型(如 LightX2V); 需要极致性能的低精度计算场景。

**注意事项**: 需要 Blackwell 架构(SM120)支持; 输入维度需要满足对齐要求; 需要校准数据集来确定合适的 global_scale。