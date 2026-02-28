# 0x0. 前言

在[【翻译】在 GPU 上如何加速 GPTQ Triton 反量化kernel](https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg) 中 PyTorch 官方给出了一系列优化 GPTQ INT4 反量化 Triton Kernels 的方法，如 L2 Cache（Block swizzled)，向量化读取，SplitK优化改善Warp Stalling，这里单独再利用目前最先进的 o1-preview 模型解析下这个Triton代码实现，看一下目前最先进的模型在阅读 Triton kernel 上面的表现。

# 0x1. 前置知识

除了上面的 Blog 之外，我之前学习 Triton MatMul 教程的时候也学习了这里提到的 L2 Cache 优化，并把这些知识也记录到了[【BBuf的CUDA笔记】十三，OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ) 这篇文章中，想深入了解这个Block swizzled是如何影响矩阵乘法可以阅读。

在[CUDA-MODE课程笔记 第7课: Quantization Cuda vs Triton](https://mp.weixin.qq.com/s/1gCgpp49NF7sDw__EpO-nw) 这篇 CUDA-MODE 学习笔记中，PyTorch的Core Developer较为全面的讲解了大模型量化的方法以及对比了用CUDA/Triton来开发这些流行的量化kernel的优缺点。对LLM量化背景有所欠缺的读者也可以选择读一下这篇笔记。

# 0x2. 代码链接

Triton 实现的GPTQ INT4 反量化 Kernel 代码链接：https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py


# 0x3. 基于 o1-preview-128k 的代码解析

尝试用最先进的 o1-preview-128k 模型来解析一下这个 Triton 代码，看下它能不能指出在 [【翻译】在 GPU 上如何加速 GPTQ Triton 反量化kernel](https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg)  提到的三个关键优化点。

prompt为：
```shell
请详细解析一下下面的代码实现，每一行都需要加注释，用中文回复。+code...
```

粘贴一下o1-preview-128k给出的回复：

```python
# 导入 PyTorch 库
import torch
# 导入 Triton 库
import triton
# 从 Triton 中导入语言模块，简写为 tl
from triton import language as tl

# 使用 @triton.jit 装饰器，将函数编译为 Triton 内核
@triton.jit()
def swizzle_tile(pid,
                 m, n,
                 block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    # 计算 M 方向上的网格大小，向上取整
    grid_m = tl.cdiv(m, block_m)
    # 计算 N 方向上的网格大小，向上取整
    grid_n = tl.cdiv(n, block_n)

    # 计算每个组的宽度，即组内程序数
    width = group_m * grid_n
    # 计算当前组的 ID，pid 为程序的全局 ID
    group_id = pid // width
    # 计算当前组的大小，防止越界
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    # 计算当前程序在 M 维度上的块索引 pid_m
    pid_m = group_id * group_m + (pid % group_size)
    # 计算当前程序在 N 维度上的块索引 pid_n
    pid_n = (pid % width) // group_size

    # 返回计算的 pid_m 和 pid_n，用于后续计算
    return pid_m, pid_n

# 定义矩阵乘法的 Triton 内核，支持 K 维度的分块计算（Split-K）
@triton.jit()
def matmul_split_k_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          stride_scales_g, stride_scales_n,
                          stride_zeros_g, stride_zeros_n,
                          groupsize,
                          m, n, k,
                          block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
                          group_m: tl.constexpr, split_k: tl.constexpr):
    # 获取当前程序的 ID，在第一个维度（M*N 维度）
    pid = tl.program_id(0)
    # 获取在 K 维度上的程序 ID
    pid_k = tl.program_id(1)
    # 计算 K 维度上总的块数，向上取整
    total_blocks_k = tl.cdiv(k, block_k * split_k)

    # 使用自定义的 swizzle_tile 函数计算当前程序对应的块索引
    pid_m, pid_n = swizzle_tile(pid,
                                m, n,
                                block_m, block_n, group_m)

    # 计算当前程序在 M、N、K 维度上的元素偏移
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    # 确保 offs_m 和 offs_n 的连续性和对齐
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    # 计算矩阵 A 和矩阵 B 中当前块的指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    # 计算 scales 和 zeros 的指针
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    # 计算移位量，用于从压缩的表示中提取实际的值
    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    # 初始化累加器为 0，形状为 (block_m, block_n)，数据类型为 float32
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    # 遍历 K 维度上的所有块
    for k in range(0, total_blocks_k):
        # 从全局内存中加载矩阵 A 和矩阵 B 的当前块
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # 计算当前分组的 ID，用于获取对应的 scales 和 zeros
        g_id = (k * split_k + pid_k) // (groupsize // block_k)

        # 加载对应的 scales
        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        # 加载对应的 zeros
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        # 处理 zeros，提取实际的偏移值
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        # 处理矩阵 B，提取实际的值并进行反量化
        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        # 进行矩阵乘法并累加结果
        acc += tl.dot(a, b)
        # 更新指针，移动到下一块数据
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    # 将累加器的数据类型转换为 float16
    acc = acc.to(tl.float16)

    # 重新计算输出矩阵 C 的元素偏移
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    # 计算输出矩阵 C 的指针位置
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # 使用原子加法将计算结果写入全局内存中的矩阵 C
    tl.atomic_add(c_ptrs, acc, sem='release')

# 定义矩阵乘法的封装函数，设置相关参数并调用内核
def matmul_split_k(a, b, scales, zeros):
    # 获取矩阵 A 的形状，m 行 k 列
    m, k = a.shape
    # 获取矩阵 B 的形状，_ 行 n 列
    _, n = b.shape

    # 量化的组大小
    quant_groupsize = 128
    # 定义块大小
    block_m = 16
    block_n = 32
    block_k = 128
    # 每组的 M 维度块数
    group_m = 8
    # 阶段数和线程束数，用于性能优化
    num_stages = 3
    num_warps = 4
    # K 维度上的分块数量
    split_k = 4

    # 计算总的块数，向上取整
    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    # 计算 M*N 维度上的总程序数量
    total_programs_mn = total_blocks_m * total_blocks_n
    # K 维度上的程序数量
    total_programs_k = split_k

    # 定义内核的网格大小
    grid = (total_programs_mn, total_programs_k)

    # 打印问题规模和块信息
    print(f"problem m size: {m}, tile size m: {block_m}, total blocks m: {total_blocks_m}")
    print(f"problem n size: {n}, tile size n: {block_n}, total blocks n: {total_blocks_n}")
    print(f"problem k size: {k}, tile size k: {block_k}, total thread blocks k: {split_k}")

    # 打印总的线程块数量
    print(f"total thread blocks k: {k}, total thread blocks m and total thread blocks n = {total_blocks_m=} x {total_blocks_n} = {total_programs_mn}")
    print(f"{total_programs_mn=}, {total_programs_k=}")

    # 初始化输出矩阵 C，数据类型为 float16
    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    # 调用 Triton 内核函数，传入参数和内核执行配置
    k = matmul_split_k_kernel[grid](a, b, c, scales, zeros,
                                    a.stride(0), a.stride(1),
                                    b.stride(0), b.stride(1),
                                    c.stride(0), c.stride(1),
                                    scales.stride(0), scales.stride(1),
                                    zeros.stride(0), zeros.stride(1),
                                    quant_groupsize,
                                    m, n, k,
                                    block_m, block_n, block_k,
                                    group_m, split_k, num_stages=num_stages, num_warps=num_warps)

    # 打印内核的性能信息，包括寄存器使用、溢出和共享内存使用
    print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    # 将内核的汇编代码等信息写入文本文件，便于调试和分析
    with open('matmul_split_k.txt', 'w') as f:
        print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        print("IR", k.asm['ttir'], file=f)
        print("TTGIR", k.asm['ttgir'], file=f)
        print("PTX", k.asm['ptx'], file=f)
        print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)

    # 返回计算结果矩阵 C
    return c

# 定义辅助函数，用于创建指定形状和数据类型的张量
def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # 如果数据类型是 int32，生成随机整数填充张量
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # 否则，创建张量并使用正态分布随机数填充
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    # 返回生成的张量
    return res

# 主函数，程序入口
if __name__ == '__main__':
    # 定义矩阵的尺寸
    m = 16
    k = 4096
    n = 4096
    # 定义量化组的大小
    groupsize = 128
    # 计算量化组的数量
    g = k // groupsize

    # 创建矩阵 A，形状为 (m, k)，数据类型为 float16
    a = make_tensor(m, k, dtype=torch.float16)
    # 创建矩阵 B，形状为 (k//8, n)，数据类型为 int32
    b = make_tensor(k // 8, n, dtype=torch.int32)
    # 创建矩阵 C，形状为 (m, n)，数据类型为 float16
    c = make_tensor(m, n, dtype=torch.float16)
    # 创建 zeros 和 scales，用于量化和反量化
    zeros = make_tensor(g, n // 8, torch.int32)
    scales = make_tensor(g, n, torch.float16)

    # 调用矩阵乘法函数，计算结果
    split_k_output = matmul_split_k(a, b, scales, zeros)
    # 打印计算结果的形状和部分数据
    print(f"{split_k_output.shape=}, {split_k_output[0][0:4]}")
```



代码的流程上挑不出太大的问题，不过对于swizzle_tile这个优化程度很高的技巧来说，AI还是没办法读懂的，这个函数的解释请忽略上面生成的注释转看：[【BBuf的CUDA笔记】十三，OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ) 。然后代码中看起来莫名其妙的是这几行，涉及到对INT32->INT4以及INT4->FP16的反量化：

```python
zeros = (zeros >> zeros_shifter) & 0xF
zeros = (zeros + 1) * scales

b = (b >> shifter[:, None]) & 0xF
b = b * scales[None, :] - zeros[None, :]
```

问一下 o1-preview-128k，如何理解这几行代码，并给我们一个详细的解释：

![](https://files.mdnice.com/user/59/5093f5cb-d803-43f5-bb63-b534636ded63.png)
![](https://files.mdnice.com/user/59/eaa3d918-5b7f-42e6-ba66-62e6463f1f7c.png)
![](https://files.mdnice.com/user/59/9d8e4a0a-b37f-490f-8b0a-37451b6fd46b.png)
![](https://files.mdnice.com/user/59/f93105b8-0c93-412f-9591-74fcf3862e47.png)
![](https://files.mdnice.com/user/59/8441928f-de7f-4f9c-a02a-29780099d0d2.png)
![](https://files.mdnice.com/user/59/a5b7b4ea-9c96-46f1-9275-569e5e2153d5.png)
![](https://files.mdnice.com/user/59/95e86461-a81a-47ee-9ac1-e70d3ae949a9.png)
![](https://files.mdnice.com/user/59/ffebd57c-f7ee-4801-8445-73af554f3208.png)
![](https://files.mdnice.com/user/59/faee62ba-ccc7-44b1-a13b-34b53c41567e.png)

o1-preview-128k 对这几行代码完全理解了，并且可以正确还原背后的数学原理，非常棒。

接下来我们看一下向量化读取的优化是否可以被 o1-preview-128k 正确理解:

![](https://files.mdnice.com/user/59/606a148a-9c0e-41c5-9f45-6271627211bb.png)
![](https://files.mdnice.com/user/59/152bb066-e54e-47e3-aefe-b3c809577a54.png)
![](https://files.mdnice.com/user/59/5e43c64e-b1b9-4160-afe9-d0d7967731e8.png)
![](https://files.mdnice.com/user/59/fe8f9dc9-0b33-4d49-9ab3-c5ffb096affd.png)
![](https://files.mdnice.com/user/59/6ac94494-3ebc-40cb-9332-9f4ba5478711.png)
![](https://files.mdnice.com/user/59/956c533f-00e4-4b7b-9532-e342cb5be018.png)
![](https://files.mdnice.com/user/59/874a0e55-4988-4a59-9a0a-d836c97a495c.png)

o1-preview-128k 完全理解这个优化，并且还给我们举例，画图来说明向量化读取的原理，并指出可以简化地址计算等。

# 0x4. 总结

从上面看，在L2 Cache，向量化读取，SplitK方面 o1-preview-128k 模型都可以理解这些优化的作用。需要说明的是，L2 Cache优化方面，o1-preview-128k 模型给出的解释并不能说明他彻底理解了这个Block swizzle的原理，这个优化我们还是需要查看Triton文档或者[【BBuf的CUDA笔记】十三，OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ) 来理解。总的来说，我们可以用大模型来帮助我们更好的阅读代码和探究背后的原理，这确实算得上是生产力革命，最近Cursor的大火也说明了这一点。不过我们仍然需要最先进的大模型来让我们获得最好的代码阅读体验，特别是在专业领域的代码上，读者感兴趣也可以尝试下其它大模型对上面的代码的解释。







