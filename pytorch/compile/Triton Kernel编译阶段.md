> 博客来源：https://pytorch.org/blog/triton-kernel-compilation-stages/ 。这里做个翻译，是[CUDA-MODE 课程笔记 第29课 Triton内部机制](https://mp.weixin.qq.com/s/7tfTXaG7D208l_5DzN9hBw)的补充。

# Triton Kernel Compilation Stages

> by Sara Kokkila-Schumacher*, Brian Vaughan*, Raghu Ganti*, and Less Wright+ (*IBM Research, +Meta) 

Triton 是一个开源编程语言和编译器，提供了一种基于 Python 的高效 GPU 代码创建方法。在这篇博客中，我们强调了 Triton 程序的编译过程和中间表示。有关 Triton 的介绍，请参阅这篇博客(https://openai.com/index/triton/)。

## Triton 语言和编译

Triton 编程语言支持不同类型的现代 GPU，并遵循块式编程方法。作为示例，我们将按照 Triton 向量加法教程进行一些小的修改。向量加法kernel和辅助函数定义如下:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements, 
               BLOCK_SIZE: tl.constexpr, 
               ):
  
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
 
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
 
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    triton_kernel=add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    # Save compilation stages - some of the stages identified here are specific to NVIDIA devices:
    with open('triton_IR.txt', 'w') as f:
        print(triton_kernel.asm['ttir'], file=f)
    with open('triton_TTGIR.txt', 'w') as f:
        print(triton_kernel.asm['ttgir'], file=f)
    with open('triton_LLVMIR.txt', 'w') as f:
        print(triton_kernel.asm['llir'], file=f)
    with open('triton_PTX.ptx', 'w') as f:
        print(triton_kernel.asm['ptx'], file=f)
    with open('triton_cubin.txt', 'w') as f:
        print(triton_kernel.asm['cubin'], file=f)

    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

Triton 向量加法kernel包含 `@triton.jit` 装饰器。Triton 编译器会编译被 `@triton.jit` 标记的函数,通过多个编译阶段将函数降级。辅助函数 `add` 分配输出张量,计算适当的 GPU 网格大小,并额外保存中间编译阶段。

聚焦于编译过程,Triton 内核通过以下图中所示的一系列阶段被降级为设备特定的汇编代码。

![](https://files.mdnice.com/user/59/44c05c30-e5f1-4684-8289-76f96c5674c9.png)

内核编译首先通过遍历被装饰的Python函数的抽象语法树(AST)来创建Triton中间表示(Triton-IR)。Triton-IR是一个未优化的、与机器无关的中间表示。它引入了块级编程要求,并基于开源LLVM编译器项目。接下来,Triton编译器优化并将Triton-IR转换为Triton-GPU IR(Triton-TTGIR)阶段,然后转换为LLVM-IR。Triton-IR和Triton-GPUIR表示都是以MLIR Dialect的形式编写的,其中MLIR是LLVM的一个子项目,旨在改进异构硬件的编译。

对于Triton向量加法教程kernel,示例Triton IR片段如下:

```shell
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/u/saraks/triton_blog/01-vector-add.py":28:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/u/saraks/triton_blog/01-vector-add.py":28:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/u/saraks/triton_blog/01-vector-add.py":28:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/u/saraks/triton_blog/01-vector-add.py":28:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc5)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc7)
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc8)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc9)
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc10)
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc11)
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
```

注意到 Triton kernel 中的主要函数现在表示为:

![](https://files.mdnice.com/user/59/091a0ac6-7215-4b87-b864-4cd3a86735cc.png)

在 Triton IR 阶段，`%arg0: !tt.ptr&lt;f32>` 和后续的张量引用表明中间表示已经按数据类型进行了专门化。

我们在一台配备 Tesla V100-SXM2-32GB GPU、CUDA 12.2 版本、Python 3.11.9 和 PyTorch 2.4.1(使用 PyTorch 默认安装的 Triton 版本)的机器上运行了这个示例。在这个设备上，这个简单的向量加法有以下 Triton GPU IR 片段(为了清晰起见省略了一些行):

```shell
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:70", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}
    ⋮
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc8)
    ⋮
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc10)
    %13 = arith.addf %9, %12 : tensor<1024xf32, #blocked> loc(#loc11)
    ⋮
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc13)
    ⋮
  } loc(#loc)
} loc(#loc)
```

在这个阶段，一些硬件特定的信息被包含在内。例如，计算能力与如何将张量分布到core和warps或AMD GPU的wavefronts有关。在这个例子中，张量表示为`#blocked`布局。在这种编码中，每个波前拥有张量的连续部分。目前，其他可能的内存优化包括`slice`(沿着一个维度重新构造和分布张量)、`dot_op`(优化块矩阵乘积的布局)、`shared`(表示GPU共享内存)、`nvidia_mma`(由NVIDIA Tensor Cores生成)、`amd_mfma`(由AMD MFMA矩阵核心生成)和`amd_wmma`(由AMD WMMA矩阵核心生成)。在最近的Triton会议上宣布，这种布局表示将过渡到一个新的线性布局，以统一内部和跨后端的布局。从Triton-GPUIR到LLVM-IR的阶段将Triton-GPUIR转换为LLVM的表示。此时，Triton有NVIDIA和AMD设备的第三方后端支持，但其他设备支持正在由开源社区积极开发。

一个 LLVM-IR 向量加法参数的子集示例如下:

```shell
  %19 = extractvalue { i32, i32, i32, i32 } %18, 0, !dbg !16
  %39 = extractvalue { i32, i32, i32, i32 } %38, 0, !dbg !18
  %23 = bitcast i32 %19 to float, !dbg !16
  %43 = bitcast i32 %39 to float, !dbg !18
  %56 = fadd float %23, %43, !dbg !19
```

在进行一些指针算术运算和内联汇编调用从全局内存中检索数据后,向量元素被提取并转换为正确的类型。最后它们被加在一起,并通过内联汇编表达式写回全局内存。

Triton编译过程的最后阶段将LLVM-IR降级为设备特定的二进制文件。对于向量加法示例,在NVIDIA GPU上,下一个中间表示是PTX(并行线程执行)。低级PTX语法指定了NVIDIA设备在线程级别的执行,这从CUDA 1.0版本就开始了。有关PTX的深入指南,请参阅NVIDIA的文档。在向量加法中,kernel参数从主机传递到kernel,分配地址,`mov`指令促进线程级数据访问,最终使用`add.f32`表示元素加法调用,如下例所示:

```shell
add.f32 	%f17, %f1, %f9// add type float32, output register, input register for x, input register for y
```

Triton编译器通过不同的硬件后端管理汇编代码的编译,以生成二进制文件。Triton kernel现在可以使用了。


## 总结

Triton提供了一个高级抽象来为不同类型的硬件编写和编译kernel。在这篇博客中，我们强调了Triton代码表示和Triton编译器的不同阶段。有关包括自定义Triton kernel或使用Triton kernel加速不同工作负载的详细信息，请查看PyTorch Triton教程(https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)、Triton GPTQ kernel博客(https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg)、使用Triton进行Llama3 FP8推理(https://mp.weixin.qq.com/s/v6Ah4uFtI2zTgiAZ3-mKvw)、LLMs的CUDA-Free推理(https://mp.weixin.qq.com/s/KlxBzBNxyRBnoEr8qXjgeg)，或PyTorch 2.2节(https://pytorch.org/assets/pytorch2-2.pdf)中关于Triton代码生成的内容。



