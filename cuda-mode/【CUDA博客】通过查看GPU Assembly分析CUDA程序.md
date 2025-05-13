> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/analyze-cuda-programs-by-looking-at-gpu-assembly/

# 通过查看GPU Assembly分析CUDA程序

2025年4月21日

这是一篇关于如何通过分析SASS代码来提高内存受限CUDA程序性能的简短笔记。

## 简单的向量复制

让我们考虑以下两个用于复制向量的程序。

```c++
#define threadsPerBlock 1024

__global__ void vectorCopy(float *input, float *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < N) {
    output[i] = input[i];
  }
}

__global__ void vectorCopyVectorized(float4 *input, float4 *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < (N >> 2)) {
    output[i] = input[i];
  }
}
```

结果表明，当 $N = 1 << 30$ 时，向量化版本的性能为 $\frac{3.94361ms-2.802ms}{2.802ms} = 41\%$ 更快。

## 分析

为了理解其原因，查看每个kernel的SASS代码会很有帮助。我们可以通过使用godbolt(https://godbolt.org/)或使用NVIDIA NCU工具分析kernel来获取SASS代码。在godbolt上，我们应该选择适当的nvcc版本，并为编译器配置命令参数，如：-arch sm_90 -use_fast_math -O3。选择适当的架构很重要。

然后我们可以查看SASS代码。NVIDIA文档(https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm)将SASS描述为低级汇编语言，它编译为二进制微代码，可在NVIDIA GPU硬件上原生执行。

以下是H100上这两个kernel的SASS代码：

```shell
vectorCopy(float*, float*, int):

 LDC R1, c[0x0][0x28] 

 S2R R7, SR_TID.X 

 ULDC UR4, c[0x0][0x220] 

 S2R R0, SR_CTAID.X 

 LEA R7, R0, R7, 0xa 

 ISETP.GE.AND P0, PT, R7, UR4, PT 

 @P0 EXIT 

 LDC.64 R2, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R4, c[0x0][0x218] 

 IMAD.WIDE R2, R7, 0x4, R2 

 LDG.E R3, desc[UR4][R2.64] 

 IMAD.WIDE R4, R7, 0x4, R4 

 STG.E desc[UR4][R4.64], R3 

 EXIT 

```

```shell
vectorCopyVectorized(float4*, float4*, int):

 LDC R1, c[0x0][0x28] 

 S2R R7, SR_TID.X 

 ULDC UR4, c[0x0][0x220] 

 USHF.R.S32.HI UR4, URZ, 0x2, UR4 

 S2R R0, SR_CTAID.X 

 LEA R7, R0, R7, 0xa 

 ISETP.GE.AND P0, PT, R7, UR4, PT 

 @P0 EXIT 

 LDC.64 R4, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R2, c[0x0][0x218] 

 IMAD.WIDE R4, R7, 0x10, R4 

 LDG.E.128 R8, desc[UR4][R4.64] 

 IMAD.WIDE R2, R7, 0x10, R2 

 STG.E.128 desc[UR4][R2.64], R8 

 EXIT 
```

我们看到向量化版本多了一条指令，这是因为我们执行了位移来计算`N / 4 = N >> 2`。我们可以通过传递`N/4`给kernel来优化掉这条指令（`USHF.R.S32.HI UR4, URZ, 0x2, UR4`），这样就可以少一条指令，但这并没有太大区别。因此在接下来的分析中，我们忽略这个位移操作。

逻辑中有趣的部分（即我们执行复制的地方）是：

```shell
LDC.64 R2, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R4, c[0x0][0x218] 

 IMAD.WIDE R2, R7, 0x4, R2 

 LDG.E R3, desc[UR4][R2.64] 

 IMAD.WIDE R4, R7, 0x4, R4 

 STG.E desc[UR4][R4.64], R3
```

与向量化版本：

```shell
LDC.64 R4, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R2, c[0x0][0x218] 

 IMAD.WIDE R4, R7, 0x10, R4 

 LDG.E.128 R8, desc[UR4][R4.64] 

 IMAD.WIDE R2, R7, 0x10, R2 

 STG.E.128 desc[UR4][R2.64], R8 
```

我们看到，通过使用`LDG.E.128/STG.E.128`而不是`LDG.E/STG.E`，我们加载/存储128位而非32位！这意味着我们需要相同数量的指令，但需要的块数要少得多！

为了理解这一点，让我们比较：

```c++
template <int threadsPerBlock>
__global__ void vectorCopy(float *input, float *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < N) {
    output[i] = input[i];
  }
}

template <int threadsPerBlock>
void launchVectorCopy(float *input, float *output, int N) {
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorCopy<threadsPerBlock>
      <<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
```

与：

```c++
template <int threadsPerBlock>
__global__ void vectorCopyVectorized(float4 *input, float4 *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < (N >> 2)) {
    output[i] = input[i];
  }
}

template <int threadsPerBlock>
void launchVectorCopyVectorized(float *input, float *output, int N) {
  const int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
  vectorCopyVectorized<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
      reinterpret_cast<float4 *>(input), reinterpret_cast<float4 *>(output), N);
}

```

如果我们取`N = 1 << 30`和`threadsPerBlock = 1 << 10`，在第一个版本中我们启动了`1048576`个块，而在第二个版本中我们启动了`262144`个块。如果我们忽略位移指令的成本（或者如上所述简单地消除它），我们就能理解为什么第二个kernel快得多：我们执行的指令少得多，因为我们只启动了非向量化版本中加载块数的一小部分。

我希望这篇博文能帮助你更好地理解向量化的加载和存储。代码可以在我的github上找到(https://github.com/simveit/vector_copy_vectorized)。makefile还包括在NVIDIA Nsight Compute中使用的分析kernel的命令。如果你喜欢这篇博文，你可以在Linkedin上与我联系。我喜欢交流关于CUDA和一般MLSys的想法。



