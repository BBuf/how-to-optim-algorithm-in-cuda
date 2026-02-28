> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/indexing-in-cuda/

# CUDA中的索引

## 介绍

在这篇博文中,我想解释在CUDA中行优先格式的矩阵是什么意思。这对于理解CUDA kernel及其对所处理矩阵的索引方法至关重要。

让我们考虑一个形状为`(M, N)`的2D数组`A`。在CUDA中,这样的数组默认以行优先格式线性化,以适应计算机中内存空间的扁平结构。实际上,这意味着矩阵坐标`(i,j)`被映射到`i * N + j`。让我们称这个函数为`f`。

看这个公式我们就能明白为什么这被称为行优先。让我们看看两个不同坐标的内存映射的差值:

```shell
d = f(i2, j2)-f(i1,j1) = (i2-i1) * N + (j2-j1)
```

我们可以看到,对于相邻的列`d = 1`,对于相邻的行`d = N`。

我们可以进一步将其推广到形状为`(M1, M2, M3)`的3D数组。这里坐标`(i, j, l)`被映射到`l + M3 * (j + i * M2)= i * M2 * M3 + j * M3 + l`。

## 代码分析

现在让我们用这个范式来理解CUDA kernel的索引。关于2D Block Tiling(https://siboehm.com/articles/22/CUDA-MMM)的完整解释,我推荐这篇优秀的博文,它详细解释了2D Tiling的概念。这里我们将专注于索引部分。完整代码(https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh)可以在github上找到,在继续之前你应该阅读并尝试理解它。

第一步我们分配共享内存。每个矩阵可以被解释为一个2D矩阵,`As`的形状为`(BM, BK)`,`Bs`的形状为`(BK, BN)`。

```c++
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];
```

这个共享内存随后由全局内存中的相应元素填充。

我们要仔细分析的代码是以下部分:

```c++
// calculate per-thread results
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // block into registers
  for (uint i = 0; i < TM; ++i) {
    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
  }
  for (uint i = 0; i < TN; ++i) {
    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
  }
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      threadResults[resIdxM * TN + resIdxN] +=
          regM[resIdxM] * regN[resIdxN];
    }
  }
```

当我第一次看这些索引时,我不确定如何自己推导它们,但实际上使用上面的解释并不太难。

### 分析As的索引

```c++
(threadRow * TM + i) * BK + dotIdx
= threadRow * TM * BK + i * BK + dotIdx
```

从这里我们可以看出`As`被解释为形状为`(..., BK, TM)`的3D数组。从上面我们知道原始矩阵的形状是`(BM, BK)`,因为矩阵仍然是相同的,这意味着我们作为3D数组的解释具有形状`(BM/TM, TM, BK)`,因为数组中的元素数量保持不变。

从这里可以清楚地看出为什么我们这样索引:我们想要进一步对内存块进行Tiling,并通过将`(BM, BK)`转换为`(BM/TM, TM, BK)`来精确地实现这一点。

`threadRow * TM * BK + i * BK + dotIdx -> (threadRow, i, dotIdx)`。

- `const int threadRow = threadIdx.x / (BN / TN);`即它对应于一个线程束。一个线程束处理一个BM/TM块。
- `i`: 这对应于寄存器数组的索引
- `dotIdx`: 这对应于原始2D数组的列索引

### 分析Bs的索引

`dotIdx * BN + threadCol * TN + i = dotIdx * BN/TN * TN + threadCol * TN + i`。使用与上面类似的技术,我们看到我们将`Bs`解释为形状为`(BK, BN/TN, TN)`的3D数组。`dotIdx * BN/TN * TN + threadCol * TN + i -> (dotIdx, threadCol, i)`

- `const int threadCol = threadIdx.x % (BN / TN)`; 即它对应于一个线程。在这种情况下,线程是处理一列的线程束中的一个元素。
- `i`: 这对应于寄存器数组的索引
- `dotIdx`: 这对应于原始2D数组的列索引

### 整体理解

我们现在能够理解完整的循环: `float threadResults[TM * TN] = {0.0}`; 所以结果最初是一个`(TM, TN)`的2D数组。

```c++
threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
```

算法工作如下:

- 我们要将形状为`(BM, BK)`和`(BK, BN)`的两个矩阵`As`和`Bs`相乘
- 我们将`As`视为`(BM/TM, BK, TM)`,将`Bs`视为`(BK, BN/TN, TN)`
- `As`的第一维和`Bs`的第二维由一个线程束处理
- 我们在`BK`维度上循环`dotIdx`
- 我们循环`TM`并将`TM`元素写入寄存器,每个元素对应一个线程束和一个`dotIdx`
- 我们循环`TN`并将`TN`元素写入寄存器,每个元素对应一个线程束和一个`dotIdx`
- 我们通过乘以寄存器中的元素,将所有`k`的结果累积到大小为`(TM, TN)`的2D数组中

执行此操作后,结果被写入结果矩阵,如下所示:

```c++
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
  for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
    C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
        alpha * threadResults[resIdxM * TN + resIdxN] +
        beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
  }
}
```

`(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN = (threadRow * TM + resIdxM) * N/TN * TN + threadCol * TN + resIdxN`。`C`最初被认为是形状为`(M, N)`的矩阵,现在被解释为形状为`(M/TM, TM, N/TN, TN)`的矩阵,这意味着我们也以Tiling方式写回内存。

`(threadRow * TM + resIdxM) * N/TN * TN + threadCol * TN + resIdxN -> (threadRow, resIdxM, threadCol, resIdxN)`。所以每个线程束写入一行和一列。

我希望这篇博文能帮助你更好地理解CUDA中的索引。













