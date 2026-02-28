来自：https://zhuanlan.zhihu.com/p/20579515046 & https://zhuanlan.zhihu.com/p/21142007017

![](https://files.mdnice.com/user/59/17e8530e-73ee-4c95-9048-3a154ead44e8.png)

# 实用 Swizzle 教程（一）

本文实验仓库地址：Chtholly-Boss/swizzle: A practical way of learning Swizzle

## 前言

最近科研工作中需要使用 Tensor Core 进行算子优化，在优化过程中发现了大量的 Bank Conflict
，感到十分苦恼。听说 CUTLASS

提出了一种 Bank Conflict Free 的 Swizzle 技术，感觉很高大上，遂进行了高强度的 RTFM 和博客阅读。 Zhihu 上对于 Swizzle 的讲解已有了不少的优秀的文章，笔者看完后感觉十分长脑子，但在落地的时候我的手跟我说他还没学会，他对以下几点感到十分苦恼：

    他不想使用 CUTLASS 库，因为还不会
    他不知道从哪里写起，大部分博客没代码抄，或者有代码，但调的 CUTLASS

笔者对他感到十分失望，遂经过两三天的实验后，为他写下了本文，试图教会他 **怎么将 Swizzle 技术应用到算子中以消除 Bank Conflict**

## 问题的产生

Swizzle 能解决 Bank Conflict，但哪里来的 Conflict 呢？这就要从我们最开始的目标说起，即 “使用 Tensor Core 进行算子优化”。具体来说，在查阅 CUDA Programming Guide 7.24(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions) 后我们知道可以通过如下的方式调用 Tensor Core API 完成一次 `(m,n,k) = (16,16,16)` 的 `C = A B^T FP16` 矩阵乘法：

```c++
__device__ void mma_simple(half *a, half *b, half *c) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);

    fill_fragment(c_frag, 0.0f);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(c, c_frag, 16, mem_row_major);
}
```

关于函数的具体使用说明可直接查看 CUDA Programming Guide 7.24(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)。

这看起来似乎很友好，于是我兴奋地写好了测试代码后一顿操作猛如虎进行了 profile，得到了如下结果：

![bank conflict profile (ld st)](https://files.mdnice.com/user/59/9af2664d-c94b-4341-a0a1-0fa22d086b65.png)

Emm..., 如图所示，Global Load/Store 均不造成 bank conflict，那么这里的 12 次冲突是哪来的呢？我们在调用 `mma_simple` 的时候一般是从 Shared Memory 加载/存储数据（笔者的测试也是如此），因此合理推断是 mma 的相关操作产生了冲突。

通过反汇编查看 SASS 代码，我们可以知道 `load_matrix_sync` 生成了 `LDSM` 指令，`store_matrix_sync` 生成了 `STS` 指令，由此我们可以查看对应指令造成的冲突。

![bank conflict profile (LDSM STS)](https://files.mdnice.com/user/59/fb61e4b5-01d2-4ff0-aeb1-655b1343d444.png)

很好，我们成功地找到了罪魁祸首。因此现在的问题转为如下两个：

- `load_matrix_sync` 是怎么使用 `LDSM` 的？
- `store_matrix_sync` 写入的模式是怎么样的？

## LDSM 指令

为解决前述问题，在一顿对 AI 的拷打以及 RTFM 之后，我们知道与 `LDSM` 相对应的 PTX 指令有 `ldmatrix`，查阅 PTX ISA 9.7.14(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions) 可以了解到指令的相关信息，对于我们的问题来说，只需要关注 `ldmatrix.sync.aligned.x4.m8n8.shared.b16{.trans} r, [p];` 的用法即可。

该条指令的基础版本为 `ldmatrix.sync.aligned.x1.m8n8.shared.b16` ，从指令名可以看出其作用为加载一个 8x8 矩阵。这时候就有读者要问了：

- 为什么需要这条加载指令？有普通的 Load 指令不就够了吗？

事实上，从该指令在手册中所处的位置也可以知道，`ldmatrix` 是为 Tensor Core 的矩阵计算而生的一条加载指令，为使用 HMMA 指令利用 Tensor Core 进行矩阵乘法计算，矩阵的元素需要 **分布存储** 到一个 warp 的 32 个线程中。以一个 16x16 FP16 矩阵为例，由于每个 32 位寄存器可以存两个 FP16，每个线程各提供 4 个寄存器 R0, R1, R2, R3 ，共同存储 32 * 4 * 2 = 256 个矩阵元素。 为利用 HMMA 指令进行计算，在官方手册中要求矩阵的元素与线程的各个寄存器有一定的对应关系，以矩阵 A 为例，该对应关系如下所示：

![m16n8k16 fragment A layout ](https://files.mdnice.com/user/59/a68b140c-8670-410d-9e32-0d90f74898db.png)

其中 `{a0,a1}` 表示寄存器中存储原矩阵的 `a0, a1` 元素。

由该图可知，如果我们要用正常的 Load 指令将数据加载到寄存器，需要 4 条指令，但如果使用 `ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%addr];` 则只需一条指令，很大程度上减少了 Issue 的指令数。

`ldmatrix` 的 `.x{1,2,4}` 修饰符用于指定加载多少个 8x8 矩阵，每个矩阵的 8 个行的地址用对应线程给出，如 `thread0-7` 给出的 8 个地址用于加载第一个 8x8 矩阵到 warp 内各线程的 `R0` 寄存器中，`thread8-15` 给出的 8 个地址用于加载第二个 8x8 矩阵到 warp 内各线程的 `R1` 寄存器中... 该过程在实际运行中应当是分 4 阶段进行的。

`load_matrix_sync(frag_a, smem_a, 16)` 对该指令的使用方式可用下图说明：

![ldmatrix.sync.aligned.x4.shared.b16](https://files.mdnice.com/user/59/14e45ac5-e50d-47d4-b2f1-8195d1c24cb5.png)

为方便后续调用，我们将该指令进行封装如下：

```c++
#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))
__device__ __forceinline__ void ldmatrix_sync(fp16 *dst, void *addr) {
 asm volatile(
 "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
 : "=r"(REG(dst[0])),
 "=r"(REG(dst[2])),
 "=r"(REG(dst[4])),
 "=r"(REG(dst[6]))
 : "l"(__cvta_generic_to_shared(addr)));
}

// 每个32位寄存器可以存储2个FP16（半精度浮点数）值
// 在ldmatrix指令中，.x4表示要加载4个8x8的矩阵
// 这些索引的选择是为了确保数据被正确加载到对应的寄存器位置
// 具体来说：
// dst[0]和dst[1]会被加载到第一个寄存器（R0）
// dst[2]和dst[3]会被加载到第二个寄存器（R1）
// dst[4]和dst[5]会被加载到第三个寄存器（R2）
// dst[6]和dst[7]会被加载到第四个寄存器（R3）
// 为什么选择0,2,4,6这样的间隔？这是因为：
// 每个索引位置实际上包含两个FP16值（因为32位寄存器可以存两个FP16）
// 这种间隔选择确保了数据能够按照Tensor Core要求的特定布局被加载到寄存器中
// 这种布局对于后续的矩阵乘法运算（HMMA指令）是必需的
```



其中 dst 传入的参数即为含有 8 个 fp16 的 `fragment`

## STS 指令

STS 即 `st.shared`, 表示将寄存器的内容存到 shared memory 中。由于该指令处理的是结果矩阵，因此需要关注结果矩阵的布局。在 sm90 后结果矩阵的存储也可能使用 `stmatrix` 指令，由于思路都差不多，本文重点关注 Load，读者可根据需要决定是否对 Store 过程也进行优化。

## Bank Conflict

观察 `ldmatrix` 的第一个阶段，如下图所示：

![bank conflict in LDSM](https://files.mdnice.com/user/59/4d5a1171-dd50-4cf0-b328-2292b5d3e747.png)

由于共享内存共有 32 个 4B 的 bank，我们可以知道 16x16 FP16 矩阵的 4 行刚好占满所有 bank，因此在第一个阶段的加载中，会有如上图所示的冲突。

每个 16x16 矩阵会有 4 次冲突，由于我们加载了 A 和 B 两个矩阵，因此 LDSM 的冲突共有 8 次，与实验结果相符。

更可怕的点在于全局矩阵的一行大小刚好占满所有 bank 或者为 128 B的整数倍，如 16x64 FP16 矩阵，在加载一个局部 16x16 矩阵时就会产生 7 * 4 = 28 次冲突，4 个 16x16 就是 112 次！由此可见，在不做任何处理的情况下使用 Tensor Core API 可能会导致访存上的性能损失。

## 解决方案

由前述可知，bank conflict 产生于 `ldmatrix` 指令给出的地址在共享内存中的冲突性。因此，我们可以很自然地朝该方向寻求解决方案。

## 地址重排法

最自然的想法就是对给出的地址进行重排，如下图所示：

![](https://files.mdnice.com/user/59/bbcf5446-7dfe-440d-9238-ed9ceae3d675.png)

需要注意的是，此时 `thread16-31` 的 `R0` 寄存器存储的应是原先的 `R2` 寄存器存储的内容，在后期我们需要对他们的寄存器的内容进行交换才能保证得到正确的结果。

然而，该方法是不可扩展的。以 16x64 FP16 矩阵为例，若一个局部的 16x16 矩阵每行均与上一行冲突，则不管怎样重排给出的地址都会有冲突产生。具体细节可参照下图：

![](https://files.mdnice.com/user/59/893f1ba4-0872-4e32-9427-bdcb2af3f155.png)

因此，在不改变 Shared Memory 布局的情况下，我们很难达到 bank conflict free 的目标。

## 布局重映射法

布局重映射法通过将一个 8x8 子块的每一行分布到不同的 bank 中以实现共享内存的无冲突访问。此处我们以对全局内存里一个 16x64 FP16 矩阵的第一个 16x16 矩阵块进行 `ldmatrix` 为例，整个过程如下图所示：

![shift version](https://files.mdnice.com/user/59/78d0ace3-032e-4c57-b892-db34cc4f5ab8.png)

另一种可行的方式如下图所示：

![xor version](https://files.mdnice.com/user/59/fb713f46-6623-4d08-96da-2037f3ba7e40.png)

上面两种方式不同的只是地址的映射方式，第二种为 CUTLASS 所采用的方式，即利用异或进行重映射。让我们更细致地考察这一过程：

![](https://files.mdnice.com/user/59/0dc6e037-eeb4-4917-a114-9914b1cca42a.png)

我们可以总结得到以下几点：

- 仅需改变列偏移以实现 bank 重映射
- 对于每一行，我们应映射到不同的列

由第二点不难发现，shared memory 的列偏移可以由**行偏移的若干位**与 **Global Memory 中的列偏移** 异或得到。实际操作可以不局限于异或，只要能映射到不同的 bank，对列偏移进行其他变换也是可以的。

对于 Global Memory 中的行/列，我们称为逻辑行/列，即我们在运算时是根据全局内存的 layout 进行的。对于 Shared Memory 中的行/列，我们暂称为物理行/列，即在实际存储中由于需要避免 bank conflict，物理行/列并不如我们通常的模型一般排布，直接索引可能会出现问题。

由逻辑行/列到物理行/列的映射在实际应用中可由如下代码实现：

```c++
// kernel launch: <<<1, dim3(32,4)>>>
// 16x64 A矩阵加载到 shared memory
half smem_a[16 * 64];
// 向量化加载 128bit 的地址
int tIdx = tx + ty * blockDim.x;
int gAddr = tIdx * 8;
int gRow = gAddr / 64;
int gCol = gAddr % 64;
int sCol = (gCol / 8) ^ (gRow & 0x7);
int sAddr = gRow * 64 + sCol * 8;
// ld_st_128bit(dst, src)
ld_st_128bit(smem_a + sAddr, a + gAddr);
```

然后将 `load_matrix_sync` 改写为：

```c++
int r_ = threadIdx.x % 16;
int c_ = (r_ & 0x7) ^ (2 * threadIdx.y + threadIdx.x / 16);

// 利用前面的封装
ldmatrix_sync(a_frag.x, smem_a + r_ * 16 + c_ * 8)
```

对于矩阵 B 的操作类似，经过 Swizzle 后应当只有 STS 的时候才会产生冲突。现在就让我们揭晓最终的结果吧！

实际运行如下：

![](https://files.mdnice.com/user/59/52d53e80-74a6-443e-b1f9-49e0b819d92e.png)

Hoooooooray!!!

最后让我们聊一下其他情形，再取一个实际的例子，如用 `ldmatrix` 加载 16x32 矩阵。相信读到此处，读者们都能做出估计：不做 Swizzle 时，每两行填满所有 bank，因此共会有 3 * 4 * 2 = 24 次冲突。而我们的 Swizzle 应按如下图所示的思路进行：

![](https://files.mdnice.com/user/59/5a9492d5-fb16-4c37-9b94-536d5212c855.png)

由于相邻的两行不会冲突，因此行偏移的最低位不做任何贡献，利用行偏移的中间 2 位与原列偏移进行异或即可得到新的列偏移。

由以上过程不难知道，我们提取出了地址的三个部分用于 Swizzle：

- 块内字节偏移
- 列偏移
- 行偏移中用于异或列偏移的位

在 CUTLASS 中所用的三个参数含义类似。

## 结语

到目前为止，我们基本上 “从零开始” 自己开发出了 Swizzle 技术。当然，CUTLASS 对于 Swizzle 进行了更好的抽象，并且还提出了 Block Swizzle
用于调整 block issue 顺序以提高 Cache 命中率。本文章的目的仅止步于 shared memory bank conflict 问题，我的手也已经学会了，就让我们在这里结束吧。


# 实用 Swizzle 教程（二）

## 问题的抽象

在 实用 Swizzle 教程（一） 中，虽然我们已经了解到 swizzle 的基本思想，但在实现的过程中却采取了略微 dirty 的做法，即手动提取行/列偏移进行位运算。很自然地，我们便想要将其抽象为一个高效的函数以进行复用。

回顾上一篇文章给出的例子：

![基于 XOR 运算的 swizzle](https://files.mdnice.com/user/59/99849127-db9e-46fa-9a33-52cd3c4ca188.png)

从上图中可以得知，swizzle 实质上解决了这么一个问题：

- 给定一组全局地址 gAddr (上图中 warp 内 32 个线程给出了 32 个地址)，如何映射到一组共享内存地址 sAddr，使得某种访问模式 mode (上图中即为访问一个 8x8 同色子块) 在共享内存中无 bank conflict

经过我们努力的探索后，该问题可用下面的话回答：

- 将各 gAddr 的 **不同之处** 作用到共享内存的 **bank 分布** 上即可

我亲爱的读者，你是否感到这个回答是 make sense 的呢？如果不是的话，请听我娓娓道来...

## Swizzle 模板实现

我们重新关注一下上一篇文章中的地址映射过程：

![全局内存到共享内存的地址映射](https://files.mdnice.com/user/59/d2cd31c6-b681-44d5-b348-b70e6d097e6d.png)

注意，图中的地址实际上是一个 `FP16` 指针的偏移量。`8cols` 所表示的列实际上是一个 `8*2B=16B` 的 `bank4`，即由 4 个 `4B` 的 `bank` 形成的一个列。

从图中不难看出，由于我们的访问模式为 8x8 矩阵，我们便将 `gAddr` 的 行偏移作用到 `sAddr` 的列偏移上，使得表示 bank4 分布的列偏移位包含了 `gAddr` 的行信息，从而成功地将 8x8 矩阵的每一行分布到了不同的 `bank4` 上。

由此可以看出，在由 `gAddr` 形成 `sAddr` 时，我们关注一个地址的以下三部分：

- `MBase`: 表示一个块所需的位，上例中为 3，一个块有 8 个 `FP16`
- `BBits`: 表示 bank 分布的位，上例中为 3，即中间 3 位所表示的 8cols
- `SShift`: 包含 `gAddr` 信息的位移动到 `BBits` 的距离，用于映射 bank 分布 `BBits`

进而我们可以通过以下方式对 swizzle 进行抽象：

![Swizzle 模板代码](https://files.mdnice.com/user/59/d7459cf7-776f-411a-afdd-c085548e3e01.png)

我们将三个参数作为模板参数以实现编译期计算，可以很好的利用 Cpp(ComPile time Programming) 的优势。可以看到，实际运行时仅有 `return` 处的开销，这部分 bit 操作笔者选择相信编译器。

Elegant !!! 我们成功地实现了属于自己的 swizzle。下面就让我们用这层抽象大杀四方吧桀桀桀~~~

## Swizzle 示例

本部分我们将通过几个例子说明如何在实践中使用 swizzle，笔者相信这部分会很有意思（笑）。这部分涉及对于 16x16 矩阵进行 ldmatrix 加载的相关内容，不熟悉的读者可以绕道 实用 Swizzle 教程（一） 复习一下。

各示例的参考实现可见 实验仓库 的 src/mma.cuh。

### 基本用法

- Problem 1.1: 给定全局内存中的一个 FP16 类型的 16x16 矩阵 A 和一个整数 n , 请使用 Tensor Core 计算 A 的 n 次幂
- Problem 1.2: 给定全局内存中的一个 FP16 类型的由两个 16x16 矩阵 A, B 拼接而成的 16x32 矩阵 M, 请使用 Tensor Core 计算 A 与 B 的乘积。

Problem1.1 的核心在于无冲突访问共享内存中的 16x16 矩阵。经过上一篇文章和本文的抽象，相信读者能够轻蔑一笑，甩出下面的解法：

![16x16 swizzle](https://files.mdnice.com/user/59/bef73793-ca64-47ca-88be-517973bda2af.png)

Problem1.2 类似，不过是将两个 16x16 视为了一个整体，读者仍能毫不费力的甩出下面的解法：

![16x32 swizzle](https://files.mdnice.com/user/59/11c1cf6a-89f9-4d2a-8281-be32fc3b7923.png)

### 多层(Multi-) Swizzle

- Problem 2.1: 给定全局内存中的两个 FP16 (half) 类型的 16x256 矩阵 A, B 与两个结果存储地址 C1, C2 ，kernel 启动配置为 <<<1, dim3(32,16)>>>，即使用 16 个 warp 进行计算，现要求完成以下操作：
    - 以 16 列为单位，对于每个 16x16 矩阵块，使用 Tensor Core 进行矩阵乘法 Csub = Asub * Bsub^T 后写到全局内存 C1
    - 将每一行的 256 个元素视为一个 16x16 矩阵块，使用 Tensor Core 进行矩阵乘法 Csub = Asub * Bsub^T 后写到全局内存 C2 

该问题图示如下：

![多模式访问同一个矩阵](https://files.mdnice.com/user/59/9bbabf41-3fb6-4ae5-b176-86514acc2621.jpg)

这个问题的难点在于同时支持两种访问模式的无冲突访问，直觉较强的读者可能已经隐隐有得道飞升之感...

我们首先考虑应用第一层 swizzle 以解决第一个子问题的无冲突访问，如下图所示：

![第一层 Swizzle](https://files.mdnice.com/user/59/95cca358-48d9-47e8-bbf6-2863c22fd718.png)

注意到，此种模式下对于第二个子问题的访问仍会存在冲突。以第一行为例，敏锐的读者可能已经注意到了其会产生与 Problem1.1 相同的冲突。于是乎，我们便有了一个有趣的想法：

再施加一层 Swizzle

You get it!, 通过连续施加两层 Swizzle，我们便获得了两种模式下的无冲突访问。

整个过程如下图所示：

![双层 Swizzle](https://files.mdnice.com/user/59/ba585773-4db9-42d2-99a8-a36b6b9bdbbe.jpg)


### 混叠(Interleaving-) Swizzle

- Problem 3.1: 给定全局内存中的两个 FP16 (half) 类型的 16x256 矩阵 A, B 与结果存储地址 C，kernel 启动配置为 <<<1, dim3(32,16)>>>，即使用 16 个 warp 进行计算，现要求完成以下操作：
    - 以 16 列为单位，对于每个 16x16 矩阵块，使用 Tensor Core 进行矩阵乘法 Csub = Asub * Bsub^T 后写回共享内存
    - 将每一行的 256 个元素视为一个 16x16 矩阵块，使用 Tensor Core 进行矩阵乘法 Csub = Asub * Bsub^T 后写回共享内存
    - 将共享内存的内容写回全局内存

细心的读者可能已经发现了，本例使用与多层 Swizzle 一样的技术也可解决。但实际上还有另一种更有意思的做法，此处仅提供思路如下：
    - 仅为第一种方式使用 Swizzle 加载到共享内存，而后进行第一个子问题的计算
    - 在第一个子问题计算结束时，为第二种方式使用 Swizzle 写回共享内存
    - 进行第二个子问题的计算，结束后以第二种方式的 Swizzle 写回共享内存
    - 以第二种方式的 Swizzle 将共享内存的内容写回全局内存

需要注意的是在第二步写回之前需要进行块内的同步 syncthreads()，因为 Swizzle 的方式发生了变化，可能写回的地址还未完成第一步计算。

通过混叠 Swizzle，我们可以在算子的整个计算过程中随意切换共享内存的 Layout 以适应不同的访问模式，在 bank conflict free 的天堂反复横跳。

## 结语

Congratulations!!! 相信读者们已经掌握了如何定义和使用自己的 swizzle，并能在 bank conflict free 的世界里尽情遨游啦。笔者认为，在现阶段的算子优化中，swizzle 已经成为了算子仙人们的必备法宝之一，这篇文章也旨在帮助初入门的工作者们理清相关概念和用法，希望能对大家有所帮助！

读者如果有其他有趣的问题或玩法，就请在评论区留下相关思考吧！



















