知乎Reed佬cutlass入门笔记拷贝，基于cute在Ampere架构上实现SOTA的gemm。汇总后连续看。


# MMAOperation

operation通过指令实现D = AB + C计算，该operation需要设定A/B/C/D的操作数的类型。其中fma的参数形式依赖于D/A/B/CRegisters的类型和数据量。即，如果DRegister的类型为float[2], 则fma的接口中最前面的两个参数为float输出。如SM75_16x8x8_F32F16F16F32_TN 表示SM75算力的Turing架构的MMA，16x8x8表示矩阵的MNK大小，F32F16F16F32表示D、A、B、C的数据类型分别为float32、float16、float16、float32。T表示A矩阵为行优先，B矩阵为列优先（blas中约定normal矩阵为列优先，T表示transpose，即对列优先的矩阵进行转置则为行优先），

```shell
struct SM75_16x8x8_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float    const& c0, float    const& c1, float const& c2, float const& c3)
  {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" ...);
  }
};
```

图里的 `float[4]`（例如 `using DRegisters = float[4];`、`using CRegisters = float[4];`）表示的是 **该条 `mma.sync` 指令在“每个线程（lane）”上，用来承载矩阵片段（fragment）的寄存器数量和类型**：

## `float[4]` 具体含义
- **不是全局内存/共享内存里的数组**，而是一个“类型别名”，表示：  
  这个线程需要 **4 个 32-bit 的寄存器** 来保存该 fragment 的数据。
- 对于 `SM75_16x8x8_F32F16F16F32_TN` 这条 MMA：
  - **C 和 D 是 FP32**（累加器/输出），所以用 `float`。
  - 每个线程在一次 warp 级 MMA 运算中，负责输出 tile 的一部分，因此它会得到/持有 **4 个 FP32 输出累加值**，就用 `float[4]` 表达。

你在下面的接口里也能直接看到这种对应关系：

```cpp
fma(float &d0, float &d1, float &d2, float &d3, ...,
    float const& c0, float const& c1, float const& c2, float const& c3)
```

这里 `d0..d3` / `c0..c3` 就是这 `float[4]` 的四个标量寄存器。

## 顺便解释下为什么 A/B 用 `uint32_t[...]`
同一段代码里：
- `ARegisters = uint32_t[2]`
- `BRegisters = uint32_t[1]`

这是因为 **A/B 是 FP16 输入**，在 PTX/底层实现里通常会 **把两个 half 打包进一个 32-bit 寄存器**（例如 half2 或者按位打包），所以用 `uint32_t` 来做“原始位模式容器”更直接。

![](https://files.mdnice.com/user/59/a27d76a0-b9a3-42cd-a4b6-f930766ce0eb.png)

## 这张图在讲什么（总览）

这张图把 **一条 Warp 级 MMA 指令**（`m16n8k8`，A/B 为 FP16，累加/输出为 FP32）拆成两部分解释：

- **左侧代码（CUTE / CUTLASS 的 Traits）**：用一组 `Layout` 描述  
  “**warp 内 32 个线程**分别持有 A/B/C/D fragment 的哪些元素（寄存器）”，以及这些寄存器如何映射到矩阵 tile 的 `(m,n,k)` 坐标。
- **右侧彩色小格子图**：把这种“线程—元素映射”可视化。  
  每个小格子代表 tile 中的一个元素位置，格子里像 `T13 V1` 的标记表示：  
  **这个位置的值由第 13 号线程（lane 13）的第 1 个寄存器槽（V1）来提供/保存**。


## 左侧代码逐行解释（对应右图）

### 1) 数据类型（Element*Val）
```cpp
using ElementDVal = float;
using ElementAVal = half_t;
using ElementBVal = half_t;
using ElementCVal = float;
```
表示这条 MMA：
- A、B 输入是 `half`（FP16）
- C 累加器和 D 输出是 `float`（FP32）

这就解释了你上一张图里为什么 `CRegisters/DRegisters` 是 `float[4]`：  
**每个线程拿 4 个 FP32 累加/输出寄存器槽（V0..V3）**。



### 2) MMA tile 的形状（Shape_MNK）
```cpp
using Shape_MNK = Shape<16, 8, 8>;
```
表示每条 `mma.sync` 在一个 warp 内完成的是：

- `M = 16`
- `N = 8`
- `K = 8`

也就是计算一个 **16×8 的输出块**：`D(16×8) = A(16×8) * B(8×8) + C(16×8)`。



### 3) 线程布局（ThrID）
```cpp
using ThrID = Layout<_32>;
```
表示 warp 内 lane id：`0..31`。



### 4) ALayout / BLayout / CLayout 的含义

你会看到注释类似：

- `// (T32, V4) -> (M16, K8)`（对应 A）
- `// (T32, V2) -> (N8, K8)`（对应 B）
- `// (T32, V4) -> (M16, N8)`（对应 C/D）

这里的 `(T32, Vx)` 是关键：

- `T32`：32 个线程（warp）
- `Vx`：**每个线程持有 x 个寄存器槽**（也就是 fragment 的“向量长度”）

所以：
- A：每线程 `V4` 个 half（常见实现里就是 4 个 half，可能打包到 2 个 `uint32`）
- B：每线程 `V2` 个 half
- C/D：每线程 `V4` 个 float

#### ALayout 示例
```cpp
using ALayout = Layout<Shape<4,8,1>, Stride<-8,1,0>>;
```
它表达的是：把 `(thread_id, v)` 这类“线程内寄存器编号”映射到 **A tile 的 (m,k) 坐标**。

你不用死抠 `Stride<-8,1,0>` 的符号意义（那是 CUTE 为了同时支持转置/行列主序等，stride 允许负号），重点是：

- **ALayout 定义了每个 lane 的 V0..V3 分别对应 A tile 哪几个 (m,k) 元素**  
- 右边 A 的格子图就是这个映射的展开：每个 `(m,k)` 格子里写着 `T? V?`

同理：
- `BLayout` 定义 `(thread_id, v)` -> `(n,k)`
- `CLayout` 定义 `(thread_id, v)` -> `(m,n)`（也就是累加器/输出块的分配）



## 右侧三张/四张小图分别代表什么

右侧一般会有 A / B / C(D) 的 tile 映射可视化（你这张图右边有多块）：

- **上面那块（常见是 B 或 A 的一个视角）**  
  是输入 fragment 的线程分配（谁提供哪个输入元素）
- **中间/下面的大块（常见是 C/D 的 16×8 输出块）**  
  每个格子对应输出 `D(m,n)` 的一个元素  
  格子内 `Txx Vyy` 表示：这个输出元素保存在 **lane xx 的第 yy 个 float 寄存器**中

> 你可以用一个事实校验：  
> 输出 tile 一共 `16×8 = 128` 个 float。  
> warp 一共 32 个线程，每线程 `V4` 个 float 寄存器：`32×4 = 128`。  
> 刚好一一对应——这就是 `float[4]` 在右图里呈现出来的根本原因。



## 这张图标题里的 `TN` 是什么

`SM80_16x8x8_F32F16F16F32_TN` 里的 `TN` 表示 **A 是 Transpose，B 是 Normal**（这是 GEMM 的操作数布局约定）。

它会反映在 `ALayout/BLayout` 的 stride 选择上（比如 A 的 stride 可能带负号/不同的排列方式），从而改变右图里“线程拿到哪些位置”的模式。

# MMA_Traits

针对特定的MMAOperation类型，定义其相关的辅助类型或值给MMA_Atom使用，用以完成块状的矩阵乘法，其需要提供出的类型信息如下，

```c++
using ElementDVal =  // Logical A-value type
using ElementAVal =  // Logical B-value type
using ElementBVal =  // Logical C-value type
using ElementCVal =  // Logical D-value type

using ElementAFrg =  // A-type consumed by MMA  (if ommitted, same as ElementAVal)
using ElementBFrg =  // B_type consumed by MMA  (if ommitted, same as ElementBVal)
using ElementCFrg =  // C_type consumed by MMA  (if ommitted, same as ElementCVal)

using Shape_MNK =    // Logical MxNxK shape of the MMA

using ThrID     =    // Logical thread id (tid) -> tidx

using ALayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MK-coord
using BLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat NK-coord
using CLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MN-coord
```

# TiledMMA

TiledMMA整体表达了矩阵在MNK空间维度如何通过Atom组织而来，其结构内部定义了很多函数，这些函数提供了对给定计算块的划分能力，但是这部分逻辑终端用户早期可以不用太多关注(这些)，只需要关注如下两个API即可，第一个是TiledMMA的模版参数，第二时TiledMMA提供的get_thread_slice函数。模版参数表达了TiledMMA在MMA_Atom上的扩展逻辑：`AtomLayoutMNK`表示M N K方向上分别重复几次Atom，这种重复会要求更多的执行线程，`ValueLayoutMNK`表述了对该Atom在M N K方向上重复几次，这里的重复是通过重复计算来完成的。get_slice、get_thread_slice函数功过给定线程id则获取线程对应到ThrMMA结构，形式如下

```c++
template <class MMA_Atom,
          class AtomLayoutMNK   = Layout<Shape<_1,_1,_1>>,
          class ValLayoutMNK    = Layout<Shape<_1,_1,_1>>,
          class PermutationsMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom {
  ...;
  ThrMMA get_slice(ThrIdx thr_idx)；
  ThrMMA get_thread_slice(ThrIdx thr_idx);
  ...;
}

```

cutlass3.4版本中更新了该接口去掉了ValLayoutMNK，具体的参数解读可以参考cute核心作者Cecka的解释(link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/discussions/1345)。

# ThrMMA

该结构由TiledMMA根据具体的线程id分解而来（ThrMMA即Thread MMA），其描述了线程级实现D = A x B + C任务时的功能抽象，主要是如下`partition`类函数和`partition_fragment`类函数。其中`partition`函数完成对逻辑Tensor针对该线程的划分，即Tensor参数提供大块的逻辑矩阵单元，其返回值返回该线程需要进行的任务的Tensor描述。如Tensor C为`BLK_M x BLK_N`，则`partition_C`可以得到线程级别的任务，维度为`（MMA, MMA_M, MMA_N）`, MMA表达了TileMMA一次能计算的单元，MMA_M, MMA_N表达了M方向和N方向需要分块数量。`partition_fragment`类函数是按照`partition`类函数返回的Tensor形状生成的对应的寄存器表示。

```c++
ThrMMA {
  Tensor partition_C(Tensor C);
  Tensor partition_A(Tensor A);
  Tensor partition_B(Tensor B);
  Tensor partition_fragment_C(Tensor C);
  Tensor partition_fragment_A(Tensor A);
  Tensor partition_fragment_B(Tensor B);
}
```

![](https://files.mdnice.com/user/59/75573408-c073-4828-a3cd-2818c3706393.png)

# CUDA的存储层次和数据加载路径

![](https://files.mdnice.com/user/59/95c1df6e-d268-4cbf-952c-ebf03ee96c01.png)

经典的GPU存储层次如图1所示，主要有：片外存储结构-全局内存（global memory）；片上SM（stream multi-processor）内的shared  memory 和 L1 data cache，以及寄存器堆（register file）；以及在全局内存和SM之间的L2 Cache。具体地，全局内存（图中标记为绿色global memory）容量最大，在数据中心级A100显卡中存储空间可以达到80GB，采用HBM2e技术实现，最高带宽可达2TB/s；再往内层为L2 Cache，在A100-80GB上为80MB，带宽可达20TB/s; 再向内层为片上(on-chip)存储shared memory 和L1 data cache，shared memory 和 L1 data cache 共享192KB的空间，可以配置shared memory 和L1的大小，最大可将shared memory配置为164KB。离计算单元Tensor Core和CUDA Core（图中分别标记为TC和CUDA）更近的存储结构为寄存器堆（图中标记为Register File），计算单元计算所需要的数据必须来自寄存器（Ampere及之前架构如此，Hopper架构的Tensor Core可以直接读取存储在shared memory数据进行计算），是GPU中最快的存储结构，一个线程最多使用255个32bit的寄存器。对于一个用GPU进行计算的问题，其原始数据来自于全局内存，可以经过三条路径到达核心计算单元上（Tensor Core或CUDA Core）。第一条路径，如图中路径1所示，从global内存经过L2到达shared memory（L1 bypass），然后从shared memory到达寄存器；第二条路径，如图中路径2所示，从global内存经过L2到达L1然后到达shared memory，再从shared memory到达寄存器；第三条路径从global内存经过L2到达L1，然后到达寄存器。其中路径1和2只在Ampere及其之后的架构才支持。Ampere之前的架构从全局内存到寄存器只支持路径3，从全局内存到共享内存只能先通过3将数据加载到寄存器，然后通过路径4存储到共享内存。这些存储结构我们可以编程控制的部分为全局内存共享内存和寄存器，L1 Cache和L2 Cache是缓存机构，我们可以控制其bypass与否，同时我们也可以通过PTX指令modifier控制L2 cache的数据预取行为。

## 高效的ldmatrix指令

在矩阵计算的优化中，非常重要的一个技术是通过数据分块实现数据复用，数据复用可以减少对低层级存储器的访问数据量，提升数据访问效率继而提升总体计算效率。在GPU的实现中，可编程的数据复用发生在共享内存部分，即用户通过编程手段将部分数据加载到共享内存然后复用共享内存中的数据，实现数据由共享内存到寄存器，然后实现更高效的计算。前面MMA章节我们介绍了Tensor Core的基础信息，如果我们认真研究Tensor Core的汇编指令我们不难发现，参与计算warp内的线程只持有矩阵的部分数据，这些数据保存在线程的私有寄存器中（SIMT架构中，可以认为寄存器为线程所私有），warp内的所有线程的寄存器共同组成完整的矩阵计算数据。这是NVidia在SIMT架构下实现warp level的Tensor Core计算的创新实践。如图2所示，在SIMT意义下每个线程持有两个数据（如float16，可以表达为一个寄存器），warp内的32个线程共同构成64个数据，形成8x8的warp level的小矩阵，供Tensor Core计算用。

![Figure 2. SIMT寄存器协作构成warp level矩阵](https://files.mdnice.com/user/59/4dde5841-e95f-4736-8fd0-d422f80f732d.png)

通过各个线程提供的寄存器可以完成warp level的矩阵表示和存储，利用Tensor Core则可以完成高效的存储在寄存器上的矩阵计算。就数据从共享内存到寄存器的加载方面而言，可以通过SIMT意义下的LDS（load shared）来完成，但是由于数据是分布在不同的线程的寄存器上连续性方面不友好。为了更极致的性能NVidia从Turing架构开始提供了专门针对这种场景的加载指令ldmatrix。如图3，其展示了SIMT形式的模式加载矩阵数据和ldmatrix协作式加载矩阵数据的对比，ldmatrix协作式加载可以通过线程提供共享内存的地址（提供16Byte数据）完成数据的加载然后将数据分配到warp中各个线程的寄存器中，实现了跨越SIMT寄存器边界的写出，而如果是SIMT形式的加载，则只能使用更窄的数据位宽，这也就需要更多的指令完成等量数据的读写，同时ldmatrix由于是单线程提供16Byte的数据地址，warp内所有线程可以提供16Byte x 32 = 512Byte的数据到寄存器的加载，单指令实现16x16 float16矩阵的加载，减少指令数提高调度效率，同时其可以在写出时合并矩阵转置能力（可以参考tensorcore中ldmatrix指令的优势是什么？(https://www.zhihu.com/question/600927104/answer/3029266372)）。通过ldmatrix可以实现warp level共享内存到寄存器的数据搬运，自然地，cute对这种数据搬运提供了对应的抽象能力。

![Figure 3. SIMT形式加载矩阵数据和ldmatrix协作式加载矩阵的对比](https://files.mdnice.com/user/59/d36be7e3-7fda-47ac-8b49-05d8768ce157.png)


## cute Copy抽象及其相互关系

和MMA类似，cute对数据搬运提供了对数据搬运的数据结构抽象，主要包括`CopyOperation`、`Copy_Traits`、`Copy_Atom`、`TiledCopy`、`ThrCopy`和拷贝函数`cute::copy`。这些结构和函数共同完成对GPU各个层级存储之上的数据进行搬运的抽象和实现，具体地，

- CopyOperation提供了指令级的数据搬运的封装，NVidia在不同的硬件架构、不同的存储层次之间数据搬运提供了不同的指令，如前文提到的`ldmatrix`和`LDS`等，还有针对Ampere架构的`cp.async`等，我们在使用时只需要根据我们的硬件支持的指令情况和需要搬运的内存层次来选择已经提供的Operation即可;
- Copy_Traits和MMA_Traits类似，提供了CopyOperation类型没有提供，但是其使用者Copy_Atom却需要的起到桥梁作用的信息；
- Copy_Atom提供了指令级别不可分割的数据搬运的拷贝能力；
- TiledCopy是对Copy_Atom的能力的封装通过重复拷贝执行单元的个数（增加执行线程）或者做多次的拷贝实现对原子能力的重复；
- TildCopy提供的是逻辑上的拷贝的概念，在具体的kernel执行之时，为了复合CUDA的编程范式，需要写成线程级别的指令，ThrCopy可以实现将大块的数据根据TiledCopy所描述的划分规则，通过提供当前线程的线程号threadIdx.x对大块的Tensor进行划分，得到当前线程为了完成D = S 拷贝所需要该线程做的任务；
- cute::copy在ThrCopy提供了当前线程的任务之后，便可以通过copy函数触发具体的数据搬运指令。

![Figure 4. cute Copy核心结构和其相互关系](https://files.mdnice.com/user/59/ffdb47cc-edac-4e00-a163-b47aa3ef0019.png)


如图4所示，在硬件和拷贝方向之上提供了了指令抽象CopyOperation，再往上形成D = S的拷贝逻辑抽象，包含指令级别所能完成的拷贝原子能力Copy_Atom和对Atom重复后得到的TiledCopy能力，再逻辑之上针对具体的线程划分出具体的线程级的任务，通过cute::copy函数触发相应的拷贝任务，所有线程共同完成Tensor到Tensor的拷贝。下面我们针对每一个抽象层次，具体的介绍各个数据结构和抽象的细节。

## CopyOperation

Operation封装了特定硬件支持拷贝能力。它一般通过PTX汇编指令（或CUDA实现）来完成，实现指令集的拷贝能力抽象，其定义了源数据类型和目标数据类型以及个数，同时提供copy函数供框架层调用，示例如下，源寄存器为一个uint128_t(128bit数据)，目标寄存器位一个uint32_t的数据：

这里可以把 `CopyOperation` 理解成“**一条硬件拷贝指令的 C++ 封装**”，只做两件事：

- **[规定寄存器打包形式]** 用 `SRegisters` / `DRegisters` 表达该指令在每个线程里需要多少个寄存器槽、每个槽的位宽是什么。
- **[提供最底层的执行入口]** 用 `copy(...)` 调用对应 PTX 指令，把源（通常来自 shared memory / global memory 的某种视图）搬到目标寄存器。

你在上面的例子里看到的 `uint128_t[1] -> uint32_t[1]`，本质是在描述：

- **[SRegisters]** 该指令一次会从源侧“看见”一个 128-bit 的打包数据（可能对应 shared memory 中多个元素的组合）。
- **[DRegisters]** 指令执行后在目标侧产出一个 32-bit 的寄存器值（可能是解包后的一部分、或按指令语义重排后的片段）。

```c++
struct SM75_U32x1_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  void copy(uint128_t const& smem_src, uint32_t& dst) {
    asm volatile ("ldmatrix.sync. ...\n");
  }
};
```

## Copy_Traits

traits补充了CopyOperation的信息，如其提供了执行operation所需要的线程数，源数据和目标数据的Layout排布情况，其描述了线程和数据的存放关系，即通过线程号和寄存器号可以得到数据的逻辑位置，同时提供RefLayout供线程级的数据拆分时实现retile能力，具体的定义如下，

可以把 `Copy_Traits` 看成 CopyOperation 的“**使用说明书/元信息**”，回答三个更上层的问题：

- **[需要多少线程协同]** `ThrID = Layout<_32>` 通常表示一个 warp（32 threads）协同执行一次拷贝原子。
- **[每个线程手里的寄存器片段对应哪一部分数据]** `SrcLayout`/`DstLayout` 定义了映射：
  - 输入侧：`(src_thread, src_value_index) -> bit/coord`
  - 输出侧：`(dst_thread, dst_value_index) -> bit/coord`
- **[如何做线程级 retile]** `RefLayout` 提供一个“参考坐标系”，用于把已经是线程私有的数据重新排布成指令想要的形状。

### 如何阅读 `Layout<Shape<...>, Stride<...>>`

你可以用一个统一心智模型理解它：

- `Shape<...>`：描述一个多维索引空间的大小（类似张量的维度）。
- `Stride<...>`：描述每个维度索引加 1 时，线性地址/bit-index 增加多少。
- `Layout`：把多维索引映射到一个线性“坐标”（这里的注释写的是 `bit`，意思是它在做 bit-level 的定位；在不同上下文也可能是 element-level 的定位）。

所以 `SrcLayout`/`DstLayout` 这种注释“to bit”的 layout，常见于 `ldmatrix` 这类指令：它们关心的不只是元素序，而是**按 bit 打包/解包后的布局**。

```c++
struct Copy_Traits<SM75_U32x1_LDSM_N> {
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

## Copy_Atom

Atom将Operation和Traits进行封装和抽象，定义了内部数据类型，供形成TiledCopy和后续的ThrCopy分解任务时提取信息，如其继承来自Traits的线程情况和数据Layout情况，提供call方法实现对底层指令的调用入口，

`Copy_Atom` 可以理解成“**最小不可再分的拷贝计算单元**（原子拷贝能力）”。它把：

- Operation：怎么发指令、寄存器怎么打包（`copy`）
- Traits：需要多少线程、线程和数据怎么映射（`ThrID/SrcLayout/DstLayout/RefLayout`）

组合成一个上层可以统一调用的接口 `call(...)`。

从框架视角，`Copy_Atom` 是后续 `TiledCopy` 的“积木块”：你要么增加线程数，要么重复执行多次 Atom，把它拼成更大的拷贝。

```c++
struct Copy_Atom<Copy_Traits<Args...>, T>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = T;

  void call(Tensor<TS,SLayout> const& src, Tensor<TD,DLayout>& dst);
};

```

## TiledCopy

tiled抽象通过对Atom能力进行重复得到更大的块的拷贝能力，对Atom的重复可以通过提供线程-存储的Layout来提供，也可以直接通过提供Atom能力和MMA中的tiled_mma实现，如`make_tiled_copy_A/B/C`，因为MMA已经提供了计算`D = AxB + C`时所需要的数据划分能力，当然这些函数时针对寄存器表达能力的，具体的模版参数和形参如下。除了描述对Atom的重复方式外，TiledCopy提供的核心函数时`get_slice`和`get_thread_slice`,其可以实现将逻辑Tensor的拷贝能力根据线程的id得到每一个线程的Layout描述的拷贝任务，所以以上两个函数的返回对象为ThrCopy：

`TiledCopy` 的核心作用是：把“原子拷贝”扩展成“**能拷一块 tile 的拷贝器**”。这里 tile 是逻辑概念（MN/其他坐标空间），并不直接等于 block 维度。

重点看两个模板参数：

- **[LayoutCopy_TV]** `(tid, vid) -> coord`：描述线程与线程内寄存器槽（value index）如何覆盖这块 tile。
- **[ShapeTile_MN]** `coord space`：描述这块 tile 的逻辑形状。

你可以把它类比成 MMA 里的 tiled_mma：

- MMA 负责把一个输出 tile 的计算分配给 warp 内线程。
- TiledCopy 负责把一个输入/输出 tile 的搬运分配给 warp 或 block 内线程。

```c++
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTile_MN>   // coord space
struct TiledCopy : Copy_Atom {
  ThrCopy get_slice(ThrIdx const& thr_idx)；
  ThrCopy get_thread_slice(ThrIdx const& thr_idx));
};

CUTE_HOST_DEVICE
auto make_tiled_copy_A(Copy_Atom<Args...> const& copy_atom,
                  TiledMMA           const& tiled_mma)
```

## ThrCopy

thread copy是线程级别的拷贝的抽象，其通过TiledCopy调用`get_slice`方法而得到，其核心函数为`partition_S/D`和`retile_S/D`,其中S和D分别表示source和destination，partition表示对一个大的逻辑Tensor进行划分得到当前线程的拷贝所需要的源Tensor和目标Tensor， 而retile系列的函数表示其输入的数据已经是当前的线程的私有的数据了，但是其可能不满足拷贝所要求的形状，需要将其变换到拷贝所支持的形状，形式如下代码：

`ThrCopy` 是“**把 tiled copy 落到单线程怎么做**”的那一层。

你可以用下面两步理解它的 API：

- **[partition]** 从“整个大 Tensor”里切出“这个线程要负责的那一小片视图”。
  - `partition_S`：切源（S）
  - `partition_D`：切目标（D）
- **[retile]** 已经是线程私有的数据（比如你先用别的方式拿到了寄存器 fragment），但它的形状/布局不一定符合 Copy_Atom 的要求，需要做一个视图变换/重排，使其匹配指令的 `(tid,vid)` 语义。
  - `retile_S` / `retile_D`：把线程私有数据变成 Atom 支持的形状。

一个常见场景是：你手里已经有一份寄存器 fragment（可能来自 MMA 的 partition），但你想用某个 copy atom 把它搬到 shared memory 或从 shared memory 搬回寄存器，这时 retile 就很重要。

```c++
template <class TiledCopy, class ThrIdx>
struct ThrCopy {
 auto partition_S(Tensor&& stensor);
 auto partition_D(Tensor&& dtensor);
 auto retile_S(Tensor&& stensor);
 auto retile_D(Tensor&& stensor);
};
```

## cute::copy

copy函数是拷贝的实际执行函数，调用该函数会触发线程级别的拷贝的发生，完成线程指令的执行，实现src到dst到数据拷贝指令，实现逻辑上的`D = S`。我们用块状逻辑对数据进行拷贝的时候可能遇到边界处理的情况，这时可以通过copy_if实现对某些数据拷贝的mask，从额避免非法的数据访问，其函数原型如下，

`cute::copy` 是最上层的“**执行器**”：给它一个 `TiledCopy` 和源/目标 tensor 视图，它会在每个线程里调用对应的 `ThrCopy/Copy_Atom`，从而发出真正的硬件拷贝指令。

这里的关键点是：

- `copy(...)` 依赖你已经把 `src/dst` 组织成正确的逻辑布局（通常通过 `partition_*` 得到线程视图）。
- `copy_if(...)` 用谓词 `pred` 做 mask，常用于处理边界：
  - 逻辑 tile 覆盖到了矩阵边缘以外
  - 或者 shared memory 的某些位置不合法
  - 从而让越界 lane 不发起拷贝/不写回

```c++
void copy(TiledCopy const& copy, Tensor const& src, Tensor& dst);
void copy_if(TiledCopy const& copy, PrdTensor const& pred, Tensor const& src, Tensor& dst);
```

![](https://files.mdnice.com/user/59/2091b697-e554-4939-a3d0-9592ed36e953.png)

## 对“32x2x2=128线程、MNK=32x32x16”的计算补充解释

这里的推导核心是：`make_tiled_mma(mma_atom{}, layout_m, layout_n)` 会把一个 warp-level 的 `mma_atom` 通过在 **M 方向**和 **N 方向**做重复（replication），拼成一个更大的 `TiledMMA`。因此你可以把 `TiledMMA` 看成：

> 由若干个 `mma_atom` 组成的“原子阵列”，每个 atom 还是 32 个线程协同执行的 warp-level MMA。

下面把文中这一段计算拆开：

### 1) 为什么线程数是 `32 x 2 x 2 = 128`

- **[32]**：`mma_atom` 本身是 warp-level（SM80 Tensor Core MMA），一次原子 MMA 需要一个 warp 的 32 个线程协同。
- **[x2]**：`make_layout(Shape<_2,_2,_1>{})` 会让 `mma_atom` 在某个二维网格上重复（可以理解成把 atom 排成 2×2 的“阵列”，这会把需要协同参与的线程/warp 数增加）。
- **[再 x2]**：第二个 `make_layout(Shape<_1,_2,_1>{})` 让 atom 在另一个方向再做一次重复（同样会进一步放大并行参与的线程/warp 数）。

因此总线程数可以用“原子需要的线程数 × atom 阵列的重复规模”来估：

`threads = 32 * (M方向重复数) * (N方向重复数)`

你文中给出的结果 `32x2x2=128`，对应的直观理解就是：这个 `TiledMMA` 由 4 个 warp（共 128 线程）协同工作。

### 2) 为什么覆盖的矩阵大小是 `MNK = 32 x 32 x 16`

先从单个 `mma_op = SM80_16x8x16_...` 出发，它的原子形状是：

- **[M]**：16
- **[N]**：8
- **[K]**：16

`make_tiled_mma` 做的事情本质是：

- 在 **M 方向**拼更多 atom，就让可处理的 **M** 变大。
- 在 **N 方向**拼更多 atom，就让可处理的 **N** 变大。
- **K** 方向通常不靠“拼 atom”变大，而靠外层循环（sliced-k / split-k）推进，所以很多构造里 K 方向的重复是 1。

于是你看到的：

- `M = 16 x 2 x 1 = 32`
  - 16 来自原子指令
  - `x2` 来自某个 layout 在 M 方向的重复
  - `x1` 表示另一个 layout 在 M 方向不扩展
- `N = 8 x 2 x 2 = 32`
  - 8 来自原子指令
  - 两个 layout 都在 N 方向扩展了 2（或者等价地：一个 layout 扩展了 atom 的排列，另一个 layout 扩展了每线程寄存器/fragment 在 N 方向覆盖的次数）
- `K = 16 x 1 x 1 = 16`
  - K 仍由原子指令给定（16）
  - `x1 x1` 表示这次 tiled 组合并不在 K 方向做额外扩展

### 3) 两个 `make_layout(Shape<...>)` 分别在表达什么

在 CUTE/CUTLASS 的 `make_tiled_mma` 里，这两个 layout 可以粗略理解为两类“重复方式”的编码：

- **[layout 1]**：描述 atom 在更大 tile 里的“排布网格”（类似把多个 atom 按二维坐标放进一个大 tile）。
- **[layout 2]**：描述每个 atom 内部 A/B/C fragment 在更大 tile 中的进一步展开方式（常常体现在 N 方向对 B/C 的扩展更明显）。

更严谨的对应关系需要结合具体 `MMA_Traits` 和 `TiledMMA` 的内部布局（`thr_mma.partition_*` 的 layout）去看；但在写笔记时，你可以用一个可靠的检查法：

- **线程数检查**：最终 `size(TiledMMA{})` 是否等于你计算的线程数。
- **元素数检查**：输出 tile 的元素数 `M*N` 是否等于所有线程累加器寄存器槽位数的总和（warp/group 内 per-thread accumulator 数量 × 线程数）。


# Cute 简单的矩阵乘法

## Tensor 表示

![Figure 2. 矩阵乘法问题的Tensor表示及其属性](https://files.mdnice.com/user/59/1931d489-d973-420e-94e3-07d271d37f69.png)

如图2所示，本文我们研究深度学习中常用的C = AB的矩阵乘法问题，其中矩阵A、B存储在GPU的全局内存中，输出C矩阵将会被存储在全局内存中；维度方面A为m行k列，B为k行n列，输出C为m行n列；数据存储Layout方面，A为行优先，B为列优先，C为行优先。数据类型方面A、B、C都位16bit的半精度浮点数，在cuda中表示为half类型（cute封装为cute::half_t类型）。我们将矩阵ABC的信息形成如下表格，同时将row/column major表示为stride形式，填充上指针变量名称

![](https://files.mdnice.com/user/59/b8115bdc-3f02-452a-bf6b-a5f5945cc341.png)

我们将ABC表示为Tensor形式，则可以写出如下kernel代码

```c++
template <typename T>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); 
}
```

其中m`ake_gmem_ptr()` 函数用于标识tensor指针的存储层次，方便后续使用时能从指针中取出该指针的存储层次。同时，我们修改了Tensor B的矩阵表示形式把它的形状表示(n, k)，对应的stride 为(k, 1); 这样后续的循环的时候我们就可以写成reduce的形式。为了编译时决策和优化，我们把stride中的连续维度1表示为编译时常量的形式，即Int<1>{}，这样后续对矩阵进行操作的时候如果需要用到stride的计算则可以利用编译时的决策和优化减少不必要的运行时运算。

## 以C矩阵为中心的任务划分策略

GPU中有多个SM（Stream Multiprocessor）我们编程时以grid、block的软件层级进行编程来利用这些SM，在矩阵计算中，我们以输出矩阵C作为划分thead block的单元进行任务拆分，也就是说一个thread block完成C矩阵中的一个小块（TileC）的计算任务，如图3所示，我们定义矩阵TileC的大小为kTileM、kTileN，分别表示小块矩阵包含元素的行数目和列数目，根据块状矩阵乘法的公式完成TileC的计算需要A矩阵中的绿色高亮部分和B矩阵的黄色高亮部分，它们的形状分别为(kTileM, k)和(kTileN, k)。我们对AB矩阵的k轴按照kTileK的大小进行分块，则可以将TileC矩阵表达为AB矩阵块的点积运算

$TileC = \sum_{i_{\text{tile}}=0}^{\text{num\_tile}} TileA_{i_{\text{tile}}}TileB_{i_{\text{tile}}}$ 

![Figure 3. sliced-k模式的C矩阵为中心的任务划分方法](https://files.mdnice.com/user/59/60cf5c46-814b-41a1-928d-c3013ce29c34.png)

如此，我们在k轴上移动kTileK得到AB上的小块 $TileA_{i_{\text{tile}}}$ 和 $TileB_{i_{\text{tile}}}$ 将他们相乘的积累加到TileC上，便可以得到TileC的计算结果。这种沿着k轴移动的策略称sliced-k方法。这样我们使用一个block（坐标如图blockIdx.x, blockIdx.y）便可以完成C矩阵中一个小块的完整的计算。通过block维度的扩展，如图所示的矩阵C中M轴方向的blockIdx.y和N轴方向的blockIdx.x的扩展，便可以完成整个C矩阵的计算。由此我们可以计算出完成整个C矩阵所需要的gird维度：grid.x = N / kTileN, grid.y = M / kTileM (此处暂时不考虑不可整除的情形)。根据上面的计算过程我们可以继续完善代码，

```c++
template <typename T, int kTileM, int kTileN, int kTileK>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
}

int main() {
  ...
  dim3 grid(n / kTileN, m / kTileM);
  ...
}
```

如我们上面所述，我们在模版参数中给定划块的超参kTileM、kTileN、kTileK，同时使用Tensor章节中介绍local_tile方法对矩阵进行固定大小的分块，同时通过指定坐标的的全slice方法，得到当前thread block要处理的Tensor gA, gB, gC。和构造Tensor A时类似，我们在进行Tensor分块的时候也将编译时能确定的量进行了`Int<>`化，用以指示该维度信息是编译时常量，编译器可以在编译阶段完成必要的路径决策和优化计算，避免运行时的开销。同时我们在主函数中给出了grid的大小（如上代码）。值得注意的是，经过`local_tile`函数我们得到的gA, gB, gC的维度信息如下表格


![](https://files.mdnice.com/user/59/cb793c56-10a7-4469-9f69-66feac4e64d4.png)

先选定TileC，然后沿着k轴移动小块进行累加求和的策略为sliced-k，它对于m、n维度较大的场景（m n分块所需要的block数目足以填充所有的SM）比较有效。对于k比较大，而m、n比较小的场景，由于m、n较小而我们根据C来划分thread block，这时需要的thread block数目比较小，当这个数目无法填充所有的SM时，则存在很多SM无任务，而有任务的SM需却又需要循环多次的问题，这时候可以考虑将k轴拆分成多段，每一段都计算一个TileC结果，最后再通过额外的累加过程将多段的结果进行求和，这种模式的任务划分方法成为split-k方法。如图4，把k拆分成两段，由不同的计算单元来完成不同段段段计算，如此则得到多份C，最后将多个C进行累加求和得到最终结果。该方法在特殊场景下有用，而且实现不困难，本文暂不实现该策略。

![Figure 4. split-k策略的计算逻辑](https://files.mdnice.com/user/59/8d0521a3-2d79-4f7e-aa4f-e3d37a82341b.png)

除了sliced-k、split-k策略在任务划分方面还有stream-k方法，stream-k方法作者们指出sliced-k或者split-k方法都是静态的划分任务，在划分的任务数目和SM执行单元不能整除的时候，总会在存在某轮（wave）计算中存在SM空闲的问题。stream-k则是抛弃以任务为中心的划分逻辑，而是变成了以计算资源为核心的分配任务方式，使得SM的任务量基本相当，如图5，其展示了假设只有4个SM的情况下，不同任务划分逻辑的差异，其中stream-k对计算资源的利用效果最好，具体可以参考发表在PPoPP‘23上的poster。现阶段cuBLAS中的kernel依然多为sliced-k和split-k实现。本文暂不实现该策略。

![Figure 5. stream-k策略任务划分逻辑（引用自PPoPP23: Stream-K）](https://files.mdnice.com/user/59/e811518f-1835-497f-b36c-8933e3f1ae98.jpg)

## TiledMMA：主机端选择指令，设备端将分块划分到线程

前面经过把C++ pointer封装成Tensor，然后利用`local_tile`将Tensor划分成小块，我们便可以得到一个thread block需要处理的任务。这时，假设我们通过前面MMA章节构造了一个TiledMMA能力，则借助其方法我们便可以通过ThrMMA的partition_A/B/C方法实现对TileA、TileB、TileC、的划分，通过partition_fragment_A/B/C便可以构造矩阵乘所需要的寄存器表示。通过cute::gemm方法便可以完成线程级别寄存器表示的矩阵乘法。具体的kernel代码为

```c++
template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC); 
}

int main() {
  ...
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  auto MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  ...
}
```

其中`get_slice`函数能够将TiledMMA能力根据具体的线程id得到每一个线程所需要的layout信息。利用partition函数实现对gA gB gC的矩阵在线程级别的分解，经过partition后得到的维度信息为(MMA, MMA_M, MMA_K, num_tile_k)，其中MMA表示TiledMMA一次能做的矩阵运算所需要的数据，MMA_M、MMA_K表示(kTileM, kTileK)按照TiledMMA能力去划分的时候，M方向和K方向需要重复多少次才能完成该矩阵乘法，即M K方向需要循环多少遍TildMMA才能完成计算，而num_tile_k则为将gA的维度自然带入下来。也就是说partition_A的逻辑本质上是将Tensor的前两维进行划分，划分后得到一个三维的结果，其中第一维度表示TiledMMA单次所能处理的数据，接下来的两维表示两个方向上的重复，而如果被partition的维度再多，则后续的维度自然的继承下来即可。partition_fragment类函数和前面的类似，只是其返回的是寄存器声明。同时我们注意到对于partition_fragment_A/B，我们输入的gA时保留了前两个维度，对第三个维度我们选择了位置0，这就等价于传递给partition_fragment_A/B的是形状为(kTileM, kTileK)、（kTileN, kTileK）的Tensor。自然的其返回的结果形状也是类似的：第一维表示TiledMMA单次能力所需的数据，接下来的两维表示对TileMMA能力在M方向和K方向的重复次数。在得到TileC的寄存器表示后，我们使用clear方法将其初始化为0，以备后面进行矩阵乘法时的累加操作。

在main函数中，我们选择使用Ampere架构提供的16x8x16的Tensor Core矩阵乘法指令，数据精度和计算精度都为fp16。然后通过MMA_Traits得到mma_traits，继续将traits转换成MMA_Atom。我们知道SM80的Tensor Core执行是warp level的，也就是说这个MMA_Atom是32个线程，我们对MMA_Atom能力通过增加线程的方式进行M、N方向的重复，同时我们让B矩阵C矩阵使用更多寄存器在N方向扩展2次，得到main函数中的MMA类型。这样，我们便可以得到TiledMMA需要32x2x2 = 128线程，其能处理的矩阵的大小: M = 16 x 2 x 1 = 32, N = 8 x 2 x 2 = 32, K = 16 x 1 x 1 = 16, 即TiledMMA能处理的MNK为32x32x16。

## 补充：`32x2x2=128线程` 与 `MNK=32x32x16` 的因子分别来自哪里

这段计算可以用 CUTE 里 `make_tiled_mma(mma_atom, AtomLayout, ValLayout)` 的常见理解来解释：

- `mma_op = SM80_16x8x16_...` 决定了**单个 MMA atom** 的基本 shape：
  - `m_atom = 16`
  - `n_atom = 8`
  - `k_atom = 16`
- `mma_atom` 是 **warp-level atom**：每个 atom 需要一个 warp（32 个线程）协同执行。

在你的代码里：

```c++
auto MMA = decltype(make_tiled_mma(
  mma_atom{},
  make_layout(Shape<_2, _2, _1>{}),
  make_layout(Shape<_1, _2, _1>{})
));
```

可以把这两个 `Shape<...>` 近似理解为：

- `AtomLayout = Shape<_2,_2,_1>`：**atom 阵列在 (M,N,K) 三个方向上的排布个数**
  - M 方向排 2 个 atom
  - N 方向排 2 个 atom
  - K 方向排 1 个 atom
- `ValLayout = Shape<_1,_2,_1>`：**每个线程寄存器/fragment 在 (M,N,K) 方向上的进一步展开倍数**（不增加 atom 个数，更多是在改变每线程寄存器片段如何覆盖输出 tile）
  - M 方向不展开（1）
  - N 方向展开 2（这就是你文字里说的“让 B/C 在 N 方向扩展 2 次”的直观来源）
  - K 方向不展开（1）

### 1) 线程数为什么是 `32 x 2 x 2 = 128`

线程数由 atom 的数量决定（每个 atom 一整个 warp）：

- atom 数量 = `2 * 2 * 1 = 4`
- 每个 atom 线程数 = `32`

所以总线程数 = `32 * 4 = 128`。

注意：这里的线程数主要由 `AtomLayout` 决定；`ValLayout` 更偏向寄存器片段/覆盖方式的展开，通常不额外增加 warp 数。

### 2) MNK 覆盖范围为什么是 `32 x 32 x 16`

把“单 atom 的 shape”乘上两个 layout 的展开倍数即可得到 tiled 的覆盖范围（这是写笔记时很好用的记忆法）：

- `M = m_atom * AtomLayout.M * ValLayout.M = 16 * 2 * 1 = 32`
- `N = n_atom * AtomLayout.N * ValLayout.N = 8 * 2 * 2 = 32`
- `K = k_atom * AtomLayout.K * ValLayout.K = 16 * 1 * 1 = 16`

其中最关键的是：

- `AtomLayout` 让 tile 在 M/N 上“拼更大的块”（更多 warp 协作）。
- `ValLayout` 常用来让每线程的寄存器片段在某个方向（这里是 N）覆盖更多输出位置，从而把最终的 N 再放大一倍。


## Loop Over K

有了TiledMMA的数据划分之后，我们调用`cute::gemm`即可以完成 `C[kTileM, kTileN] =A[kTileM, kTilleK] B[kTileN, kTileK]` 利用Tensor Core进行矩阵乘的能力，我们对其进行这个块沿着k方向循环便可以得到最总的矩阵计算结果，实现如下

```c++
template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itle) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  ...
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  auto MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  gemm_simple<T, kTileM, kTileN, kTileK, MMA>(Cptr, Aptr, Bptr, m, n, k);
  ...
}
```

通过gA我们可以获取k方向的tile需要循环的次数num_tile_k，然后利用sliced-k模式循环k方向的tile，通过`cute::copy`实现全局内存到寄存器到直接copy，数据拷贝到寄存器后通过`cute::gemm`完成Tile块的矩阵乘法。在循环结束后再次利用`cute::copy`实现寄存器到全局内存的写出。`cute::copy`在不指定Copy_Atom时采用UniversalCopy实现，即简单的cuda语言层面的T d = s形式。到此，我们已经可以使用TileMMA 和 cute::copy实现简单的矩阵乘法。


- gemm_simple.cu


```c++
#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
```


# Cute之GEMM流水线

![](https://files.mdnice.com/user/59/141a116a-cf04-4f9a-80f0-be4b2c3dc4ed.png)

前面文章我们介绍了cute的Copy抽象、MMA抽象，基于这些抽象我们进行了简单的GEMM实现。从逻辑上而言，cute的介绍已经结束了，但是对于我们要完成的GEMM运算而言还有很重要的一项优化需要考虑，那就是如何高效的、并行的利用GPU中的数据加载和计算单元，亦即如何组织cute中的Copy抽象和MMA抽象以完成高效的GEMM计算。这部分内容属于GEMM的策略部分，不是cute的功能范畴，但是为了这系列文章的标题的对称性，我们依然将题目取为“cute之GEMM流水线”，我们需要知道的是，本质而言流水线部分本身不属于cute，而是GEMM的优化策略。文章结构方面，本文首先通过回顾经典的RISC硬件实现的指令流水线引入流水线对性能提升的作用，然后由类比介绍了GEMM算法常用的软件流水线（Tile间和Tile内），其后介绍了NVidia Ampere架构提供的异步拷贝指令和MultiStage流水线，最后文章总结了GEMM流水线和cute的关系。


## RISC硬件流水线

在当今的处理器微架构中，流水线技术是提升指令并行的核心技术。流水线处理器是指将每一个指令的执行过程分为多个阶段（Stage），并且允许不同指令的不同阶段可以被同时处理。以经典的RISC（Reduced Instruction Set Computer）流水线为例，一条指令的执行被分为五个阶段：

- 取指（IF = Instruction Fetch），从指令缓存中根据程序运行的位置（PC = program counter）取出一条要执行的指令；
- 译码（ID = Instruction Decode），将取出的二进制编码分解成要进行的运算类型、源寄存器和目的寄存器；
- 执行（EX = EXecute），执行单元执行特定的运算；
- 访存（MEM = MEMory），如果指令有对内存的访问需求，该阶段则负责相应的内存读写；
- 写回（WB = Write Back），将执行单元的执行结果和（或）内存的访问结果写出到目的寄存器。

![Figure 1. 流水线和非流水线处理器指令执行的时间分析](https://files.mdnice.com/user/59/c4bbe0ba-b535-4ea4-bde2-5f055070173e.png)

图1对比了非流水线和流水线架构执行指令时的时间对比，可以看到非流水线结构执行每一条指令都需要执行所有的阶段，其执行三条指令（Inst-1, Inst-2, Inst-3）所需要的时间如图上半部分所示。在流水线结构下，第一条指令在做取指后进入译码阶段，这时候第二条指令则可以进入取指阶段，后续的指令阶段也是类似的可以产生重叠。如图中的下半部分所示，以流水线的形式执行三条指令所需的时间要比非流水线的模式小很多。流水线模式提升了指令执行中的各个阶段的不同单元的使用率，使得每一个时刻每一个单元都能充分利用，而不是非流水线结构中一个时间点只有一部分单元运行，而其他时间都在空闲等待。

## GEMM软件流水线（Tile间）

我们看到指令流水线通过硬件的设计逻辑可以提升各个单元的利用情况，提升并行度既而提升运行效率。对于GEMM问题而言，我们也可以采用这种思路利用软件编程的过程的来实现更好的并行。

![Figure 2. 循环k模式的矩阵乘法的指令构成](https://files.mdnice.com/user/59/cb6e4277-32b0-4820-b81e-0fd446cd127e.png)

如图2所示，一个典型的sliced-k模式的GEMM实现中，通过循环K轴方向的tile来累加得到最终的CTile的结果。则类似RISC中的流水线模式，我们将计算每一个Tile的矩阵乘积作为一个基本的单元（如RISC中的指令），则该指令的执行类比RISC流水线可以划分为多个阶段，

- 数据加载到共享内存（LDGSTS = LoaD Global STore Shared memory）
- 数据加载到寄存器（LDSM = LoaD Shared Matrix）
- 块状矩阵乘法运算（MMA = Matrix Multiply Accumulate）

其中第一个阶段的输出数据存放在共享内存中，第二个阶段的数据存放在寄存器中。和RISC中的流水线类似，我们把一个Tile的计算过程分成三个阶段，如果各个阶段可以重叠则效率可以极大的提升，于是有了流水线思路优化的GEMM执行效果，如图3所示，

![Figure 3. 非流水线模式和流水线模式下的GEMM执行逻辑](https://files.mdnice.com/user/59/2964b3ba-4379-4ed7-bc31-f8a33589fc7f.png)

这样三个阶段的执行便可以并行起来，通过流水线使得各个单元能够同时工作，提高了各个单元的利用效率，包括全局内存到共享内存到数据加载、共享内存到寄存器到数据加载、矩阵计算，提升GEMM的运行效率。

## Tile内的流水线

![Figure 4. GEMM Tile内的流水线模式](https://files.mdnice.com/user/59/8ad09ba4-2891-4807-9dec-c8aae097d3ca.png)

对于矩阵乘法中的分块模式Tile内也可以使用流水线模式（本文称为tile内小k循环），如图4所示，对于Tile级别的矩阵乘，一般一个Tile内包含的矩阵大小需要若干个指令（MMA_Atom中的指令）才能完成矩阵乘法，并且各个矩阵乘法的输入数据相互独立，所以我们可以将数据加载和计算组成流水线模式以提高数据加载单元和计算单元的利用率，如此便可以形成Tile内的流水线模式（二级流水线），如图中pipelined标记的部分，其可以通过重叠数据加载和计算提高Tile内完成矩阵计算的整体效率。

## 异步拷贝和MultiStage流水线

为了提升数据加载效率，NVidia在Ampere架构的GPU中提供了异步拷贝指令`cp.async`（SASS汇编为LDGSTS = LoaD Global Store Shared）。该异步拷贝指令可以异步地完成全局内存到共享内存的数据加载。在Ampere架构之前，全局内存到共享内存数据的加载必须经过寄存器，所以在寄存器层面会产生数据依赖，由于GPU的顺序执行机制和scoreboard的依赖解决方法（in-order issue, in-order execute），使得全局内存到共享内存到数据有寄存器依赖引入到stall。而Ampere下提供的`cp.async`则克服了这个约束，直接做实现全局内存到共享内存到加载，由于数据是异步加载的，即指令发射出去后便可以执行后续指令而无需等待，所以该架构提供了commit和wait机制来做显式的同步。其中commit用于标记事件的同步点，wait用于同步到特定同步点，保证某个同步点之前到数据都已经拷贝完成。

![Figure 5. 异步拷贝机制](https://files.mdnice.com/user/59/62aa54e9-b124-4074-a0ad-dcd03743eba8.png)

如图5所示，我们通过`cp.aync`指令提交了三个全局内存到共享内存到拷贝任务，同时通过commit提交了三个事务点和两个wait，其中`wait<1>`表示可以允许最多有一个未完成的异步事务（G2 -> S2），即`wait<1>`执行结束能够确保G1到S1的拷贝已经完成，`wait<0>`表示允许有零个未完成的事务。也就是其会等待之前所有的commit的任务都完成，即G1->S1, G2->S2, G3->S3全部已经完成。

有了异步数据拷贝指令我们便可以完成全局内存到共享内存的异步加载，即完成矩阵A、B Tile的加载，再整合Tile间和Tile内的流水线，我们可以得到GEMM计算的MultiStage流水线模型，如图6所示，

![Figure 6. MultiStage流水线模型](https://files.mdnice.com/user/59/4e1dcd58-33b1-426a-8874-d49d2666ae8a.png)

其中浅绿色的 $G^i \rightarrow S^i$ 表示全局内存到共享内存到异步数据加载其对应的大小为Tile的大小，其合并表示TileA和TileB的数据加载（即tile循环），棕色的 $S_j \rightarrow R_j$ 表示共享内存到寄存器到数据加载，其对应Tile内的小矩阵的数据加载，也是合并的表示A B的加载（即tile内的小k循环）；深绿色 $\mathrm{mma}(R_i)$ 表示寄存器上的矩阵乘法计算，亦tile内的小k循环。mma的边界有两条黑色边界线，两边界线通过曲线虚线连接，其表示tile内小k循环的起点和终点，即黑线之间完成tile内的矩阵乘法；曲线虚线表示完成一个tile的计算之后继续进行下一个tile的计算。在第一个tile开始计算之前（即第一个黑色实现边界之前）对于multistage实现（图示kStage为4），需要将stage - 1个异步的全局内存到共享内存到加载任务发射出去（G0->S0, G1->S1, G2->S2），同时为了能够读取第一个Tile的内容，则在所有异步任务发射之后，我们wait S0完成，wait之后表示数据已经到达了共享内存，在进入tile的小k循环之前我们首先从S0中取出ik = 0的矩阵计算所需要的数据到寄存器R0(第一条黑色虚线和第一条黑实线之间)，这时，我们已经有了第一个矩阵计算所需要的数据了，于是我们进入tile内的小k循环，进入循环我们需要执行三个动作：1. 发射异步读取新的Tile数据G3->S3，2. 从共享内存读取下一个小k矩阵乘法所需要的数据R1，3. 执行第一个小k的矩阵运算。其中共享内存写出的数据和mma所需数据依赖关系通过其中的箭头表示。进入小k循环后重复上面的2、3步骤即可以流水线的完成数据加载和计算在最后一个小k循环之前，我们需要读取下一个tile中的第一个小k的数据（共享内存到寄存器），但是此时下一个tile的数据（全局内存到共享内存）需要通过wait S1保证数据加载完成，所以在最后一个小k循环之前需要插入对S1的异步事务等待，等待结束后我们便可以像之前进入小k循环之前一样在进入下一个循环（tile循环）之前加载共享内存到数据到寄存器，值得注意的是此处的共享内存已经不是当前tile，而是下一个tile，即S1。读完R0之后，完成最后一个小k的mma计算，如此便完成了tile内的小k循环，小k循环结束便重复下一个tile的计算，最终完成tile循环。

如上便是multi stage的GEMM流水线（Tile间多级，Tile内二级），其中multi表示多个，具体的个数则为shared memory中间buffer的个数。即stage为5的GEMM流水线指有五个shared memory buffer的流水线设计，其中每个buffer可以存放一个Tile的数据（包含TileA和TileB）。在Tile循环开始之前需要先发射stage - 1个全局内存到共享内存的加载，然后在循环中加载下一个Tile，如此循环使用以上stage个buffer完成所有数据到加载。在没有异步拷贝支持的GPU架构中，寄存器的依赖关系和`syncthread`的全局影响，决定了其最多只能有两个memory buffer（实质是寄存器buffer），一个用于当前数据的计算，一个用于后续数据的加载，这也是常说的双缓存机制（double buffer），双算缓存机制可以认为是multi stage中stage为2的一个特例。合适的stage大小本质是数据加载能力和矩阵计算能力的balance，其由Tile大小和硬件latency决定，在具体选择时可以通过micro-benchmark来获取相应的指令的latency来正向设计，也可以通过具体环境试验tuning得到。后面的文章中，我们会利用这种软件流水线方式实现更高效的GEMM。

合适的stage大小本质是数据加载能力和矩阵计算能力的balance，其由Tile大小和硬件latency决定，在具体选择时可以通过micro-benchmark来获取相应的指令的latency来正向设计，也可以通过具体环境试验tuning得到。后面的文章中，我们会利用这种软件流水线方式实现更高效的GEMM。

# cute 之 Swizzle

前面的文章我们介绍了GEMM中的流水线技术，流水线的核心是将拷贝和计算并行或者说是将数据加载隐藏在计算过程中。矩阵计算中的数据加载是从全局内存到共享内存然后到寄存器。共享内存作为中间的媒介可以减少矩阵计算时对全局内存的访问数据量，从而提升计算访存比。共享内存为了提升访问的并行性采用多bank结构，这也造成了编程时的困难，cute通过提供swizzle抽象简化了逻辑空间和多bank存储空间的映射的复杂度。本文首先介绍了shared memory的多bank存储结构，之后介绍了矩阵计算中的ldmatrix指令对逻辑空间和存储空间的要求，再次我们介绍了异或运算的特性和Swizzle抽象，最后我们简单介绍了Thread Block Swizzle和对本文进行了总结。

## 局部性原理和Shared Memory

局部性原理（Principle of Locality）是计算机科学的基石之一，它包括空间局部性和时间局部性。其中的空间局部性（也叫数据局部性）是指对数据的使用会限制在一个相对临近的存储空间中。Cache是针对空间局部性的很好的解决方案，但是Cache的数据更新和替换逻辑一般会实现在硬件中，表现为不可编程。在SIMT（single Instruction Multiple Thread）编程模式下，线程私有的寄存器提供了线程级别的存储能力，有时线程间需要交换一些数据来协同的完成特定任务，为了追求更好的数据局部性和实现线程间的数据共享，提供可编程的、线程间可共享的Cache就显得尤其重要。CUDA在硬件SM（Stream Multiprocessor）上提供了Shared Memory存储机构，同时软件上通过提供了相应的读写接口和同步源语言来实现其读写、同步和可见性，这样线程块内的线程便可以通过共享内存完成数据共享，同时对于线程块公共使用的数据便可以存储在其中达到线程块级别的可编程的数据局部性。

由于Shared Memory是为线程块服务的，所以其必须能支持线程块内的线程并行的对其进行访问（包含数据读取和写入），为了保障Shared Memory存储结构在多线程并发读写下的效率（更低的Latency和更高的Throughput），其硬件被实现为多bank的模式，每个bank都是可以独立寻址的存储空间，bank之间可以并行的读写数据，相互之间不会影响。在NVidia的架构中，shared memory包含32个bank，bank中可寻址的基本单元为4byte，如图1所示，每个bank为黑框所包含的单元，用户看到的地址空间为箭头所示的方向，即相邻的4byte占用不同的bank。如图2，当32个线程同时访问32个不同的bank时，各个bank是并行执行的，其效率是最高的，即32个线程并发的访问32个bank中不同颜色的单元，是可以并行的，值得注意的是其中的线程编号（如图2中的T0所示）和bank中的行位置并没有连续性要求。如图3，如果某两个线程T0、T2要同时访问相同bank-2的不同地址，则这两次访问会被排队执行，即先访问该bank的一个地址，然后再访问第二个地址，这样两次访问在发射任务维度上（产生访问请求指令）时间维度上是并行的，但是在真正bank读写数据在时间维度上是串行的。这就是所谓的bank conflict。由于一个bank上有两次冲突，这种情况称为二路冲突（two-way conflict）。

![Figure 1. 共享内存bank结构和地址连续方向](https://files.mdnice.com/user/59/279ed9ee-bca3-4d4b-a2f1-f27222cd4f05.png)

![Figure 2. 无bank conflict的共享内存访问模式](https://files.mdnice.com/user/59/89face01-5352-402f-aa0f-ab21d0eee6a4.png)

![Figure 3. 两路冲突的共享内存访问模式](https://files.mdnice.com/user/59/f4e325bf-2f2b-43dc-9fca-7d1fe0dc92fd.png)

为了减少指令数，我们在进行kernel优化时会采用向量化的读写指令（也叫大字长读写），如以128bit的形式读写共享内存，此时线程需要访问的单位数据量为16byte，32个线程需要访问的数据量为16byte x 32 = 512byte。完整的512byte需要4个phase才能完成访问，第一phase，T0-T7无bank conflict的访问所有bank，第二phase，T8-T15无bank conflict的访问所有bank，第三phase，T16-T23无bank conflict的访问所有bank，第四phase，T24-T31无bank conflict的访问所有的bank。这种情况也可以看作是：shared memory基本单元为16byte，总bank数为8，冲突与否的分析不在是32线程，而变成4个phase中的不同线程。如果采用64bit的访问形式，则相应的基本单元可以看作是8byte，总bank数目为16，冲突与否的条件变成两个phase内的线程是否冲突。整体上shared memory空间可以看作二维存储空间，其中列方向表示bank情况，行方向表示自由定义的大小。值得注意但是冲突与否是通过内存访问事务级别来判定的，具体的可以参考NVidia开发者论坛的讨论(link.zhihu.com/?target=https%3A//forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900)。

## 共享内存读取（ldmatrix指令）

![Figure 4. ldmatrix输入和输出数据](https://files.mdnice.com/user/59/b17ceb98-8379-4c0d-9557-9ec50160a7a0.png)

在GEMM流水线中，利用Tensor Core可以完成特定规格的矩阵计算乘计算（如 $D_{16\times8} = A_{16\times16} B_{16\times8} + C_{16\times8}$），其中矩阵数据A、B、C、D是通过warp内的所有线程提供一部分寄存器共同表示的。如图4中右侧的register file所示，其表示32个线程T0-T31每一个线程提供一个寄存器V0（4byte），共同表示形状为8x8 half类型的矩阵块多个8x8的块可以构成更大的16x16，16x8的块。前序文章已经介绍过，这部分数据可以利用ldmatrix指令通过warp level实现。针对一个8x8-half的寄存器表示的作为输出的矩阵块，ldmatrix其输入要求为8个shared memory地址，每个地址指向一个16byte共享内存中的数据，其中T0-Addr0指向的16byte数据经过ldmatrix会被分派到T0-T3的V0寄存器中。T1-Addr1指向的数据会被分派到T4-T7的V0寄存器中。通过ldmatrix指令便可以实现矩阵数据从共享内存到寄存器到加载，前面的介绍我们知道共享内存是有bank结构的，并且按照16byte的形式进行读取，所以T0-T7读取该数据时会被作为一个独立的phase，这就要求所有的16byte表示的8个数据必须分布在不同的bank，才能确保读取共享内存数据时不产生bank conflict。图5展示了一种ldmatrix时无bank conflict的布局形式。

![Figure 5. ldmatrix指令无bank conflict时的bank占用情况](https://files.mdnice.com/user/59/d1a12b06-650d-4726-8b11-81789d543a17.jpg)

从数学逻辑上看，8x8-half的寄存器数据表示连续的矩阵块，共享8x16byte的内存也有很好空间局部性的矩阵块，但是从共享内存的存储逻辑上看，为了避免读取时的bank冲突，其必须分配在不同的bank中。所以其横向位置在共享内存排列时不是简单的逻辑向下排列，而需要横向（bank方向）错开来避免bank conflict。

## Shared Memory写入

在GEMM流水线中数据的起点是全局内存，如图6所示，矩阵乘法所需要的寄存器数据来自于共享内存，共享内存数据来自于全局内存，数学逻辑上寄存器表示的数学空间和全局内存的位置是对应的。但是共享内存由于有bank的存在，其块状数据在共享内存存储时不是简单行列排列。需要根据ldmatrix的要求来避免冲突，这样从全局内存读取数据后写入共享内存时，也需要按照逻辑要求进行存储空间的映射。同时在全局内存向共享内存加载时，为了提升全局内存的读取效率需要考虑合并访存和L2 Cache line的情况，一般会要求其线程沿着线性地址的空间顺序排列，如图T0->Tn所示。也就是说我们在做全局内存到共享内存数据搬运时，思考模型是逻辑空间，而执行时需要考虑存储空间以避免bank conflict。

![](https://files.mdnice.com/user/59/068350d9-741c-4006-8634-b266bec92c8d.jpg)

##  异或运算的封闭性和双射性

计算机异或指令（符号通常写作 `^`，也常记为 $\oplus$）接收两个输入，对于 1bit 数据，如果输入的 bit 相同则输出 0，如果 bit 不同则输出 1。对于多 bit 数据，则针对各个位置进行 1bit 异或操作。例如 $5 \oplus 3 = \texttt{0b0101} \oplus \texttt{0b0011} = \texttt{0b0110} = 6$。异或计算满足交换律、结合律。同时对于集合 $S = \{x \mid x \in [0, 2^n-1]\}$ 中的任意两个元素，异或所形成的输出仍落在 $S$ 中，因此满足封闭性。如图7，我们可以发现该结果满足双射性（bijective），以上这些性质可以通过集合理论来进行严谨的证明。

![Figure 7. 使用异或避免共享内存bank冲突](https://files.mdnice.com/user/59/133525f8-cf35-4787-b767-b0d2979912dc.png)

图7中左侧的逻辑矩阵可认为是 $i_{\text{col}}=1$ 的共享内存列，其在共享内存中对应一个 bank，即矩阵的逻辑位置为 $(i_{\text{row}}\in[0,7],\ i_{\text{col}}=1)$。我们可以通过对 column 做异或映射（swizzle）得到新的 bank 索引，例如定义 $i_{\text{bank}} = i_{\text{row}} \oplus i_{\text{col}}$，从而将坐标写为 $(i_{\text{row}}\in[0,7],\ i_{\text{bank}} = i_{\text{row}} \oplus i_{\text{col}})$。如图7中右侧黑框标注的部分，可以看到这些数据被分配到了不同的 bank 中，在读写时可以避免 bank conflict。

## Swizzle抽象

cute中通过swizzle抽象来实现共享内存bank conflict的冲突解决。通过前面的描述我们知道，在整个计算体系中，我们需要的是二维的逻辑空间来描述矩阵块，但是为了避免共享内存的冲突，我们在共享内存存储数据时需要的是物理空间。回顾之前的介绍我们知道描述逻辑空间我们可以使用Layout（本质是函数），而为了避免bank 冲突，cute中定义了swizzle抽象，swizzle的本质也是函数，swizzle作用在layout上，即函数作用在函数上，复合函数复合的定义。Layout的作用是给定坐标返回offset，而swizzle的作用则是给定offset返回bank conflict free的offset。即 $\mathrm{offset}_{\text{bank\_conflict\_free}} = \mathrm{Swizzle}(\mathrm{Layout}(\mathrm{coord}))$。为了达成这个目的，Swizzle定义了三个参数: B、M、S。它们共同表达描述一维坐标向二维空间映射的三个层次。当我们把一个一维度坐标转换成二维坐标时，我们首先将一维中连续的几个元素作为新空间中的基础元素，然后描述该二维空间有多少行和列。其中一维坐标中连续的 $2^M$ 个元素构成二维空间中最基本的元素，$2^S$ 表示新的二维空间中有多少列，$2^B$ 表示新的二维空间中有多少行。

![](https://files.mdnice.com/user/59/e8e1a8a1-a3cc-496a-add5-8f1ce3d78a8a.png)

如图8所示，当 $B=1,\ M=1,\ S=2$ 时，$M$ 描述了一维坐标连续的两个元素构成一个二维空间的基本单元，$S$ 描述了形成二维空间后的列数，$B$ 描述了形成二维空间的行数，如此我们便得到了图2-D(a)，其二维空间包含二行四列，基本单元包含两个元素。然后我们对二维空间中的列坐标和对应的行坐标进行异或得到新的列号（$i_{\text{col}}' = i_{\text{row}} \oplus i_{\text{col}}$），形成2-D(b)。如果一维坐标映射后超过出了 $2^B$ 的大小，则超出的部分的行号从0开始记，但是offset上要加上前面的所有的元素个数。在实际操作时，如我们有一块half类型，`shape: (8, 32), stride: (32, 1)` 的共享内存，我们定义 `Swizzle<3, 3, 3>` 作用到该 shared memory Layout 上，形成 `A = Composition(Swizzle<3, 3, 3>{}, Layout<Shape<8, 32>, Stirde<32, 1>>{});` 则Layout中有效的offset为 $0\sim 256$。Swizzle中 $M=3$，所以8个元素形成一个新的最小的元素，即 $8 \times 2\,\text{byte} = 16\,\text{byte}$；Swizzle中 $S=3$，所以2D空间中一行包含8个元素，则有 $8 \times 16\,\text{byte} = 128\,\text{byte}$，$128\,\text{byte}$ 为shared memory无conflict访问所有bank的最大宽度；Swizzle中 $B=3$，则2D空间 $i_{\text{row}}$ 更新的间隔为8。如此则实现了将一个逻辑的空间向2D的shared memory空间的映射，其中列的宽度为 $128\,\text{byte}$ 占满所有的bank，行列异或后得到新的列号，避免了在bank方向（亦即icol方向）的冲突。

## Thread Block Swizzle

除了避免共享内存冲突的swizzle外，cute（cutlass）中还有另一种swizzle，为thread block swizzle，在以C为中心的任务划分模式中，如果没有Thread Block Swizzle，则任务块会按照线性的行优先或者列优先的顺序分配给所有的执行单元（如图9中SM0-3，假设硬件只有4个SM），进行Thread Block Swizzle后，可以形成如图9右侧所示的任务划分关系，在某些场景下，其可以提升L2 Cache的命中率，数学上表现为在相同的元素能覆盖更大的面积，同时这部分面积(A、B)能够很好的被L2缓存住，具体的可以参考cutlass中的thread block swizzle实现。

![Figure 9. Thread Block Swizzle](https://files.mdnice.com/user/59/e4752bd8-69dc-4bf9-9753-95eb7b7075f6.jpg)

# cute 之 高效GEMM实现

前面的文章我们介绍了cute中的Layout抽象、Tensor抽象、MMA抽象、Copy抽象、Swizzle抽象和流水线技术。本文将组合利用这些抽象和技术形成高效的矩阵乘法。为了更好地实现高效的矩阵乘法，本文从多个维度介绍了高效的实现方法。在文章结构上，本文首先介绍了计算高效，然后介绍了访存高效，再后介绍了算法高效，最后介绍了尾阶段高效。我们基于这些高效方案利用cute实现了高效的矩阵乘法，并和cuBLAS、cuBLASLt进行了性能对比，对比结果表明我们的实现达到了SOTA水平。文章最后讨论了搜索kernel的启发式算法和参数相容性问题并对文章进行了总结。

## 计算高效

GEMM中的核心计算部分是块状的矩阵乘法，针对输入为半精度类型（half precision），Accumulator为半精度类型的计算任务，Ampere架构提供了Tensor Core上的如下计算指令

- `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16，`

cute将这两条指令抽象为MMA_Operation：

- `SM80_16x8x8_F16F16F16F16_TN`
- `SM80_16x8x16_F16F16F16F16_TN`

当我们的问题规格较大时，我们尽可能选用计算量更大的指令，这样同一条指令产生的计算任务就更多，可以减少指令数目，提升调度效率。

选定计算指令之后，我们便可以通过MMA_Traits，如图1，在MMA_Operation的基础上补充上后续计算所需要的其它信息，如矩阵计算的形状、该指令所需要的协作线程（此处为32线程）、A、B矩阵的寄存器Layout分布情况。有了MMA_Traits之后，我们便可以将其进一步封装为MMA_Atom，其利用Traits提供的信息，提供数据划分所需要的信息和Operation的执行功能。MMA_Atom描述了矩阵计算的原子能力（单条指令的计算能力，最小能力），我们通过增加更多的线程、每个线程做多次任务则可以将计算的规格增大，如此则有了TiledMMA，TiledMMA针对每一个线程则被分裂为ThrMMA，TiledMMA和ThrMMA利用MMA_Atom提供的信息，能够实现对矩阵块的划分。调用相应的cute::gemm函数即可以完成矩阵乘法计算。

![Figure 1. MMA能力层次和各层的主要功能](https://files.mdnice.com/user/59/dcbd196a-8d1d-40b4-9b6f-8f1078c417b2.png)


此时，我们便可以得到如下主机端代码

```cpp
using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  static constexpr int kMmaVRepeatM = 1;
  static constexpr int kMmaVRepeatN = 2;
  static constexpr int kMmaVRepeatK = 1;

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_V_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}, Int<kMmaVRepeatK>{})));

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));
```

其中前三行选择了MMA指令形成了Atom能力，然后定义了对该Atom能力的重复方法（包括线程重复和寄存器重复），它们分别形成各自重复的Layout，然后利用`make_tile_mma`接口形成更大块的矩阵乘法描述。设备端的代码如下：

```c++
TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)
```

将TileMMA提供线程号，则获得具体线程的数据划分能力，对给定的数据块进行划分，得到线程级的数据描述。

![Figure 2. partition_A/B/C逻辑示意图](https://files.mdnice.com/user/59/2e798aab-1c25-4794-841e-b678b9ef681f.jpg)

如图2所示，其展示了ThrMMA提供的partition_A/B/C和partition_fragment_A/B/C函数的计算逻辑，给定一个静态大小的Tensor TileB（被划分维度为Int<>编译时常量），则thr_mma可以对其进行划分，其中划分的逻辑为：以TileMMA中描述的矩阵大小对目标Tensor进行周期性平铺，对高亮的部分进行选取形成新的矩阵，其中第一个维度为TiledMMA中单个线程的数据描述，第二个维度和第三个维度为行方向和列方向需要重复的次数。如果TileB的维度比两维高，则高出的部分继承到N,K维度之后。类似地，A/C的划分采用同样的逻辑。

## 访存高效

整个GEMM计算体系中数据的在到达Tensor Core进行计算之前需要包含：全局内存到共享内存，共享内存到寄存器，在流水线章节我们介绍过全局内存到共享内存的异步拷贝方法，和共享内存到寄存器的ldmatrix指令。在cute中，针对全局内存到共享内存，我们和选择MMA能力类似，选择cute已经定义好的抽象能力即可，此处我们选择`SM80_CP_ASYNC_CACHEGLOBAL` Copy_Operation，该指令可以实现全局内存到共享内存到异步拷贝，同时CACHEGLOBAL指示了数据只在L2做Cache，对L1则做bypass。于是我们可以形成如下主机端代码：

```c++
using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;
```

和MMA时的make_tiled_mma类似，Copy抽象提供了make_tile_copy能力，其通过制定线程和数据的重复方法将Atom能力扩展到块状能力。数据拷贝时可以区分AB矩阵的不同拷贝方法，我们此处选用同样的Copy能力。设备端代码如下

```c++
G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)
```

和图1中的MMA层级类似，Copy时先将TileCopy通过指定线程号得到线程级的Copy能力抽象ThrCopy。也和图2中的MMA划分类似，ThrCopy抽象提供了partiton_S/D函数，其实现将大块的矩阵划分到线程维度上。经过partition_S/D划分的数据维度为(E, M, K)，E表示该线程要做的数据大小（包含分布），M、K表示由于给定的被划分的块需要在纵轴和横轴上需要重复的次数。如果被划分的Tile的维度大于2，则多出的维度附加到（，M，K）维度之后。

对于共享内存到寄存器的拷贝，cute提供了对ldmatrix指令的封装，主机端代码和设备端代码分别如下：

```c++
// shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;
```

设备端代码：

```c++
auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)
```

其中主机端选择ldmatrix指令的x4模式，形成Atom抽象，设备端通过`make_tiled_copy_A`函数借助tiled_mma抽象出共享内存到寄存的TileCopy。前面的全局内存到共享内存TileCopy不同的是，这里直接利用tiled_mma的信息形成块状拷贝，原因是TiledMMA包含了计算所需要的数据描述，所以对于以它作为目标的Copy而言，tiled_mma自然也是精准描述这部分数据的，所以就不需要用户额外制定Copy_Atom到TileCopy的信息，而是之间从MMA能力中获得，其一定程度上可以避免了独立设置的不一致性问题。由于MMA时已经声明了寄存器存储空间，此处直接对其进行线程级小块的retile即可，不再是大块到小块的partition。

## 算法高效

前面两个章节介绍了计算的高效和访存的高效，如何将这两个步骤组合起来也是GEMM性能的关键要素，这部分我们成为算法的高效。其主要包含两个部分：1. 分块；2. 流水线。这两部分内容我们在简单GEMM实现和流水线中都有介绍，我们此处就不再赘述。对于分块部分，我们主机端和设备端的代码如下：


```c++
static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
```

设备端：

```c++
作者：reed
链接：https://zhuanlan.zhihu.com/p/675308830
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

// slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _));  // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _));  // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix));  // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage)
```

分别定义了分块的大小和设备端如何通过Tensor抽象和local_tile将矩阵进行分块。

流水线方面，为了做multi stage流水线，需要在共享内存分配时指定流水线级数，同时设备端需要做必要的数据加载和计算重叠方案。主机端代码如下

```c++
作者：reed
链接：https://zhuanlan.zhihu.com/p/675308830
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
```

其定义了共享内存的Layout，其中Swizzle用来避免bank conflict，其详细介绍可以参考前序文章cute之Siwzzle抽象，其中kStage表示流水线的级数。

核心设备端代码由两个for循环组成，具体如下，

```c++
作者：reed
链接：https://zhuanlan.zhihu.com/p/675308830
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

// loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik
  }    // itile
```

外层循环对Tile做循环，内层循环对Tile内做k循环，同时在ik == 0 和 ik == nk -1的时候发射了后kStage-1的Tile的全局内层到共享内存的数据加载和针对即将读取的共享内存的数据同步。


## 尾阶段高效（Epilogue）

通过上面的计算、加载、算法，我们便可以得到矩阵乘之后的块状的数据，这些数据表示为线程内寄存器，由于数据存储，将这些数据存储出去并不平凡，如图3所示，如果将寄存器数据直接写出，则在全局地址空间中会产生内存地址的不连续，这将导致存储时需要更多的内存事务，并且不能使用向量化存储指令（STG.128）。

![Figure 3. 寄存器堆直接存储至全局内存引入的不连续](https://files.mdnice.com/user/59/5af057b2-6675-46f6-8010-0bba2cc8a381.png)

针对这个问题，cute中（实质为cutlass中），专门提供了Epilogue来通过共享内存作为中间媒介。先将寄存器数据存储到共享内存，然后再从共享内存中以更连续、更高位宽的形式存储到全局内存中去。PACT'20 Fireiron文章有对该问题的详细探讨，可以参考之。

![Figure 4. Epilogue中Accumulator寄存器通过共享内存实现到全局内存的数据搬运](https://files.mdnice.com/user/59/bc2f11a2-44f0-42f8-997a-938b531cbcf1.jpg)

本文通过共享内存实现高效的TileC的存储，具体代码可以参考github上的实现，整体过程如图4所示。

```c++
#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include <cute/tensor.hpp>

#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

// 这个示例展示了一个“多级流水线（multi-stage）”的 GEMM：
// - 计算：使用 Tensor Core（tiled_mma）做块状 MMA
// - 访存：使用 cp.async 将 gmem -> smem；使用 ldmatrix 将 smem -> reg
// - 流水线：在 k 方向上用 kStage 个 smem buffer 做环形队列（ring buffer），
//   以隐藏全局内存访问延迟
// - 尾阶段（Epilogue）：将每线程的 accumulator（reg）先写入 smem，再以更连续/更宽的写出方式写回 gmem

template <typename Config>
__global__ void /* __launch_bounds__(128, 1) */
gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
                 int k) {
  using namespace cute;
  using X = Underscore;

  // --------------------------
  // 1) 从 Config 中取出本 kernel 的“策略参数（policy）”
  // --------------------------
  // 这些 using / constexpr 会在编译期把：tile 大小、拷贝指令、layout、tiled_mma 形状
  // 等策略固化下来。
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  // tile / pipeline 形状
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // --------------------------
  // 2) 动态共享内存划分
  // --------------------------
  // shm_data 既承载 A/B 的流水线 buffer，也会在 Epilogue 阶段复用为 C/D 的 scratchpad。
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  // 当前线程（在一个 ThreadBlock 内），以及当前 ThreadBlock 对应的 tile 坐标
  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // --------------------------
  // 3) 将 (ptr, shape, stride) 包装成 cute::Tensor
  // --------------------------
  // 注意：这里 A 的 shape 是 (m, k)，B 的 shape 是 (n, k)，D 的 shape 是 (m, n)
  // 这与后面 cublasHgemm 里 CUBLAS_OP_T / CUBLAS_OP_N 的排列是一致的。
  // use Tensor notation to represent device pointer + dimension
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));  // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));  // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));  // (M, N)

  // --------------------------
  // 4) ThreadBlock 级分块：取出当前 block 要算的 gA/gB/gD
  // --------------------------
  // local_tile 的直觉：把全局大矩阵按照 (kTileM,kTileK)/(kTileN,kTileK)/(kTileM,kTileN)
  // 进行切片；iy/ix 选择当前 block 的 tile 坐标。
  // slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _));  // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _));  // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix));  // (kTileM, kTileN)

  // --------------------------
  // 5) ThreadBlock 级共享内存 tile（带 swizzle 的 layout，用于减少 bank conflict）
  // --------------------------
  // sA/sB 的 layout 里含有 kStage 这一维：表示同一个 ThreadBlock 的 A/B tile 在 smem 中
  // 有 kStage 份 buffer，用于流水线环形队列。
  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage)

  // --------------------------
  // 6) MMA：把 ThreadBlock 的 tile 按 TiledMMA 规则拆成每线程寄存器 fragment
  // --------------------------
  // tiled_mma: 描述“一个 ThreadBlock 里所有线程协作”能覆盖的 MMA tile。
  // thr_mma:  取出当前线程 idx 对应的切片（即当前线程在 MMA 中负责的那部分 fragment）。
  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  // method
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

  // fill zero for accumulator
  clear(tCrD);

  // --------------------------
  // 7) 构造“拷贝”抽象：
  //    - g2s: gmem -> smem（通常用 cp.async）
  //    - s2r: smem -> reg（通常用 ldmatrix / ldsm）
  // --------------------------
  // s2r 的 make_tiled_copy_A/B 会利用 tiled_mma 的数据需求来生成合适的 smem->reg tile copy。
  // g2s 的 G2SCopyA/B 则是 Config 里直接指定的（描述每个线程负责搬哪些元素）。
  //
  // 记号解释（后面 partition_S/partition_D 常见维度）：
  // - CPY:    每线程一次 copy 的“向量化元素数/分布”维度
  // - CPY_M/N/K: 在 tile 的 M/N/K 方向的重复次数
  // - kStage: 共享内存环形 buffer 的 stage 维度
  // gmem -cp.async-> shm -ldmatrix-> reg
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

  // --------------------------
  // 8) multi-stage 流水线状态
  // --------------------------
  // itile_to_read: 下一次需要从 gmem 搬到 smem 的 k-tile 索引
  // ismem_write:   下一次写入的 smem stage（环形队列写指针）
  // ismem_read:    当前 compute 将要读取的 smem stage（环形队列读指针）
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  // --------------------------
  // 9) 预取（prologue）：先提交 kStage-1 个 tile 的 gmem->smem 异步拷贝
  // --------------------------
  // 典型 multi-stage 流水线：在进入主循环前，先把“即将用到的”前几块 tile 填满环形队列。
  // cp_async_fence()：提交（commit）一组 cp.async，使其进入“可被 wait 追踪”的队列。
  // cp_async_wait<N>()：等待队列里只剩 N 组尚未完成；这里的 <kStage-2> 意味着至少有 1 组已完成。
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  // --------------------------
  // 10) 首次 smem -> reg：把 stage=ismem_read 的第 0 个 k-slice（ik=0）搬进寄存器
  // --------------------------
  int ik = 0;
  // smem -> reg
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // --------------------------
  // 11) 主循环：沿 K 方向推进
  // --------------------------
  // 外层 itile：在 K 方向推进 tile（每个 tile 覆盖 kTileK）
  // 内层 ik：在一个 tile 内部推进更细粒度的 k-slice（由 tiled_mma / s2r_copy 的 K 维度决定）
  //
  // 关键思想：在做当前 ik 的 MMA 的同时，尽可能提前把下一个要用的数据搬到寄存器/共享内存。
  // 形成：
  // - gmem->smem（cp.async）与 smem->reg（ldmatrix）与 mma（tensor core）三者重叠
  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        // 每个 tile 的 ik==0 时机：适合发射下一块 tile 的 gmem->smem 异步拷贝。
        // （此时我们已经开始消费当前 tile 数据，smem 上的某个 stage 很快会“腾出来”可写。）
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik
  }    // itile

  // --------------------------
  // 12) Epilogue：将 accumulator（reg）高效写回 gmem
  // --------------------------
  // 目标：避免“每线程 reg 直接写 gmem”导致的全局写出不连续。
  // 做法：reg -> smem（按更规整的 layout 摆放）-> gmem（更连续、更宽的 store）。
  //
  // 这里复用 sA 的某个 stage 作为 scratch（用更少的 smem 额外开销）。
  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

  // step（pipe）表示一次 reg->smem / smem->gmem 要走的 pipeline 深度，
  // 通常与 tiled_copy 的“分段写出”方式有关。
  int step = size<3>(tCsC_r2s);  // pipe
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // we add a temp tensor to cope with accumulator and output data type
      // difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }
}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // --------------------------
  // tile / pipeline 配置
  // --------------------------
  // kTileM/N/K: ThreadBlock 级 tile 尺寸
  // kStage:     multi-stage 流水线深度（smem 环形队列 buffer 数）
  // kSmemLayoutCBatch: epilogue 阶段 C/D 的 smem layout 在“batch/pipe”维度上的展开
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  // MMA_P_T：描述每个线程“寄存器片段/输出覆盖”的进一步展开（常见会在 N 方向更大）
  // 这里用 kMmaPM/PN/PK 构造一个 Tile<Int<M>,Int<N>,Int<K>>。
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // --------------------------
  // gmem -> smem: cp.async（这里按 uint128 做 16B 向量化搬运）
  // --------------------------
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // --------------------------
  // smem -> reg: ldmatrix（ldsm）
  // --------------------------
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // --------------------------
  // epilogue: reg -> smem -> gmem
  // --------------------------
  // SmemLayoutC 让 C/D 在共享内存里按更适合“连续写回”的方式摆放。
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  // 一个 ThreadBlock 的线程数：等于 size(MMA{})（tiled_mma 需要多少线程协作）
  static constexpr int kThreadNum = size(MMA{});
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

int main(int argc, char *argv[]) {
  using T = cute::half_t;
  using namespace cute;
  using X = Underscore;

  srand(10086);

  cublasHandle_t handle;
  cublasCreate(&handle);
  int cublas_version;
  cublasGetVersion_v2(handle, &cublas_version);
  printf("cuBLAS version: %d\n", cublas_version);

  // --------------------------
  // 这个 main 更像一个“可运行的 demo + correctness check”：
  // - 分配/初始化 A/B
  // - 跑 cublas / cublaslt / 我们的 kernel
  // - 对比结果并打印一个小 tile
  //
  // default;
  int M = 81920;
  int N = 256;
  int K = 256;

  int enable_cpu = 0;
  int enable_cublaslt = 1;
  int nt = 11;

  using ComputeType = T;

  T *Aptr;
  T *Bptr;
  T *Dptr;
  T *Dptr_cublas;
  T *Dptr_cublaslt;

  T *Aptr_host;
  T *Bptr_host;
  T *Dptr_host;
  T *Dptr_host_cpu;
  T *Dptr_host_blas;
  T *Dptr_host_cublaslt;

  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  Dptr_host = (T *)malloc(sizeof(T) * M * N);

  Dptr_host_cpu = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_blas = (T *)malloc(sizeof(T) * M * N);
  Dptr_host_cublaslt = (T *)malloc(sizeof(T) * M * N);

  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublas, sizeof(T) * M * N);
  cudaMalloc(&Dptr_cublaslt, sizeof(T) * M * N);

  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  auto tD = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));

  cpu_rand_data(&tA);
  cpu_rand_data(&tB);

  clear(tD);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
  cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
  cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);

  CublasLtGemm<T, ComputeType> cublaslt_gemm;
  if (enable_cublaslt) {
    cublaslt_gemm.init(Dptr_cublaslt, Bptr, Aptr, N, M, K);
  }

  // 选择 kernel 配置：这里用 (TileM, TileN, TileK, kStage)
  // kStage 越大通常能隐藏更多 latency，但也会带来更多 smem 占用。
  config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

  print(typename decltype(gemm_config)::MMA{});

  // 一个 block 的线程数由 tiled_mma 决定
  dim3 block = gemm_config.kThreadNum;
  dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
            (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
  int shm_size = gemm_config.kShmSize;

  half alpha = 1.f;
  half beta = 0.f;

  for (int it = 0; it < nt; ++it) {
    // blas
    cudaMemset(Dptr_cublas, 0, sizeof(T) * M * N);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                     &alpha, (half *)Bptr, K, (half *)Aptr, K,
                                     &beta, (half *)Dptr_cublas, N);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }

    if (enable_cublaslt) {
      cudaMemset(Dptr_cublaslt, 0, sizeof(T) * M * N);
      cublaslt_gemm.run();
    }

    // multi-stage（我们的实现）
    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)>
        <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
  }

  cudaMemcpy(Dptr_host, Dptr, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_blas, Dptr_cublas, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Dptr_host_cublaslt, Dptr_cublaslt, sizeof(T) * M * N,
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("block = (%d, %d), gird = (%d, %d), shm = %d\n", block.x, block.y,
         grid.x, grid.y, shm_size);

  if (err == cudaSuccess) {
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  } else {
    printf_fail("err = %d, str = %s\n", err, cudaGetErrorString(err));
  }

  gpu_compare(Dptr, Dptr_cublas, M * N);

  if (enable_cublaslt) {
    gpu_compare(Dptr, Dptr_cublaslt, M * N);
  }

  auto tD_host = make_tensor(Dptr_host, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cpu =
      make_tensor(Dptr_host_cpu, make_shape(M, N), make_stride(N, 1));
  auto tD_host_blas =
      make_tensor(Dptr_host_blas, make_shape(M, N), make_stride(N, 1));
  auto tD_host_cublaslt =
      make_tensor(Dptr_host_cublaslt, make_shape(M, N), make_stride(N, 1));

  if (enable_cpu) {
    cpu_gemm(&tD_host_cpu, tA, tB);
    cpu_compare(tD_host_cpu, tD_host, 0.1f);
  }

  auto tile = make_tile(min(8, M), min(8, N));
  auto t32x32 = local_tile(tD_host, tile, make_coord(0, 0));
  auto t32x32_cpu = local_tile(tD_host_cpu, tile, make_coord(0, 0));
  auto t32x32_blas = local_tile(tD_host_blas, tile, make_coord(0, 0));
  auto t32x32_cublaslt = local_tile(tD_host_cublaslt, tile, make_coord(0, 0));

  printf("M = %d, N = %d, K = %d\n", M, N, K);

  printf("our-impl:\n");
  print_tensor(t32x32);
  if (enable_cpu) {
    printf("cpu:\n");
    print_tensor(t32x32_cpu);
  }
  printf("cublas:\n");
  print_tensor(t32x32_blas);

  if (enable_cublaslt) {
    printf("cublaslt:\n");
    print_tensor(t32x32_cublaslt);
  }
}
```