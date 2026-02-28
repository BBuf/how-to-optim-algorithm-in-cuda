> 博客来源：https://leimao.github.io/article/CuTe-Layout-Algebra/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CuTe 布局代数

## 引言

CuTe布局代数(https://github.com/NVIDIA/cutlass/blob/v3.5.1/media/docs/cute/02_layout_algebra.md) 对于理解和应用CUTLASS(https://github.com/NVIDIA/cutlass/) 进行加速计算极其重要。尽管CuTe有关于其布局代数的文档，但如果不首先理解其数学基础，就无法完全理解它。我试图自己为CuTe布局代数创建一些证明，但意识到这是一个巨大的工作量。令人感激的是，Jay Shah创建了一篇论文"A Note on the Algebra of CuTe Layouts"(https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf)，完成了我想要创建的CuTe布局代数数学基础。

作为我的校对，我发现Jay Shah的论文大部分都没有错误，除了一些非常小的疏忽和打字错误。然而，它确实跳过了一些细节，没有这些细节，论文有点难以理解。在这篇文章中，基于Jay Shah的论文，我想为CuTe布局代数提供更多的证明和解释。大部分定义和注释将遵循Jay Shah的论文。

这篇文章可以作为Jay Shah论文的补充来阅读，但它也是理解CuTe布局代数的独立文章。

## 布局代数基础

### 定义 2.1：布局

*布局* $L$ 是一对匹配维度的正整数元组 $\mathbf{S}$ 和 $\mathbf{D}$。我们称 $\mathbf{S}$ 为*形状*，$\mathbf{D}$ 为*步长*。我们写作 $L = \mathbf{S} : \mathbf{D}$。

扁平化布局意味着形状和步长中没有内部括号。例如，$L = (5, 2, 2) : (16, 80, 4)$ 是一个扁平化布局，而 $L = (5, (2, 2)) : (16, (80, 4))$ 不是。扁平化布局不会改变布局的语义和操作。

### 定义 2.2：布局大小、长度和模式

设 $\alpha \geq 0$ 为整数，$L = \mathbf{S} : \mathbf{D} = (M_0, M_1, \ldots, M_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 为布局。那么：

• $L$ 的*大小*是乘积 $M = M_0 \cdot M_1 \cdot \ldots \cdot M_\alpha$。

• $L$ 的*长度*是整数 $\alpha + 1$。

• $L$ 的*模式*是条目 $(M_k) : (d_k)$ 中的一个，其中 $0 \leq k \leq \alpha$。我们可以将其视为长度为1的布局。

### 连接

给定两个布局 $L = \mathbf{S} : \mathbf{D}$ 和 $L' = \mathbf{S}' : \mathbf{D}'$，设 $\mathbf{S}''$ 和 $\mathbf{D}''$ 分别是由 $(\mathbf{S}, \mathbf{S}')$ 和 $(\mathbf{D}, \mathbf{D}')$ 的（扁平化）给出的形状和步长元组。那么 $L$ 和 $L'$ 的*连接*由布局给出

$$(L, L') = \mathbf{S}'' : \mathbf{D}''$$

我们说 $(L, L')$ 由 $L$ 和 $L'$ 分解。

归纳地，给定布局 $L_0, L_1, \ldots, L_N$，我们可以形成连接 $(L_0, L_1, \ldots, L_N)$。相反，给定布局 $L$，$L$ 由其模式最大分解。

### 同构

设 $\mathbf{S} = (M_0, M_1, \ldots, M_\alpha)$ 和 $\mathbf{D} = (d_0, d_1, \ldots, d_\alpha)$ 分别是 $L = \mathbf{S} : \mathbf{D}$ 的形状和步长元组。设 $M = M_0 \cdot M_1 \cdot \ldots \cdot M_\alpha$ 是 $L$ 的大小，设 $[0, M) \subset \mathbb{N}$ 是由 $0, 1, 2, \ldots, M - 1$ 给出的自然数子集。那么我们有一个同构

$$\iota : [0, M) \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha)$$

给定任何 $x \in [0, M)$，同构 $\iota$ 将 $x$ 映射到元组

$$x \mapsto \left(x \bmod M_0, \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1, \ldots, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor \bmod M_\alpha\right)$$

同构映射是双射的。在我们的情况下，给定任何元组 $(x_0, x_1, \ldots, x_\alpha) \in [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha)$，同构逆映射将元组映射到整数

$$(x_0, x_1, \ldots, x_\alpha) \mapsto x_0 + x_1 \cdot M_0 + x_2 \cdot M_0 \cdot M_1 + \ldots + x_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

验证上述同构映射有效并证明上述同构映射是双射的（通过反证法）是直接的。

可以将同构想象为一维坐标和多维坐标之间的映射。

### 定义 2.3：布局函数

给定布局 $L$，其*布局函数*是函数 $f_L : [0, M) \to \mathbb{N}$，定义为复合

$$[0, M) \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha) \subset \mathbb{N}^{\times(\alpha+1)} \xrightarrow{d_0, d_1, \ldots, d_\alpha} \mathbb{N}^{\times(\alpha+1)} \to \mathbb{N}$$

换句话说，$f_L$ 是多线性函数的复合

$$[0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha) \to \mathbb{N}$$
$$(x_0, x_1, \ldots, x_\alpha) \mapsto x_0 \cdot d_0 + x_1 \cdot d_1 + \ldots + x_\alpha \cdot d_\alpha$$

由步长确定，与由形状确定的同构 $\iota$ 的复合。

计算布局函数 $f_L$ 在点 $x \in [0, M)$ 处的值可以分解为计算布局函数在多个点处的值的和。这有时对于方便地计算布局函数在某点的值很有用。

给定布局 $L = (M_0, M_1, \ldots, M_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 和 $x \in [0, M)$，

$$x \mapsto (x_0, x_1, \ldots, x_\alpha) \mapsto x_0 \cdot d_0 + x_1 \cdot d_1 + \ldots + x_\alpha \cdot d_\alpha$$

我们也有

$$x_0' \mapsto (x_0, 0, 0, \ldots, 0) \mapsto x_0 \cdot d_0$$
$$x_1' \mapsto (0, x_1, 0, \ldots, 0) \mapsto x_1 \cdot d_1$$
$$\vdots$$
$$x_\alpha' \mapsto (0, 0, 0, \ldots, x_\alpha) \mapsto x_\alpha \cdot d_\alpha$$

因此，我们有

$$f_L(x) = f_L(x_0') + f_L(x_1') + \ldots + f_L(x_\alpha')$$

$$x_0' = x \bmod M_0$$

$$x_1' = \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1 \cdot M_0$$

$$\vdots$$

$$x_\alpha' = \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor \bmod M_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

例如，给定布局 $L = (3, 2) : (2, 3)$ 和 $x = 5$，我们有

$$f_L(5) = f_L(5 \bmod 3) + f_L\left(\left\lfloor \frac{5}{3} \right\rfloor \bmod 2 \cdot 3\right)$$

$$= f_L(2) + f_L(3)$$

$$= 2 \cdot 2 + \left\lfloor \frac{3}{3} \right\rfloor \cdot 3$$

$$= 4 + 3$$

$$= 7$$

### 布局函数的扩展

基于布局函数的定义，布局函数 $f_L$ 的扩展是函数 $\hat{f_L} : \mathbb{N} \to \mathbb{N}$，通过在 $f_L$ 的定义中将 $M_\alpha$ 替换为 $\infty$ 来定义，即复合

$$\mathbb{N} \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_{\alpha-1}) \times \mathbb{N} \subset \mathbb{N}^{\times(\alpha+1)} \xrightarrow{d_0, d_1, \ldots, d_\alpha} \mathbb{N}^{\times(\alpha+1)} \to \mathbb{N}$$

其中同构 $\iota$ 的扩展 $\hat{\iota}$ 由下式给出

$$x \mapsto \left(x \bmod M_0, \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1, \ldots, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-2}} \right\rfloor \bmod M_{\alpha-1}, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor\right)$$

同构扩展的映射也是双射的。同构扩展的逆映射也由下式给出

$$(x_0, x_1, \ldots, x_{\alpha-1}, x_\alpha) \mapsto x_0 + x_1 \cdot M_0 + x_2 \cdot M_0 \cdot M_1 + \ldots + x_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

可以将同构的扩展想象为将形状的最后一个维度定义为"批次"维度，批次大小可以是无限的。

#### 扩展函数的详细解析

布局函数的扩展 $\hat{f_L}$ 是将原始布局函数 $f_L$ 从有限域扩展到无限域的重要操作，其核心思想是通过"批次维度"概念来处理动态大小的数据。

**原始布局函数 vs 扩展布局函数对比：**

- **原始布局函数** $f_L$：定义域为有限集合 $[0, M)$，其中 $M = M_0 \cdot M_1 \cdot ... \cdot M_\alpha$
- **扩展布局函数** $\hat{f_L}$：定义域为无限集合 $\mathbb{N}$，通过将最后维度 $M_\alpha$ 替换为 $\infty$ 实现

**扩展的关键机制：**

在扩展同构映射中，前面的维度（$x_0, x_1, ..., x_{α-1}$）仍然有界限并循环使用，而最后一个维度（$x_α$）变为无界限，可以无限增长。这体现在坐标计算公式中：

- 前面维度：$x_i = \lfloor x/\prod_{j=0}^{i-1} M_j \rfloor \bmod M_i$（有mod操作）
- 最后维度：$x_α = \lfloor x/\prod_{j=0}^{α-1} M_j \rfloor$（无mod操作）

**实际应用示例：**

考虑布局 $L = (2, 3) : (1, 2)$，其中 $M_0 = 2, M_1 = 3, M = 6$：

- 原始函数定义域：$\{0, 1, 2, 3, 4, 5\}$
- 扩展函数定义域：$\{0, 1, 2, 3, 4, 5, 6, 7, 8, ...\}$

对于 $x = 7$（超出原始范围）：
```
x₀ = 7 mod 2 = 1
x₁ = ⌊7/2⌋ = 3  // 注意：没有mod 3操作
坐标：(1, 3)
函数值：f̂_L(7) = 1×1 + 3×2 = 7
```

对于 $x = 8$：
```
x₀ = 8 mod 2 = 0  
x₁ = ⌊8/2⌋ = 4
坐标：(0, 4)
函数值：f̂_L(8) = 0×1 + 4×2 = 8
```

**CUDA编程中的应用价值：**

1. **批处理支持**：可以处理任意大小的批次数据，无需预先知道确切的批次大小
2. **流水线处理**：支持连续的数据流处理，适用于实时计算场景
3. **内存访问模式保持**：前面维度的访问模式保持不变，只在最后维度上扩展，确保内存访问的局部性

扩展函数本质上将有限的多维索引空间转换为半无限空间，为处理动态大小数据提供了坚实的数学基础，这在现代GPU计算中具有重要的实际意义。

## 互补

### 定义 2.4：排序布局

设 $A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 为布局。我们说 $A$ 是*排序的*，如果 $d_0 \leq d_1 \leq \ldots \leq d_\alpha$，且对于每个 $i < j$，如果 $d_i = d_j$，则 $N_i \leq N_j$。

注意，排序布局，或更一般地，改变布局模式的顺序，将改变布局的语义和操作。

例如，假设我们有布局 $A = (2, 4) : (4, 1)$ 和布局 $B = (4, 2) : (1, 4)$。我们可以看到 $B$ 是 $A$ 的排序版本。我们可以使用查找表计算 $A$ 和 $B$ 的布局函数如下：

$f_A(0) = f_A(0, 0) = 0 \cdot 4 + 0 \cdot 1 = 0$
$f_A(1) = f_A(1, 0) = 1 \cdot 4 + 0 \cdot 1 = 4$
$f_A(2) = f_A(0, 1) = 0 \cdot 4 + 1 \cdot 1 = 1$
$f_A(3) = f_A(1, 1) = 1 \cdot 4 + 1 \cdot 1 = 5$
$f_A(4) = f_A(0, 2) = 0 \cdot 4 + 2 \cdot 1 = 2$
$f_A(5) = f_A(1, 2) = 1 \cdot 4 + 2 \cdot 1 = 6$
$f_A(6) = f_A(0, 3) = 0 \cdot 4 + 3 \cdot 1 = 3$
$f_A(7) = f_A(1, 3) = 1 \cdot 4 + 3 \cdot 1 = 7$

$f_B(0) = f_B(0, 0) = 0 \cdot 1 + 0 \cdot 4 = 0$
$f_B(1) = f_B(1, 0) = 1 \cdot 1 + 0 \cdot 4 = 1$
$f_B(2) = f_B(2, 0) = 2 \cdot 1 + 0 \cdot 4 = 2$
$f_B(3) = f_B(3, 0) = 3 \cdot 1 + 0 \cdot 4 = 3$
$f_B(4) = f_B(0, 1) = 0 \cdot 1 + 1 \cdot 4 = 4$
$f_B(5) = f_B(1, 1) = 1 \cdot 1 + 1 \cdot 4 = 5$
$f_B(6) = f_B(2, 1) = 2 \cdot 1 + 1 \cdot 4 = 6$
$f_B(7) = f_B(3, 1) = 3 \cdot 1 + 1 \cdot 4 = 7$

我们可以看到，布局 $B$ 通常被称为列主序布局，布局 $A$ 通常被称为行主序布局。它们是完全不同的布局。

更一般地，排序布局就像列主序布局的"泛化"。

### 定义 2.5：互补的可接受性

设 $A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 为布局，$M$ 为正整数。如果 $A$ 未排序，则将 $A$ 替换为其排序版本。我们说对 $\{A, M\}$ 是*可接受互补的*（或简称可接受的），如果：

• 对于所有 $1 \leq i \leq \alpha$，$N_{i-1} \cdot d_{i-1}$ 整除 $d_i$。

• $N_\alpha \cdot d_\alpha$ 整除 $M$。

$\{A, M\}$ 可接受互补也意味着：

• 对于所有 $1 \leq i \leq \alpha$，$N_{i-1} \cdot d_{i-1} \leq d_i$ 且 $d_{i-1} \leq d_i$。

• $N_\alpha \cdot d_\alpha \leq M$ 且 $d_\alpha \leq M$。

### 定义 2.6：互补

设 $A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 为布局，$M$ 为正整数。如果 $\{A, M\}$ 可接受互补，那么如果 $A$ 未排序，将 $A$ 替换为其排序版本。$\{A, M\}$ 的互补定义为布局

$$\text{complement}(A, M) = \left(d_0, \frac{d_1}{N_0 d_0}, \frac{d_2}{N_1 d_1}, \ldots, \frac{d_\alpha}{N_{\alpha-1} d_{\alpha-1}}, \frac{M}{N_\alpha d_\alpha}\right) : (1, N_0 d_0, N_1 d_1, \ldots, N_\alpha d_\alpha)$$

注意 $\{A, M\}$ 的互补的大小，$\text{size}(\text{complement}(A, M))$，是 $\frac{M}{\text{size}(A)} = \frac{M}{N_0 \cdot N_1 \cdot \ldots \cdot N_\alpha}$。

根据定义，$\{A, M\}$ 的互补对 $A$ 的模式顺序不敏感，因为它总是在互补之前排序。

$\{A, M\}$ 的互补是严格递增的。这可能不是很明显，所以我们将展示一个证明。

**证明**

假设 $B = \text{complement}(A, M)$，要证明布局函数 $f_B$（其定义域是自然数集合）是严格递增的，我们需要证明对于每两个相邻的自然数 $x$ 和 $x + 1$，$0 \leq x < x + 1 < \text{size}(B)$，我们有 $f_B(x) < f_B(x + 1)$。

由于同构，假设 $x$ 的映射如下：

$$x \mapsto (x_0, x_1, \ldots, x_\alpha, x_{\alpha+1})$$

根据布局函数 $f_B$ 的定义，我们有

$$f_B(x) = x_0 + x_1 \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$x + 1$ 的映射可能有许多不同的情况。

在最简单的情况下，

$$x + 1 \mapsto (x_0 + 1, x_1, \ldots, x_\alpha, x_{\alpha+1})$$

那么我们有

$$f_B(x + 1) = x_0 + 1 + x_1 \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) + 1$$

$$> f_B(x)$$

在更复杂的情况下，其中 $x_0 = d_0 - 1$ 且 $x_1 < \frac{d_1}{N_0 d_0} - 1$，我们有

$$x + 1 \mapsto (0, x_1 + 1, \ldots, x_\alpha, x_{\alpha+1})$$

> **解释：** 这个条件描述了多维坐标中的"进位"情况。具体含义如下：
> 
> - **$x_0 = d_0 - 1$**：第0维坐标已达到最大值（坐标范围是 $[0, d_0)$）
> - **$x_1 < \frac{d_1}{N_0 d_0} - 1$**：第1维坐标还未达到最大值
> 
> 当 $x$ 增加到 $x+1$ 时会发生：
> - 第0维坐标从 $d_0-1$ 溢出，重置为 $0$
> - 第1维坐标从 $x_1$ 增加到 $x_1+1$（进位）
> - 其他维度坐标保持不变
> 
> 这种分析确保了即使在坐标发生"进位"的复杂情况下，布局函数值仍然严格递增。额外的 $(N_0-1)d_0$ 项补偿了第0维重置带来的减少，确保整体函数值仍然增加。

那么我们有

$$f_B(x + 1) = 0 + (x_1 + 1) \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) - x_0 + N_0 d_0$$

$$= f_B(x) - (d_0 - 1) + N_0 d_0$$

$$= f_B(x) + 1 + (N_0 - 1) d_0$$

$$> f_B(x)$$

因为 $N_0 \geq 1$，我们有 $(N_0 - 1) d_0 \geq 0$，所以我们有

$$f_B(x + 1) > f_B(x)$$

一般来说，当 $x_0 = d_0 - 1$ 时，对于某个 $k \in [1, \alpha - 1]$，对于每个 $i \in [1, k]$，$x_i = \frac{d_i}{N_{i-1} d_{i-1}} - 1$，$x_{k+1} < \frac{d_{k+1}}{N_k d_k} - 1$，我们有

$$x + 1 \mapsto (0, 0, \ldots, 0, x_{k+1} + 1, \ldots, x_\alpha, x_{\alpha+1})$$

那么我们有

$$f_B(x + 1) = 0 + 0 \cdot N_0 d_0 + \ldots + 0 \cdot N_{k-1} d_{k-1} + (x_{k+1} + 1) \cdot N_k d_k + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) - x_0 - \left(\sum_{i=1}^{k} x_i \cdot N_{i-1} d_{i-1}\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) - \left(\sum_{i=1}^{k} \left(\frac{d_i}{N_{i-1} d_{i-1}} - 1\right) \cdot N_{i-1} d_{i-1}\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) - \left(\sum_{i=1}^{k} (d_i - N_{i-1} d_{i-1})\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) + \sum_{i=1}^{k} N_{i-1} d_{i-1} - \sum_{i=1}^{k} d_i + N_k d_k$$

$$= f_B(x) + \sum_{i=0}^{k} (N_i - 1) d_i + 1$$

因为对于每个 $i$，$N_i \geq 1$，我们有对于每个 $i$，$(N_i - 1) d_i \geq 0$，所以我们有

$$f_B(x + 1) > f_B(x)$$

这完成了证明。

类似地，我们也可以证明 $\{A, M\}$ 的互补的扩展是严格递增的。

### 命题 2.7

设 $\{A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha), M\}$ 可接受互补，$B = \text{complement}(A, M)$。设 $C = (A, B)$ 为连接布局。那么 $C$ 的大小是 $M$，且 $f_C : [0, M) \to \mathbb{N}$ 限制为双射 $[0, M) \cong [0, M)$。

**证明**

因为 $\text{size}(A) = \prod_{i=0}^\alpha N_i$ 且 $\text{size}(B) = \frac{M}{\prod_{i=0}^\alpha N_i}$，我们有 $\text{size}(C) = \text{size}(A) \cdot \text{size}(B) = M$。因此 $f_C$ 的定义域是 $[0, M)$。

注意 $f_C$ 的像与 $C$ 的任何置换 $C'$ 的 $f_{C'}$ 的像相同。

为了看到这一点，假设我们有以下布局 $C$ 及其置换 $C'$，其中只有一对模式被置换。

$$C = (N_0, N_1, \ldots, N_i, \ldots, N_j, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_i, \ldots, d_j, \ldots, d_\alpha)$$
$$C' = (N_0, N_1, \ldots, N_j, \ldots, N_i, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_j, \ldots, d_i, \ldots, d_\alpha)$$

$f_C$ 和 $f_{C'}$ 的定义域都是 $[0, M)$。对于任何 $x_C \in [0, M)$，我们有

$$x_C \mapsto (x_0, x_1, \ldots, x_i, \ldots, x_j, \ldots, x_\alpha)$$
$$x_{C'} \mapsto (x_0, x_1, \ldots, x_j, \ldots, x_i, \ldots, x_\alpha)$$

且 $x_C$ 和 $x_{C'}$ 是双射的。

因为根据定义，$f_C(x_C) = f_{C'}(x_{C'})$，所以 $f_C$ 的像与 $f_{C'}$ 的像相同。

对于 $C$ 的任何置换 $C'$，它可以通过一次置换 $C$ 的一对模式来获得，每次 $f_C$ 的像与 $f_{C'}$ 的像相同。因此，对于 $C$ 的任何置换 $C'$，$f_C$ 的像与 $f_{C'}$ 的像相同。

当计算 $f_C$ 的像时，我们可以对 $C$ 进行排序。不失一般性，假设 $A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$ 已经排序。排序 $C$ 后，排序的 $C'$ 只能如下：

$$C' = \left(d_0, N_0, \frac{d_1}{N_0 d_0}, N_1, \frac{d_2}{N_1 d_1}, N_2, \ldots, \frac{d_\alpha}{N_{\alpha-1} d_{\alpha-1}}, N_\alpha, \frac{M}{N_\alpha d_\alpha}\right) : (1, d_0, N_0 d_0, d_1, N_1 d_1, d_2, \ldots, N_{\alpha-1} d_{\alpha-1}, d_\alpha, N_\alpha d_\alpha)$$

因为对于每个 $i$，$d_i \leq N_i d_i$ 且 $N_i d_i \leq d_{i+1}$，当 $N_i = 1$ 时，$N_i \leq \frac{d_{i+1}}{N_i d_i}$，当 $N_i d_i = d_{i+1}$ 时，$\frac{d_{i+1}}{N_i d_i} \leq N_{i+1}$，因此 $C'$ 是排序的，$C'$ 的任何置换都会使其不排序。

然后我们可以重写

$$C' = (r_0, r_1, r_2, \ldots, r_\beta) : (1, r_0, r_0 r_1, \ldots, r_0 r_1 \ldots r_{\beta-1})$$

其中 $\beta = 2\alpha + 1$，$f_{C'}$ 达到的最大值计算如下：

$$f_{C'}(M - 1) = f_{C'}(r_0 - 1, r_1 - 1, r_2 - 1, \ldots, r_{\beta-1} - 1, r_\beta - 1)$$

$$= (r_0 - 1) + (r_1 - 1) \cdot r_0 + (r_2 - 1) \cdot r_0 r_1 + \ldots + (r_{\beta-1} - 1) \cdot r_0 r_1 \ldots r_{\beta-2} + (r_\beta - 1) \cdot r_0 r_1 \ldots r_{\beta-1}$$

$$= r_0 - 1 + r_0 r_1 - r_0 + r_0 r_1 r_2 - r_0 r_1 + \ldots + r_0 r_1 \ldots r_{\beta-1} - r_0 r_1 \ldots r_{\beta-2} + r_0 r_1 \ldots r_\beta - r_0 r_1 \ldots r_{\beta-1}$$

$$= r_0 r_1 \ldots r_\beta - 1$$

$$= M - 1$$

那么在这种情况下，要建立双射性断言，只需证明 $f_{C'}(x)$ 是单射的，即对于任何 $x, y \in [0, M)$，如果 $f_{C'}(x) = f_{C'}(y)$，则 $x = y$。

假设 $x$ 和 $y$ 的同构映射如下：

$$x \mapsto (x_0, x_1, \ldots, x_\beta)$$
$$y \mapsto (y_0, y_1, \ldots, y_\beta)$$

因为 $f_{C'}(x) = f_{C'}(y)$，我们有

$$x_0 + x_1 \cdot r_0 + x_2 \cdot r_0 r_1 + \ldots + x_\beta \cdot r_0 r_1 \ldots r_{\beta-1} = y_0 + y_1 \cdot r_0 + y_2 \cdot r_0 r_1 + \ldots + y_\beta \cdot r_0 r_1 \ldots r_{\beta-1}$$

我们将使用强归纳法来证明对于每个 $i \in [0, \beta]$，$x_i = y_i$。

因为 $f_{C'}(x) \bmod r_0 = f_{C'}(y) \bmod r_0$，我们有 $x_0 = y_0$。

现在假设通过强归纳法，给定 $i \in (0, \beta]$，对于所有 $j < i$，我们有 $x_j = y_j$。我们有

$$x_i \cdot r_0 r_1 \ldots r_{i-1} + x_{i+1} \cdot r_0 r_1 \ldots r_i + \ldots + x_\beta \cdot r_0 r_1 \ldots r_{\beta-1} = y_i \cdot r_0 r_1 \ldots r_{i-1} + y_{i+1} \cdot r_0 r_1 \ldots r_i + \ldots + y_\beta \cdot r_0 r_1 \ldots r_{\beta-1}$$

因为 $x_i \in [0, r_i)$ 且 $y_i \in [0, r_i)$，对这个等式取模 $r_0 r_1 \ldots r_i$ 并除以 $r_0 r_1 \ldots r_{i-1}$，我们有 $x_i = y_i$。

因为 $(x_0, x_1, \ldots, x_\beta) = (y_0, y_1, \ldots, y_\beta)$，且同构映射是双射的，我们有 $x = y$。

因此 $f_{C'} : [0, M) \to \mathbb{N}$ 限制为双射 $[0, M) \cong [0, M)$。$f_C$ 也是如此。

这完成了证明。

### 推论 2.8 互补不相交性

推论2.8解释了取布局的互补的含义。

在命题2.7的设置中，设 $I = [0, \text{size}(A)) = [0, N_0 N_1 \ldots N_\alpha)$ 为 $f_A$ 的定义域。那么

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

换句话说，$\hat{f_A}$ 和 $\hat{f_B}$ 在限制到 $f_A$ 的定义域时具有不相交的像，除了0。

注意在推论中，$f_A$ 和 $\hat{f_A}$ 实际上是可互换的，因为函数定义域限制为 $f_A$ 的定义域。

**证明**

设 $J = [0, \text{size}(B)) = [0, \frac{M}{N_0 N_1 \ldots N_\alpha})$ 为 $f_B$ 的定义域。那么根据命题2.7，我们有

$$f_A(I) \cap f_B(J) = \{0\}$$

为了理解这一点，对于任何 $x_A \in I$ 和任何 $x_B \in J$，由于同构，我们有

$$x_A \mapsto (x_{A,0}, x_{A,1}, \ldots, x_{A,\alpha})$$
$$x_B \mapsto (x_{B,0}, x_{B,1}, \ldots, x_{B,\alpha}, x_{B,\alpha+1})$$

那么我们有

$$f_A(x_A) = x_{A,0} + x_{A,1} \cdot N_0 + x_{A,2} \cdot N_0 N_1 + \ldots + x_{A,\alpha} \cdot N_0 N_1 \ldots N_{\alpha-1}$$
$$f_B(x_B) = x_{B,0} + x_{B,1} \cdot N_0 d_0 + x_{B,2} \cdot N_1 d_1 + \ldots + x_{B,\alpha} \cdot N_{\alpha-1} d_{\alpha-1} + x_{B,\alpha+1} \cdot N_\alpha d_\alpha$$

我们为布局 $C$ 编排新坐标如下：

$$x'_A \mapsto (0, x_{A,0}, 0, x_{A,1}, 0, x_{A,2}, \ldots, 0, x_{A,\alpha}, 0)$$
$$x'_B \mapsto (x_{B,0}, 0, x_{B,1}, 0, x_{B,2}, \ldots, x_{B,\alpha}, 0, x_{B,\alpha+1})$$

那么我们有

$$f_C(x'_A) = x_{A,0} + x_{A,1} \cdot N_0 + x_{A,2} \cdot N_0 N_1 + \ldots + x_{A,\alpha} \cdot N_0 N_1 \ldots N_{\alpha-1}$$
$$= f_A(x_A)$$
$$f_C(x'_B) = x_{B,0} + x_{B,1} \cdot N_0 d_0 + x_{B,2} \cdot N_1 d_1 + \ldots + x_{B,\alpha} \cdot N_{\alpha-1} d_{\alpha-1} + x_{B,\alpha+1} \cdot N_\alpha d_\alpha$$
$$= f_B(x_B)$$

根据命题2.7，我们有 $f_C : [0, M) \to \mathbb{N}$ 限制为双射 $[0, M) \cong [0, M)$。如果 $x'_A \neq x'_B$，那么 $f_C(x'_A) \neq f_C(x'_B)$，且 $f_A(x_A) \neq f_B(x_B)$。

显然，除了 $(0, 0, \ldots, 0)$ 之外，对于 $x_{A,0}, x_{A,1}, \ldots, x_{A,\alpha}$ 和 $x_{B,0}, x_{B,1}, \ldots, x_{B,\alpha}, x_{B,\alpha+1}$ 的任何值，$(0, x_{A,0}, 0, x_{A,1}, 0, x_{A,2}, \ldots, 0, x_{A,\alpha}, 0) \neq (x_{B,0}, 0, x_{B,1}, 0, x_{B,2}, \ldots, x_{B,\alpha}, 0, x_{B,\alpha+1})$，$x'_A \neq x'_B$，$f_C(x'_A) \neq f_C(x'_B)$，且 $f_A(x_A) \neq f_B(x_B)$。

这意味着，对于任何 $x \in I$ 且 $x \neq 0$，不存在 $y \in J$ 使得 $f_A(x) = f_B(y)$。

当 $x = 0$ 时，我们有 $f_A(x) = f_B(x) = 0$。因此我们可以声称

$$f_A(I) \cap f_B(J) = \{0\}$$

在定义2.6：互补中，我们已经证明了 $\{A, M\}$ 的互补，$f_B$，以及其扩展 $\hat{f_B}$，都是严格递增的。

此外，通过同构的扩展，我们有

$$\text{size}(B) \mapsto \left(0, 0, \ldots, 0, \frac{M}{N_\alpha d_\alpha}\right)$$

那么我们有

$$\hat{f_B}(\text{size}(B)) = 0 + 0 \cdot 1 + 0 \cdot N_0 d_0 + \ldots + 0 \cdot N_{\alpha-1} d_{\alpha-1} + \frac{M}{N_\alpha d_\alpha} \cdot N_\alpha d_\alpha$$
$$= M$$

$f_A$ 达到的最大值在 $N_0 N_1 \ldots N_\alpha - 1$ 处，且 $f_A(N_0 N_1 \ldots N_\alpha - 1) = (N_0 - 1) d_0 + (N_1 - 1) d_1 + \ldots + (N_\alpha - 1) d_\alpha$。

因为 $(N_0 - 1) d_0 < N_0 d_0$ 且对于每个 $i \in [0, \alpha - 1]$，$N_i d_i \leq d_{i+1}$，$N_\alpha d_\alpha \leq M$，我们有

$$f_A(N_0 N_1 \ldots N_\alpha - 1) = (N_0 - 1) d_0 + (N_1 - 1) d_1 + \ldots + (N_\alpha - 1) d_\alpha$$
$$< N_0 d_0 + N_1 d_1 - d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\leq d_1 + N_1 d_1 - d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$= N_1 d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\leq d_2 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\vdots$$
$$\leq d_\alpha + N_\alpha d_\alpha - d_\alpha$$
$$= N_\alpha d_\alpha$$
$$\leq M$$

因此 $f_A(N_0 N_1 \ldots N_\alpha - 1) < \hat{f_B}(\text{size}(B))$。

在 $I \cap J = I$ 的情况下，即 $\text{size}(A) \leq \text{size}(B)$。那么我们有

$$f_A(I) \cap f_B(I) = \{0\}$$

因为在这种情况下，$f_B(I) = \hat{f_B}(I)$，我们有

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

在另一种情况 $I \cap J = J$ 下，即 $\text{size}(A) \geq \text{size}(B)$。因为 $f_A$ 达到的最大值是 $f_A(N_0 N_1 \ldots N_\alpha - 1)$，且 $f_A(N_0 N_1 \ldots N_\alpha - 1) < \hat{f_B}(\text{size}(B))$，对于任何 $x \in I/J$，我们有 $f_A(x) < \hat{f_B}(\text{size}(B))$。

因此，

$$f_A(I) \cap \hat{f_B}(I/J) = \emptyset$$

因此，

$$f_A(I) \cap \hat{f_B}(I) = f_A(I) \cap \left(\hat{f_B}(I) \cup \hat{f_B}(I/J)\right)$$
$$= f_A(I) \cap \left(f_B(I) \cup \hat{f_B}(I/J)\right)$$
$$= (f_A(I) \cap f_B(I)) \cup \left(f_A(I) \cap \hat{f_B}(I/J)\right)$$
$$= \{0\} \cup \emptyset$$
$$= \{0\}$$

综合起来，我们有

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

这完成了证明。

关于论文中推论2.8的原始证明的简短说明是，Jay Shah声称 $f_A(I \cap J) \cap f_B(I \cap J) = \{0\}$，这不足以证明。充分的陈述应该是 $f_A(I) \cap f_B(J) = \{0\}$。


