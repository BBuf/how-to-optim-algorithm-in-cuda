> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

# 第12课，Flash Attention

## 课程笔记

这节课的演讲者也是之前[CUDA-MODE 课程笔记 第四课: PMPP 书的第4-5章笔记](https://mp.weixin.qq.com/s/P87c8LRJ1CEOOyaQw8L-cA) 这节课的演讲者，第四课的最后介绍了一下矩阵乘法的Tiling技术，最后还提到Tiling的经典应用就是Flash Attention。所以这一节课他来讲解下Flash Attention。

![](https://files.mdnice.com/user/59/7d78b5f9-628c-4ead-8b62-7b0a6c85febd.png)

这张Slides讨论了为什么在某些计算操作中使用分块（Tiling）技术，并展示了内存层次结构。
- Tiling（分块）的原因：
    - 在矩阵乘法（Matmul）中，每个输出使用2n个输入（一共n^2个输出）。
    - 每个输入被使用n次，如果每次都从主内存中naive地读取n次，会非常低效。
    - 解决方案：尝试重用参数（try to reuse param）。
- 应用场景：
    - 类似的情况也出现在卷积（Convolution）和FlashAttention等操作中。
- 内存层次结构（Memory Hierarchy）和特点：
    - GPU SRAM（静态随机存取内存）：带宽19 TB/s，容量20 MB
    - GPU HBM（高带宽内存）：带宽1.5 TB/s，容量40 GB
    - Main Memory（主内存，CPU DRAM）：带宽12.8 GB/s，容量>1 TB
    - 从上到下，内存容量逐渐增大，但访问速度（带宽）逐渐降低。
    - Slides中提到这个内存层次结构来自Dao等人的Flash Attention论文。

总的来说，这里解释了为什么在某些计算密集型操作中使用分块技术很重要。通过重用数据和利用更快的内存层（如GPU SRAM），可以显著提高计算效率。同时，Slides中展示的内存层次结构清楚地说明了不同级别内存之间在速度和容量上的权衡，这进一步强调了优化内存访问模式的重要性。

![](https://files.mdnice.com/user/59/0036022f-e58a-42a3-97f4-2b5610245c51.png)

作者提出了一个新的观点，那就是把注意力机制视为分类问题。slides的详细解释为：

1. 标题指出任务是"将注意力视为分类"。
2. 首先介绍了一些数学符号和公式：
    - q: d维输入激活
    - K = (ki): d维类别嵌入
    - logits (li): 通过q和ki的标量积计算
    - 使用softmax函数计算类别概率pi
3. 类别概率的计算公式为：
    - pi = exp(d^(-1/2) li) / Σj exp(d^(-1/2) lj)
    - 这里引入了一个缩放因子d^(-1/2)。
4. 最后解释了如何将这个框架应用到注意力机制：
    - 如果q和w来自线性操作：q_t = Qx_t, w_s = Kx_s
    - 那么注意力权重可以被视为一个分类问题
    - 具体来说，问题变成了"我们应该选择v_s = Vx_s的哪一行s"

> slides的第一个符号q应该写成Q感觉才正确。

![](https://files.mdnice.com/user/59/5a83371e-8c7e-46e3-9640-ae90ca6add51.png)

这张Slides讲解了Multi-head Attention（多头注意力机制），主要内容包括：
- 多头注意力机制可以同时处理多个注意力分类任务，每个任务的维度较小（用 d 表示）。这意味着模型能在多个“头”上并行计算，使得不同的注意力头可以关注输入的不同部分。
- 引用了 Vaswani 等人在Transformer论文《Attention is All You Need》中的描述：多头注意力机制允许模型同时关注输入的不同位置，且能够从不同的表示子空间中获取信息。如果仅使用单个注意力头，信息可能会被平均化，从而抑制这种能力。
- 多头注意力的每个头是完全独立的，因此它们的计算可以并行化，类似于批处理操作（batch），这种并行性使得计算非常高效，几乎没有复杂的相互依赖。

![](https://files.mdnice.com/user/59/1fc11340-e7d4-4692-80a5-042a631d5239.png)

这张Slides讨论了注意力机制的数学表示，以及如何在计算中进行并行化优化。包括：
1. **假设条件**
    - 假设当前只有一个注意力头（1 head），且没有批次维度（no batch dimension）。这种情况下，单个注意力头的计算是完全独立的（即“**embarrassingly parallel**”），可以在一个计算块中独立处理。
    - 因此，只要有效利用一个流处理器（streaming processor），我们可以仅使用一个块来完成这个计算。
2. **注意力机制的数学表示**：
    - **Query、Key、Value**（Q、K、V）三个矩阵的形状为 (N, d)，其中 N 是序列长度，d 是注意力头的维度。
    - **注意力矩阵 P 的计算公式**：
        - P = softmax(s * QK^T) ，其中s是缩放因子，用于调整大维度带来的数值不稳定问题。
    - **输出 O 的计算公式**：
        - O = PV，即注意力权重P和Value矩阵V相乘。
3. **并行化和优化问题**：
    - 提出了一个问题：如何并行化计算并在一次操作中完成？fuse
    - **P 矩阵是中间结果**，是否可以避免显式地存储它？这部分涉及数值线性代数和高性能计算（HPC）中的经典问题。

![](https://files.mdnice.com/user/59/1f498b11-ee6c-4bd7-80a3-0ff53efd699a.png)

这张slides开始引入 **Tiling Strategy（分块策略）**，用于优化注意力机制的计算，尤其是在计算矩阵乘法时通过分块来提高效率和内存利用率。

1. **目标**
    - 假设我们需要计算输出矩阵的一个元素$O[t, d]$，其中$t$是序列维度的token，$d$是隐藏层维度。
    - 计算这个元素时，需要：
        - Value矩阵的所有行$V[s, d]$ 对于每个 $s$。
        - Softmax权重 $softmax(I[t, s])$，其中$I[t, s]$是通过 Query 和 Key 的乘积得到的中间结果。
2. **中间结果的计算**
    - 为了计算Softmax的权重$I[t, s]$，需要通过矩阵乘法$I[t, s] = \sum_d{Q[t, d]K[s, d]}$，即Query 和 Key 的点积。
3. **忽略 Softmax 的情况**
    - 暂时忽略 Softmax 的计算，讨论如何在计算中使用 分块策略，即在时间步$t$和序列$s$上进行分块计算，并在t-tile上进行循环。
4. **分块计算的伪代码**
    - 外层循环：对t-tile进行循环
        - 将 Query 矩阵的 t-tile 加载到共享内存中，并初始化 $O[t, d] = 0$。
    - 内层循环：对s-tile进行循环
        - 将 Key 矩阵的 s-tile 和 Value 矩阵的 s-tile 加载到共享内存中
        - 计算 $I[t, s] = Q[t_{tile}]  K^T[s_{tile}]$，得到 Softmax 的输入项 $p[t_{tile}, s_{tile}]$。
        - 使用 Softmax 权重 $p$ 更新输出 $O[t, d]$, 即 $O[t, d] += p[t_{tile}, s_{tile}] V[s_{tile}]$
    - 将计算好的 $O[t, d]$ 写回

> 我觉得这张slides没有写清楚，结合Tiling来说这里计算的并不是t这个token的注意力结果，真正算的是前t个token的注意力结果。具体可以看下面2张图，这个Tiling的过程讲的很清晰，来源：https://zhuanlan.zhihu.com/p/669926191 

![](https://files.mdnice.com/user/59/291cb1f4-aa5d-400d-be00-06387bcc6bd9.png)

![](https://files.mdnice.com/user/59/0cfa0d85-61d5-47f9-a211-342f15a62382.png)


上面提到归一化前后的$P_{ij}$，在这张图解释：

![](https://files.mdnice.com/user/59/ba2f0338-b564-4080-bfce-2795fec77005.png)

![](https://files.mdnice.com/user/59/d86a3627-2eb1-4154-b8ee-5b1f82a45918.png)

然后视频里面下一张Slides就是讲解Safe Softmax的，内容和上面2张图讲的一样。

![](https://files.mdnice.com/user/59/8b775ead-ad71-4ca8-86df-a32fce9e0100.png)

![](https://files.mdnice.com/user/59/03cdd113-e340-4247-a808-d6d60d07a7f1.png)

这张Slides讲的是由于要做Tiling，所以我们要使用Online stabilized softmax，这部分作者讲得感觉比较一般，还是截图 https://zhuanlan.zhihu.com/p/669926191 这里的讲解来说明这个算法：

![](https://files.mdnice.com/user/59/52b99a3f-5472-4292-a824-f64f7cbf3e99.png)

> 这里需要注意上面图片中的[]为max操作。作者slides中的m和m_new就对应了上面讲解中的当前Tiling之前的局部最大值和当前Tiling上的最大值。

![](https://files.mdnice.com/user/59/96c72bc1-9caa-4e0e-afce-33386f52e2d2.png)

这张Slides讨论了一些与实现和优化相关的技术细节，主要内容为：
- 使用掩码会导致非矩形块布局
- Flash Attention v2在底层使用CUTLASS库，这将在后续讲座中详细介绍
- Flash Attention v2有一个非常大的C++文件需要编译 
- 分块选项基本上是64或128用于i和j，共有4种版本
- 本课程作者最初使用Numba开始实现，但对于这些tile大小，需要使用寄存器中的数组，因此需要转移到CUDA编程
- Flash Attention v2实现的限制因素：共享内存（shmem）大小和寄存器数量
- 其它：提到有一个很好的Triton模板，但使用它可能会减少一半的动手实现的乐趣

因此作者选择从零开始实现Flash Attention。

作者接下来show了一下根据 Flash Attention Forward Pass 实现的Naive代码，在看这个代码之前建议读一下 https://zhuanlan.zhihu.com/p/669926191 这篇文章的分块计算safe softmax这一小节。我把作者的代码摘抄到这里：

```python
import torch, math

N_inp = 64
N_out = 64
d = 128
Q = torch.randn(N_out, d)
K = torch.randn(N_inp, d)
V = torch.randn(N_inp, d)
O = torch.zeros(N_out, d)
L = torch.zeros(N_out, 1)

B_c = 16
B_r = 16
T_c = (N_inp + B_c - 1) // B_c
T_r = (N_out + B_r - 1) // B_r

scale_factor = 1 / math.sqrt(Q.size(-1))

# Q and O, L split into T_r; K, V in T_c blocks
for i in range(T_r):
    Q_i = Q[i * B_r: (i+1) * B_r]
    O_i = torch.zeros(B_r, d)
    L_i = torch.zeros(B_r, 1)
    m_i = torch.full((B_r, 1), -math.inf)
    last_m_i = m_i
    for j in range(T_c):
        K_j = K[j * B_c: (j + 1) * B_c]
        V_j = V[j * B_c: (j + 1) * B_c]
        S_i = scale_factor * (Q_i @ K_j.T)
        m_i = torch.maximum(m_i, S_i.max(dim=-1, keepdim=True).values)
        P_i = torch.exp(S_i - m_i)
        L_i = torch.exp(last_m_i - m_i) * L_i + P_i.sum(dim=-1, keepdim=True)
        O_i = torch.exp(last_m_i - m_i) * O_i + P_i @ V_j
        last_m_i = m_i
    O_i = (1.0 / L_i) * O_i
    L_i = m_i + torch.log(L_i)
    O[i * B_r: (i + 1) * B_r] = O_i
    L[i * B_r: (i + 1) * B_r] = L_i

expected = torch.nn.functional.scaled_dot_product_attention(Q[:, :], K[:, :], V[:, :])
print((O - expected).abs().max())
# tensor(1.1623e-06) 
```

我自己之前也根据Flash Attention的Forward Pass用PyTorch复现过这个算法，具体可以看我的这篇文章：https://zhuanlan.zhihu.com/p/684557290 ，我把这部分代码的解释补充到单独的《Flash Attention PyTorch naive实现讲解补充》大节中，避免打断课程笔记的连续性。

接着作者展示了一个Numba实现Flash Attention的例子，这个就略过了，一般我们不会用这个。然后这节课剩下的内容都是展示和瞎聊下面这个作者实现的Flash Attention的cuda代码。这里摘抄了下：

```python
cuda_src = r"""
constexpr int B_r = 16;
constexpr int B_c = 16;
constexpr int d = 128;
constexpr int o_per_thread_x = 1;
constexpr int o_per_thread_y = 128/32;

#define NEG_INFINITY __int_as_float(0xff800000)

extern "C" __global__
void silty_attn(float* out, float* out_l, float *K, float *V, float *Q, float scaling, int n, int T_r, int T_c) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    __shared__ float Q_i[B_r][d];
    __shared__ float K_j[B_c][d];
    __shared__ float V_j[B_c][d];
    
    __shared__ float s_i[B_r][B_c];

    float l_i[o_per_thread_x];
    float m_i[o_per_thread_x];
    float o_i[o_per_thread_x][o_per_thread_y];

    for (int ii = 0; ii < T_r; ii++) {
        for (int i = 0; i < o_per_thread_x; i++) {
            for (int dd = 0; dd < o_per_thread_y; dd++) {
                o_i[i][dd] = 0.f;
            }
            l_i[i] = 0.f;
            m_i[i] = NEG_INFINITY;
        }

        for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
            for (int dd = tid_x; dd < d; dd += blockDim.x) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
            }
        }

        for (int jj = 0; jj < T_c; jj++) {
            for (int i = tid_y; i < B_c; i += blockDim.y) {
                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    K_j[i][dd] = K[(jj * B_c) * d + dd];
                    V_j[i][dd] = V[(jj * B_c) * d + dd];
                }
            }

            __syncthreads();

            // S_ij = scale_factor * Q_i @ K_j.T
            for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
                for (int i = 0; i < o_per_thread_x; i++) {
                    float S_ij = 0.f;
                    for (int dd = 0; dd < d; dd++) {
                        S_ij += Q_i[ii][dd] * K_j[ii][dd];
                    }
                    s_i[ii][tid_x] = S_ij * scaling;
                }
            }

            __syncthreads();

            for (int ii = 0; ii < o_per_thread_x; ii++) {
                float m = m_i[ii];
                float l = l_i[ii];
                float last_m = m;

                for (int i = tid_y; i < B_c; i++) {
                    if (m < s_i[ii][tid_x]) {
                        m = s_i[ii][tid_x];
                    }
                }

                m_i[ii] = m;
                float l = exp(last_m - m) * l_i[ii];

                for (int dd = 0; dd < o_per_thread_y; dd++) {
                    o_i[ii][dd] *= exp(last_m - m);
                }

                for (int jj = 0; jj < o_per_thread_x; jj++) {
                    float S_ij = exp(s_i[ii][jj + blockDim.x * tid_x] - m);
                    l += S_ij;
                    for (int dd = 0; dd < o_per_thread_y; dd++) {
                        o_i[ii][dd] += S_ij * V_j[jj][dd + blockDim.y * tid_y];
                    }
                }
                l_i[ii] = l;
            }
        }

        for (int ii = 0; ii < o_per_thread_x; ii++) {
            for (int dd = 0; dd < o_per_thread_y; dd++) {
                out[(ii + blockDim.x * tid_x + i * B_r) * d + dd + blockDim.y * tid_y] = o_i[ii][dd] / l_i[ii];
            }
            out_l[ii + blockDim.x * tid_x + i * B_r] = 1 / l_i[ii];
        }
    }
}
"""

def fn():
    err = cuda.cuLaunchKernel(
        kernel,
        1,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        32, # block x dim
        32, # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        torch.cuda.current_stream().stream_id, # stream
        args.data_ptr(), # kernel arguments
        0,  # extra (ignore)
    )
fn()

```

作者这里实现的kernel感觉比较奇怪，特别是下标的混用bug估计会导致这个kernel存在正确性问题，此外这个Kernel里面每个线程具体负责哪些计算很难看得清楚，因此我在后面新增一节展示一下 https://github.com/tspeterkim/flash-attention-minimal 中对 Flash Attention 的极简 cuda 实现，这个实现非常清晰易懂。

## Flash Attention PyTorch naive实现讲解补充

FlashAttention V1通过分块计算的方法，将Q、K和V切块成很多小块，然后将这些切分后的小块放进SRAM（shared memory）中执行计算，最后再写回HBM中。算法流程如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/661e12b8d038edbd99c00048d6228c6d.png)

如果你想完全搞清楚这个伪代码的来龙去脉推荐看 https://zhuanlan.zhihu.com/p/669926191 这篇文章，但是从源码实现的角度来看，有了这个伪代码已经接近够了。只需要知道这些看似奇奇怪怪的公式是因为在分块遍历的时候每次计算的是一部分token，而自注意力机制要计算的最终结果是所有token间的，所以从局部到整体的更新就会用到在线的softmax算法以及在线更新最后的输出。这也是上面那堆复杂的公式由来。

我这里尝试用Python来模拟一下这个算法的流程，实现之后对Triton的实现会有帮助，因为从前面几节Triton的教程来看，相比于单纯的Python实现Triton kernel只是多了一个块级别的kernel启动过程而已。沿用上一节GPT2的设置，$N$和$d$分别设置成1024和64，那么Q，K，V的shape都是$(N, d)=(1024, 64)$，注意在FlashAttention里面就没有全局的S和P了。假设硬件是A100，A100的Shared Memory大小为192KB=196608B，那么可以计算出这里Flash Attention的分块大小，也就是上面的伪代码的第一行。

$B_c=M/4/64=768$，$B_r=min(768, 64)=64$。

然后伪代码的第2行初始化了一个全0的输出矩阵$O$，shape的大小也是$(N, d)=(1024, 64)$，同时初始化了一个$l$和$m$矩阵，维度大小都是$(N)$，不过$l$被初始化为全0矩阵，$m$被初始化为负无穷大。

接下来可以根据上面的参数直接计算出$T_r$和$T_c$，对应伪代码的第3行，$T_r=向上取整(N/B_r)=1024/64=16$，$T_c=向上取整(N/B_c)=1024/768=2$ 。

接下来的伪代码解析我直接放到下面的Python实现里，每一行代码都可以对应到上面的伪代码：

```python
import torch

N, d = 1024, 64  # 更新N和d的值

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

def standard_softmax_attention(Q, K, V):
    """
    执行标准的pytorch softmax和attention计算。
    """
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_softmax, expected_attention

def flash_attention(Q, K, V, B_r=64, B_c=768):
    """
    使用分块计算和在线softmax校正执行flash attention算法。
    """
    O = torch.zeros((N, d))  # 初始化输出矩阵，对应伪代码的第2行
    l = torch.zeros((N, 1))  # 存储softmax分母，对应伪代码的第2行
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值，对应伪代码的第2行

    # 对应伪代码的第5行，for 1<=j<=T_c，注意这里是把K, V分成了T_c=[N/B_c]块，每一块的大小是[B_c, d]这么大
    # 所以在python实现的时候就直接通过一个步长为B_c的循环来处理
    for j in range(0, N, B_c):
        # 下面三行就对应了伪代码的第6行，Load Kj, Vj from HBM to on-chip SRAM
        # 但是这里是单纯的 python 实现，我们不可能真的把这一块内存从HBM上放到SRAM上
        # 这里只是一个伪代码的逻辑说明，可以假装它做到了，因为在Triton里面真的可以在Python层做到。
        j_end = j + B_c
        Kj = K[j:j_end, :]
        Vj = V[j:j_end, :]

        # 对应伪代码的第7行，for 1<=i<T_r，注意这里是把Q分成了Tr=[N/B_r]块，每一块的大小是[B_r, d]这么大
        # 所以在python实现的时候就直接通过一个步长为B_r的循环来处理
        for i in range(0, N, B_r):
            i_end = i + B_r
            mi = m[i:i_end, :]
            li = l[i:i_end, :]
            Oi = O[i:i_end, :]
            Qi = Q[i:i_end, :]

            # 对应伪代码的第9行：on chip, compute Sij，Sij的形状是[B_r, B_c]
            Sij = Qi @ Kj.T
            # 对应伪代码的第10行
            mij_hat = torch.max(Sij, dim=1).values[:, None]
            pij_hat = torch.exp(Sij - mij_hat)
            lij_hat = torch.sum(pij_hat, dim=1)[:, None]

            # 对应伪代码的第11行求mi_new的操作，注意这里要对两个张量求整体的max，所以才有这个stack操作
            mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
            # 对应伪代码的第11行求li_new的操作
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat
            # 对应伪代码的第12行，更新O_i。这里容易有一个疑问，伪代码上有一个diag操作，为什么下面的实现忽略了
            # 这是因为这个diag是作用在vector上的，实际上是为了在伪代码上能对应上维度，而PyTorch的实现是自动
            # 支持张量广播机制的，所以这里可以直接计算。
            O_i = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

            # 对应伪代码的第13行，更新m_i，l_i，O_i。
            m[i:i_end, :] = mi_new
            l[i:i_end, :] = li_new
            O[i:i_end, :] = O_i

    return O

# 执行flash attention计算
flash_attention_output = flash_attention(Q_mat, K_mat, V_mat)

# 执行标准的pytorch softmax和attention计算
expected_softmax, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)

# 断言flash attention计算的结果与标准计算结果是否接近
assert torch.allclose(flash_attention_output, expected_attention), "error in flash attention calculation"
```

需要说明的是在上面的Attention Forward Pass流程中没有考虑到Dropout以及Mask的操作，如果考虑这两个操作整体的流程有一些变化，具体如Flash Attention V1的paper里的Algorithm2所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/807de9e1c7e23d1794205624adacd833.png)

相比于Algorithm1，多了Mask和Dropout的操作，其它的没有变化。


## Mini Flash Attention cuda 代码解读

本课程作者实现的 Flash Attention cuda kernel比较奇怪，这里推荐一个非常简单清晰的 Flash Attention 开源 cuda 实现：https://github.com/tspeterkim/flash-attention-minimal 。

```python
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    // 获取当前线程在块内的索引
    int tx = threadIdx.x;
    // 获取当前块在网格中的索引（batch和head索引）
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // 计算Q,K,V,O,l,m在全局内存中的偏移量 - 每个batch和head都不同
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // l和m的偏移量

    // 在共享内存中为Q,K,V,S定义空间
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // Qi, Kj, Vj的大小
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    // 外循环：遍历所有的K和V块
    for (int j = 0; j < Tc; j++) {

        // 将Kj, Vj加载到共享内存
        for (int x = 0; x < d; x++) {
            // Bc个线程，每个线程负责K的一列，注意转置之后，该矩阵列优先
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            // tx*d: 当前线程在Kj中的起始位置
            // x: 当前线程负责的列中的偏移量
            // qkv_offset: 当前batch和head的起始位置
            // tile_size * j: 当前K块的起始位置
            // (tx * d) + x: 当前K块内的具体位置

            // Bc个线程，每个线程负责V的一行，注意该矩阵行优先
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            // tx*d: 当前线程在Vj中的起始位置
            // x: 当前线程负责的行中的偏移量
            // qkv_offset: 当前batch和head的起始位置
            // tile_size * j: 当前V块的起始位置
            // (tx * d) + x: 当前V块内的具体位置
        }
        __syncthreads();  // 确保内部循环可以使用正确的Kj, Vj

        // 内循环：遍历所有的Q块
        for (int i = 0; i < Tr; i++)  {

            // 将Qi加载到共享内存，将l和m加载到寄存器
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                // tx*d: 当前线程在Qi中的起始位置
                // x: 当前线程负责的行中的偏移量
                // qkv_offset: 当前batch和head的起始位置
                // tile_size * i: 当前Q块的起始位置
                // (tx * d) + x: 当前Q块内的具体位置
            }

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            // lm_offset: 当前batch和head的m起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行

            float row_l_prev = l[lm_offset + (Br * i) + tx];
            // lm_offset: 当前batch和head的l起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行

            // 计算S = QK^T，并找出每行的最大值row_m
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    // Qi[(tx * d) + x]: 当前Q行的第x个元素
                    // Kj[(y * d) + x]: 当前K列的第x个元素
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                // Bc * tx: 当前线程在S中的起始位置
                // y: 当前列的偏移量

                if (sum > row_m)
                    row_m = sum;
            }

            // 计算P = exp(S - row_m)，并求每行的和row_l
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                // Bc * tx: 当前线程在S中的起始位置
                // y: 当前列的偏移量
                row_l += S[(Bc * tx) + y];
            }

            // 计算新的m和l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // 计算并写入O，更新l和m
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    // S[(Bc * tx) + y]: 当前S行的第y个元素
                    // Vj[(y * d) + x]: 当前V行的第x个元素
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
                // qkv_offset: 当前batch和head的O起始位置
                // tile_size * i: 当前O块的起始位置
                // (tx * d) + x: 当前O块内的具体位置
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            // lm_offset: 当前batch和head的m起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行
            l[lm_offset + (Br * i) + tx] = row_l_new;
            // lm_offset: 当前batch和head的l起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行
        }
        __syncthreads();  // 防止线程在内部循环中使用错误的Kj, Vj
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: 动态确定Bc, Br
    const int Bc = 32; const int Br = 32;

    // 获取输入张量的维度
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    // 计算块的数量
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // 在GPU内存中初始化O, l, m
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // 计算每个块需要的共享内存大小
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    // 设置网格和块的维度
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // 每个块Bc个线程

    // 启动kernel
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
```

这个代码写得很好，基本还原了 Flash Attention Forward Pass 的流程并且对shm和register的应用也和本节课作者所讲内容一致。

这个kernel在Batch和Num-heads方向上并行，`dim3 grid_dim(B, nh);`。然后每个Block处理一个Batch的一个Head的Attention计算。每个块开了Bc=Br=32个线程，因此每个线程负责每个 S:(Br, Bc) 中一行的计算（肯定有2个Tc和Tr的for循环，因此每个线程实际上会负责一共 Tc * Tr个S），每个thread访问的Qi对应行的起始地址为 tx*d，其中tx就是threadIdx.x，d就是每个注意力头的大小。

## 总结
这节课其实内容很少，大部分时间都是作者吹水，但确实也很少有人可以几十分钟就讲清楚Flash Attention的原理+代码实现，想深入了解 Flash Attention 的朋友可以关注我在 https://github.com/BBuf/how-to-optim-algorithm-in-cuda 这里收集的一些讲解 Flash Attention 的资料，然后可以自己动手实现一个 PyTorch 或者 cuda/triton 版本的代码。



