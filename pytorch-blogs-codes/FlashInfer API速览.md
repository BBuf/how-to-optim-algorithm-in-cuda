# 安装

## Python 包

FlashInfer 是一个 Python 包，基于 `PyTorch <https://pytorch.org/>`_ 构建，可以轻松集成到您的 Python 应用程序中。

### 先决条件

- 操作系统：仅限 Linux

- Python：3.8, 3.9, 3.10, 3.11, 3.12

- PyTorch：2.2/2.3/2.4，支持 CUDA 11.8/12.1/12.4（仅限 PyTorch 2.4）

  - 使用 ``python -c "import torch; print(torch.version.cuda)"`` 检查您的 PyTorch CUDA 版本。

- 支持的 GPU 架构：``sm75``, ``sm80``, ``sm86``, ``sm89``, ``sm90``。

### 快速开始

安装 FlashInfer 最简单的方法是通过 pip：

```shell
.. tabs::

    .. tab:: PyTorch 2.4

        .. tabs::

            .. tab:: CUDA 12.4

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4/

    .. tab:: PyTorch 2.3

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.3/

    .. tab:: PyTorch 2.2

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.2/

    .. tab:: PyTorch 2.1

        Since FlashInfer version 0.1.2, support for PyTorch 2.1 has been ended. Users are encouraged to upgrade to a newer
        PyTorch version or :ref:`compile FlashInfer from source code. <compile-from-source>` .

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.1/
```

### 从源代码编译

在某些情况下，您可能希望从源代码编译 FlashInfer 以尝试主分支中的最新功能，或根据您的特定需求自定义库。您可以按照以下步骤从源代码编译 FlashInfer：

1. 克隆 FlashInfer 仓库：

```shell
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
```

2. 确保您已安装支持 CUDA 的 PyTorch。您可以通过运行以下命令来检查 PyTorch 版本和 CUDA 版本：

```shell
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

3. 安装 Ninja 构建系统：

```shell
pip install ninja
```

4. 编译 FlashInfer:

```shell
cd flashinfer
pip install -e . -v
```

## C++ API

FlashInfer 是一个仅依赖于 CUDA/C++ 标准库的头文件库，可以直接集成到您的 C++ 项目中，无需安装。

您可以查看我们的 `单元测试和基准测试 <https://github.com/flashinfer-ai/flashinfer/tree/main/src>`_ 以了解如何使用我们的 C++ API。

> `3rdparty` 目录中的 `nvbench` 和 `googletest` 依赖项仅用于编译单元测试和基准测试，不是库本身所必需的。

### 编译基准测试和单元测试

要编译 C++ 基准测试（使用 `nvbench <https://github.com/NVIDIA/nvbench>`_）和单元测试，您可以按照以下步骤操作：

1. 克隆 FlashInfer 仓库：

```shell
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
```

2. 检查 conda 是否已安装（如果您已通过其他方式安装了 cmake 和 ninja，可以跳过此步骤）：

```shell
conda --version
```

如果未安装 conda，您可以按照 `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ 或 `miniforge <https://github.com/conda-forge/miniforge>`_ 网站上的说明进行安装。

2. 安装 CMake 和 Ninja 构建系统：

```shell
conda install cmake ninja
```

3. 创建构建目录并复制配置文件

```shell       
mkdir -p build
cp cmake/config.cmake build/  # you can modify the configuration file if needed
```

4. 编译基准测试和单元测试：
   
```shell
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

----------------------------------------------------------------------


# 注意力状态和递归注意力

FlashInfer 引入了 **注意力状态** 的概念，该概念完全描述了Query和一组键值对之间的注意力。我们进一步定义了 **合并** 操作符，该操作符作用于 **注意力状态**。这个合并操作符通过允许递归合并注意力状态，促进了完整注意力的计算。

假设我们定义 $s_i = \mathbf{q}\mathbf{k}_i^T$ 为Query $\mathbf{q}$ 和键 $\mathbf{k}_i$ 之间的预 softmax 注意力分数。索引 $i$ 处的自注意力分数可以推广到索引集 $I$：

$$
s(I)=\log\left(\sum_{i\in I}\exp\left(s_i\right)\right)
$$

我们还可以将索引 $i$ 处的值推广到索引集 $I$：

$$
\mathbf{v}(I) = \sum_{i\in I}\textrm{softmax}(s_i) \mathbf{v}_i = \frac{\sum_{i\in I}\exp\left(s_i\right)\mathbf{v}_i}{\exp(s(I))}
$$

$softmax$ 函数仅限于索引集 $I$。注意，$\mathbf{v}(\{1,2,\cdots, n\})$ 是整个序列的自注意力输出。
索引集 $i$ 的 *注意力状态* 可以定义为一个元组 $(s(I), \mathbf{v}(I))$，然后我们可以定义两个注意力状态的二元 **合并** 操作符 $\oplus$（在实际应用中，为了保证数值稳定性，我们会减去 $s$ 的最大值，这里为了简化省略了这些步骤）：

$$
\begin{bmatrix}\mathbf{v}(I\cup J)\\s(I\cup J)\end{bmatrix}=\begin{bmatrix}\mathbf{v}(I)\\s(I)\end{bmatrix}\oplus\begin{bmatrix}\mathbf{v}(J)\\s(J)\end{bmatrix}=\begin{bmatrix} \frac{\mathbf{v}(I)\exp(s(I)) + \mathbf{v}(J)\exp(s(J))}{\exp(s(I)) + \exp(s(J))} \\  \log(\exp(s(I)) + \exp(s(J))) \end{bmatrix}
$$

**合并** 操作符可以推广到任意数量的注意力状态输入：

$$
\begin{bmatrix}\mathbf{v}(\bigcup_{i=1}^{n}I_i) \\ s(\bigcup_{i=1}^{n}I_i) \end{bmatrix} = \bigoplus_{i=1}^{n}\begin{bmatrix}\mathbf{v}(I_i) \\ s(I_i)\end{bmatrix} = \begin{bmatrix} \sum_{i=1}^{n} \textrm{softmax}(s(I_i))\mathbf{v}(I_i) \\ \log(\sum_{i=1}^{n} \exp (s(I_i))) \end{bmatrix}
$$

上述 n-ary 合并操作符与二元合并操作符一致，我们可以证明该操作符是 *交换的* 和 *结合的*。通过合并索引子集的注意力状态，有多种方法可以得到整个序列的注意力状态，最终结果在数学上是等价的：

![](https://files.mdnice.com/user/59/7a03292e-0e88-4f53-9804-ba2b5471ef5c.png)

> 通用分数 $s$ 也被称为 对数和指数（``lse`` 为简写）。

## 应用

注意，$\oplus$ 操作符是 **交换的** 和 **结合的**，这意味着我们可以将自注意力计算的一部分 KV 安全地卸载到不同的设备上，并且可以 **以任何顺序** **合并** 结果。

到目前为止，FlashInfer 在递归形式的自注意力中有一些有趣的应用：

**共享前缀批量解码**
  许多 LLM 应用涉及带有共享长提示的批量解码，FlashInfer 将整个 KV Cache 的注意力分解为共享前缀注意力和唯一后缀注意力。
  这种分解使得这些组件可以卸载到不同的 kernel 实现中，从而在长上下文和大批次大小的情况下实现了显著的 30 倍加速。
  这种分解在长上下文设置中将操作加速了 30 倍。
  有关此应用的更多详细信息，请参阅 `我们的博客文章 <https://flashinfer.ai/2024/01/08/cascade-inference.html>`_，
  以及 https://docs.flashinfer.ai/api/python/cascade.html#api-cascade-attention 了解如何在 FlashInfer 中使用此功能。

**KV 序列并行性**
  对于长上下文 LLM 推理/服务，每个 GPU 的批处理大小和头数受到 GPU 内存的限制，
  默认的并行策略无法使用 GPU 中的所有 SM，从而导致性能不佳。
  受 `Split-K <https://github.com/NVIDIA/cutlass/blob/8825fbf1efebac973d96730892919ab241b755bb/media/docs/efficient_gemm.md#parallelized-reductions>`_ 技巧的启发，
  FlashInfer 将 KV 序列维度进行分区，并将注意力计算分派到不同的线程块，并在第二步中合并它们。这一想法也在 Flash-Decoding 中被提出，您可以
  查看他们出色的 `博客文章 <https://crfm.stanford.edu/2023/10/12/flashdecoding.html>`_ 以获取可视化和更多详细信息。

## 相关 API

FlashInfer 提供了多个 API 以促进递归注意力计算：

- https://docs.flashinfer.ai/api/python/cascade.html#api-merge-states 定义了用于合并注意力状态的操作符。
- https://docs.flashinfer.ai/api/python/prefill.html#apiprefill 和 https://docs.flashinfer.ai/api/python/decode.html#apidecode 定义了返回注意力状态的操作符（带有后缀 ``_return_lse`` 的 API 返回注意力输出 $v$ 和分数 $s$）。

----------------------------------------------------------------------------

# FlashInfer 中的 KV-Cache 布局

## 布局：NHD/HND

FlashInfer 为 KV-Cache 的最后三个维度提供了两种布局：``NHD`` 和 ``HND``：

- ``NHD``：最后三个维度组织为 ``(seq_len, num_heads, head_dim)``。
- ``HND``：最后三个维度组织为 ``(num_heads, seq_len, head_dim)``。

``NHD`` 布局更自然，因为它与 $xW_k$ 和 $xW_v$ 的输出一致，无需转置。``HND`` 布局在 KV-Cache 使用低精度数据类型（例如 fp8）时对 GPU 实现更友好。
在实际应用中，我们在这两种布局之间没有观察到显著的性能差异，因此我们优先选择 ``NHD`` 布局以提高可读性。FlashInfer 在这两种布局上都实现了注意力kernel，并提供了一个选项来选择它们（默认为 ``NHD``）。

## Ragged Tensor

在批量推理/服务中，输入序列长度可能在不同的样本之间有所不同。当不需要改变序列长度（例如在Prefill阶段），我们可以使用具有单个可变长度维度的 ``RaggedTensor`` 来存储 KV Cache 中的Key/Value张量：

![](https://files.mdnice.com/user/59/88eecbb5-9d02-4b80-8b20-edd24a472548.png)


所有请求的键（或值）被打包到一个没有填充的单个 ``data`` 张量中，我们使用一个 ``indptr`` 数组（``num_requests+1`` 个元素，第一个元素始终为零）来存储每个请求的可变序列长度信息（``indptr[i+1]-indptr[i]`` 是请求 ``i`` 的序列长度），当布局为 ``NHD`` 时，``data`` 张量的形状为 ``(indptr[-1], num_heads, head_dim)``。

我们可以使用 ``data[indptr[i]:indptr[i+1]]`` 来切片请求 ``i`` 的键（或值）。

### FlashInfer APIs

FlashInfer 提供了 `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` 来计算存储在 ragged tensor 中的 Query 与存储在 ragged KV Cache 中的 Key/Value 之间的Prefill注意力。


## Mask Layout (2D Ragged Tensor)

上述的 Ragged Tensor 可以推广到多个“ragged”维度。例如，当批量大小大于 1 时，FlashInfer 中的注意力掩码是一个 2D ragged tensor：

![](https://files.mdnice.com/user/59/942262b0-0690-452f-9949-524a0139ddae.png)

当请求数量大于 1 时，不同的请求可能具有不同的Query长度和 kv 长度。为了避免填充，我们使用 2D ragged tensor 来存储注意力掩码。输入的 ``qo_indptr`` 和 ``kv_indptr`` 数组（长度均为 ``num_requests+1``）用于存储每个请求的可变序列长度信息，``qo_indptr[i+1]-qo_indptr[i]`` 是请求 ``i`` 的Query长度（``qo_len[i]``），``kv_indptr[i+1]-kv_indptr[i]`` 是请求 ``i`` 的 kv 长度（``kv_len[i]``）。

所有请求的掩码数组被展平（Query作为第一维度，kv 作为最后一维）并连接成一个 1D 数组：``mask_data``。FlashInfer 会隐式创建一个 ``qk_indptr`` 数组来存储每个请求的掩码在展平的掩码数组中的起始偏移量：``qk_indptr[1:] = cumsum(qo_len * kv_len)``。

``mask_data`` 的形状为 ``(qk_indptr[-1],)``，我们可以使用 ``mask_data[qk_indptr[i]:qk_indptr[i+1]]`` 来切片请求 ``i`` 的展平掩码。

为了节省内存，我们可以进一步将布尔展平的布尔掩码数组打包成位打包数组（每个元素 1 位，8 个元素打包成一个 `uint8`），使用“little”位序（详见 `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_）。FlashInfer 接受布尔掩码和位打包掩码。如果提供布尔掩码，FlashInfer 会将其内部打包成 bit-packed 数组。

### FlashInfer APIs

`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` 和 `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` 允许用户在 `begin_forward` 函数中指定 ``qo_indptr``、``kv_indptr`` 和自定义注意力掩码 ``custom_mask``，掩码数据将在注意力 kernel 中的 softmax 之前（以及 softmax 缩放之后）添加到注意力分数中。

`flashinfer.quantization.packbits` 和 `flashinfer.quantization.segment_packbits` 是用于将布尔掩码打包成 bit-packed 数组的工具函数。

## Page Table 布局

当 KV-Cache 是动态的（例如在 append 或 decode 阶段），打包所有Key/Value是不高效的，因为每个请求的序列长度会随时间变化。`vLLM <https://arxiv.org/pdf/2309.06180.pdf>`_ 
提出将 KV-Cache 组织为Page Table。在 FlashInfer 中，我们将 Page Table 视为一个块稀疏矩阵（每个使用的Page 可以视为块稀疏矩阵中的一个非零块）并使用 `CSR 格式 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>` 来索引 KV-Cache 中的Page。

![](https://files.mdnice.com/user/59/d14c4007-6568-4039-9733-ac5fd069ecb7.png)

对于每个请求，我们记录其 ``page_indices`` 和 ``last_page_len``，分别跟踪该请求使用的Page 和最后一个Page 中的条目数量。请求 ``i`` 的 KV 序列长度为 ``page_size * (len(page_indices[i]) - 1) + last_page_length[i]``。

> 每个请求的 ``last_page_len`` 必须大于零，并且小于或等于 ``page_size``。

总体的 ``kv_indptr`` 数组（长度为 ``num_requests+1``）可以计算为：``[0, len(page_indices[0]), len(page_indices[0])+len(page_indices[1]), ...]``。总体的 ``kv_page_indices`` 数组（长度为 ``kv_indptr[-1]``）是所有请求的 ``page_indices`` 的连接。总体的 ``kv_last_page_lens`` 数组（长度为 ``num_requests``）是所有请求的 ``last_page_length`` 的连接。``kv_data`` 张量可以是一个 5-D 张量或一个 4-D 张量的元组，当存储为单个张量时，``kv_data`` 的形状为：

```python
(max_num_pages, 2, page_size, num_heads, head_dim) # NHD layout
(max_num_pages, 2, num_heads, page_size, head_dim) # HND layout
```

当存储为张量元组时，``kv_data = (k_data, v_data)``，每个张量的形状为：

```python
(max_num_pages, page_size, num_heads, head_dim) # NHD layout
(max_num_pages, num_heads, page_size, head_dim) # HND layout
```

其中，``max_num_pages`` 是所有请求使用的最大Page 数，``page_size`` 是每个Page 中容纳的 token 数量。在单个张量存储中，``2`` 表示 K/V（第一个用于Key，第二个用于Value）。

### FlashInfer APIs

`flashinfer.page.append_paged_kv_cache` 可以将一批Key/Value（存储为 ragged tensors）追加到分页的 KV-Cache 中（这些追加的Key/Value的Page 必须在调用此 API 之前分配）。

`flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` 和 `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` 实现了在存储为 ragged tensors 的Query和存储在分页 KV-Cache 中的Key/Value之间的解码注意力和预填充/追加注意力。


## 多级级联推理数据布局

当使用多级 `级联推理 <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_ 时，Query和输出存储在 ragged tensors 中，所有级别的 KV-Cache 存储在一个统一的分页 KV-Cache 中。每个级别都有一个唯一的 ``qo_indptr`` 数组，该数组是子树中累积的要追加的 token 数的前缀和，以及 ``kv_page_indptr``、``kv_page_indices`` 和 ``kv_last_page_len``，这些数组的语义与 Page Table 布局 部分中的相同。下图介绍了如何为 8 个请求构建这些数据结构，我们将这些请求的 KV-Cache 视为 3 个级别以实现前缀重用：

![](https://files.mdnice.com/user/59/1d076e3c-c2c1-4378-87cd-aa32314e5368.png)

请注意，我们不需要为每个级别更改 ragged query/output 张量或分页 kv-cache 的数据布局。所有级别共享相同的基础数据布局，但我们使用不同的 ``qo_indptr`` / ``kv_page_indptr`` 数组，以便以不同的方式查看它们。

### FlashInfer APIs
FlashInfer 提供 `flashinfer.cascade.MultiLevelCascadeAttentionWrapper` 用于计算级联注意力。

## FAQ

**FlashInfer 如何管理 KV-Cache？**

  FlashInfer 本身不负责管理Page Table（例如弹出和分配新Page 等），而是将策略留给用户：不同的服务引擎可能有不同的策略来管理Page Table。FlashInfer 仅负责计算存储在 KV-Cache 中的Query和Key/Value之间的注意力。

# FlashInfer API

## flashinfer.decode

### Single Request Decoding

- `single_decode_with_kv_cache(q, k, v[, ...])`: 使用 kv cache 对单个请求进行解码注意力，返回注意力输出。

`def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
) -> torch.Tensor:`

单请求解码注意力，使用 kv cache ，返回注意力输出。

```python

参数

q : torch.Tensor
    查询张量，形状：``[num_qo_heads, head_dim]``。
k : torch.Tensor
    键张量，形状：如果 `kv_layout` 为 ``NHD``，则为 ``[kv_len, num_kv_heads, head_dim]``，如果 `kv_layout` 为 ``HND``，则为 ``[num_kv_heads, kv_len, head_dim]``。
v : torch.Tensor
    值张量，形状：如果 `kv_layout` 为 ``NHD``，则为 ``[kv_len, num_kv_heads, head_dim]``，如果 `kv_layout` 为 ``HND``，则为 ``[num_kv_heads, kv_len, head_dim]``。
kv_layout : str
    输入键/值张量的布局，可以是 ``NHD`` 或 ``HND``。
pos_encoding_mode : str
    在注意力内核中应用的位置编码，可以是 ``NONE``/``ROPE_LLAMA``（LLAMA 风格的旋转编码）/``ALIBI``。默认为 ``NONE``。
use_tensor_cores: bool
    是否使用张量核心进行计算。对于大组大小的分组查询注意力，使用张量核心会更快。默认为 ``False``。
q_scale : Optional[float]
    查询的 fp8 输入的校准比例，如果未提供，将设置为 ``1.0``。
k_scale : Optional[float]
    键的 fp8 输入的校准比例，如果未提供，将设置为 ``1.0``。
v_scale : Optional[float]
    值的 fp8 输入的校准比例，如果未提供，将设置为 ``1.0``。
window_left : int
    注意力窗口的左（包含）窗口大小，当设置为 ``-1`` 时，窗口大小将设置为序列的全长。默认为 ``-1``。
logits_soft_cap : Optional[float]
    注意力对数的软上限值（用于 Gemini、Grok 和 Gemma-2 等），如果未提供，将设置为 ``0``。如果大于 0，对数将根据公式进行上限：
    $\text{logits_soft_cap} \times \mathrm{tanh}(x / \text{logits_soft_cap})$，
    其中 $x$ 是输入对数。
sm_scale : Optional[float]
    softmax 的比例，如果未提供，将设置为 ``1 / sqrt(head_dim)``。
rope_scale : Optional[float]
    RoPE 插值中使用的比例，如果未提供，将设置为 ``1.0``。
rope_theta : Optional[float]
    RoPE 中使用的 theta，如果未提供，将设置为 ``1e4``。

返回
-------
torch.Tensor
    注意力输出，形状：``[num_qo_heads, head_dim]``

示例
--------

import torch
import flashinfer
kv_len = 4096
num_qo_heads = 32
num_kv_heads = 32
head_dim = 128
q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
o = flashinfer.single_decode_with_kv_cache(q, k, v)
o.shape
torch.Size([32, 128])

Note
----
The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
not equal to ``num_kv_heads``, the function will use
`grouped query attention <https://arxiv.org/abs/2305.13245>`_.

```

### Batch Decoding

`class flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, use_tensor_cores: bool = False, paged_kv_indptr_buffer: torch.Tensor | None = None, paged_kv_indices_buffer: torch.Tensor | None = None, paged_kv_last_page_len_buffer: torch.Tensor | None = None)`

用于批量请求的Paged KV Cache解码注意力的包装类（首次在 vLLM 中提出）。

例子

```python
import torch
import flashinfer
num_layers = 32
num_qo_heads = 64
num_kv_heads = 8
head_dim = 128
max_num_pages = 128
page_size = 16
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)
batch_size = 7
# kv_page_indices: [0, 1, 2, ..., 128]
kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
kv_page_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
) # 注意，这里是前缀的关系，每个请求的 Paged Table 数是[17, 12, 15, 4, 18, 34, 28]
# 1 <= kv_last_page_len <= page_size
kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)
kv_cache_at_layer = [
    torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]
# create auxiliary data structures for batch decode attention
decode_wrapper.plan(
    kv_page_indptr,
    kv_page_indices,
    kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    pos_encoding_mode="NONE",
    data_type=torch.float16
)
outputs = []
for i in range(num_layers):
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    kv_cache = kv_cache_at_layer[i]
    # compute batch decode attention, reuse auxiliary data structures for all layers
    o = decode_wrapper.run(q, kv_cache)
    outputs.append(o)

print(outputs[0].shape)
# torch.Size([7, 64, 128])
```

> 为了加速计算，FlashInfer 的批量解码注意力创建了一些辅助数据结构，这些数据结构可以在多个批量解码注意力调用中重用（例如，不同的 Transformer 层）。这个包装类管理这些数据结构的生命周期。

`__init__(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, use_tensor_cores: bool = False, paged_kv_indptr_buffer: torch.Tensor | None = None, paged_kv_indices_buffer: torch.Tensor | None = None, paged_kv_last_page_len_buffer: torch.Tensor | None = None) → None`

构造 `BatchDecodeWithPagedKVCacheWrapper`.

```python
Parameters
    float_workspace_buffer : torch.Tensor
        用户预留的浮点工作区缓冲区，用于存储 split-k 算法中的中间注意力结果。推荐大小为 128MB，工作区缓冲区的设备应与输入张量的设备相同。

    kv_layout : str
        输入 k/v 张量的布局，可以是 ``NHD`` 或 ``HND``。

    use_cuda_graph : bool
        是否启用 CUDAGraph 用于批量解码注意力，如果启用，辅助数据结构将存储在提供的缓冲区中。当启用 CUDAGraph 时，此包装器的生命周期内 ``batch_size`` 不能改变。

    use_tensor_cores : bool
        是否使用张量核心进行计算。对于大型组查询注意力，使用张量核心会更快。默认为 ``False``。

    indptr_buffer : Optional[torch.Tensor]
        用户预留的 GPU 缓冲区，用于存储 Paged KV Cache 的 indptr，缓冲区的大小应为 ``[batch_size + 1]``。
        仅在 ``use_cuda_graph`` 为 ``True`` 时需要。

    indices_buffer : Optional[torch.Tensor]
        用户预留的 GPU 缓冲区，用于存储 Paged KV Cache 的页索引，缓冲区应足够大以存储此包装器生命周期内的最大页索引数（``max_num_pages``）。
        仅在 ``use_cuda_graph`` 为 ``True`` 时需要。

    last_page_len_buffer : Optional[torch.Tensor]
        用户预留的 GPU 缓冲区，用于存储最后一页的条目数，缓冲区的大小应为 ``[batch_size]``。
        仅在 ``use_cuda_graph`` 为 ``True`` 时需要。
```

`plan(indptr: torch.Tensor, indices: torch.Tensor, last_page_len: torch.Tensor, num_qo_heads: int, num_kv_heads: int, head_dim: int, page_size: int, pos_encoding_mode: str = 'NONE', window_left: int = -1, logits_soft_cap: float | None = None, data_type: str | torch.dtype = 'float16', q_data_type: str | torch.dtype | None = None, sm_scale: float | None = None, rope_scale: float | None = None, rope_theta: float | None = None) → None`

Plan batch decode for given problem specification.

```python
Parameters
    indptr : torch.Tensor
         Paged KV Cache 的 indptr，形状：``[batch_size + 1]``
    indices : torch.Tensor
         Paged KV Cache 的页索引，形状：``[qo_indptr[-1]]``
    last_page_len : torch.Tensor
        每个请求在 Paged KV Cache 中最后一页的条目数，形状：``[batch_size]``
    num_qo_heads : int
        查询/输出头的数量
    num_kv_heads : int
        键/值头的数量
    head_dim : int
        头的维度
    page_size : int
         Paged KV Cache 的页大小
    pos_encoding_mode : str
        在注意力内核中应用的位置编码，可以是
        ``NONE``/``ROPE_LLAMA``（LLAMA 风格的旋转嵌入）/``ALIBI``。
        默认为 ``NONE``。
    window_left : int
        注意力窗口的左（包含）窗口大小，当设置为 ``-1`` 时，窗口大小将设置为序列的全长。默认为 ``-1``。
    logits_soft_cap : Optional[float]
        注意力 logits 的软上限值（用于 Gemini、Grok 和 Gemma-2 等），如果未提供，将设置为 ``0``。如果大于 0，logits 将根据公式进行上限：
        $\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})$，
        其中 $x$ 是输入 logits。
    data_type : Union[str, torch.dtype]
         Paged KV Cache 的数据类型。默认为 ``float16``。
    q_data_type : Optional[Union[str, torch.dtype]]
        查询张量的数据类型。如果为 None，将设置为
        ``data_type``。默认为 ``None``。

    注意
    ----
    在任何 `run` 或 `run_return_lse` 调用之前，应调用 `plan` 方法，辅助数据结构将在此次调用中创建并缓存以供多次运行调用。

    ``num_qo_heads`` 必须是 ``num_kv_heads`` 的倍数。如果 ``num_qo_heads``
    不等于 ``num_kv_heads``，函数将使用`分组查询注意力 <https://arxiv.org/abs/2305.13245>`_。
```

`reset_workspace_buffer(float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor) → None`

Reset the workspace buffer.

```python
Parameters
    float_workspace_buffer : torch.Tensor
        新的浮点工作区缓冲区，其设备应与输入张量的设备相同。

    int_workspace_buffer : torch.Tensor
        新的整数工作区缓冲区，其设备应与输入张量的设备相同。
```

`run(q: torch.Tensor, paged_kv_cache: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, return_lse: bool = False) → torch.Tensor | Tuple[torch.Tensor, torch.Tensor]`

Compute batch decode attention between query and paged kv cache.

```python
Parameters
    q : torch.Tensor
        查询张量，形状：``[batch_size, num_qo_heads, head_dim]``
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
         Paged KV Cache ，存储为张量元组或单个张量：

        * 一个 4-D 张量元组 ``(k_cache, v_cache)``，每个张量的形状为：
            ``[max_num_pages, page_size, num_kv_heads, head_dim]`` 如果 `kv_layout` 是 ``NHD``,
            和 ``[max_num_pages, num_kv_heads, page_size, head_dim]`` 如果 `kv_layout` 是 ``HND``。

        * 一个 5-D 张量，形状为：
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` 如果
            `kv_layout` 是 ``NHD``，和
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` 如果
            `kv_layout` 是 ``HND``。其中 ``paged_kv_cache[:, 0]`` 是键缓存，``paged_kv_cache[:, 1]`` 是值缓存。

    q_scale : Optional[float]
        查询的校准比例，对于 fp8 输入，如果未提供，将设置为 ``1.0``。
    k_scale : Optional[float]
        键的校准比例，对于 fp8 输入，如果未提供，将设置为 ``1.0``。
    v_scale : Optional[float]
        值的校准比例，对于 fp8 输入，如果未提供，将设置为 ``1.0``。
    return_lse : bool
        是否返回注意力分数的 logsumexp，默认为 ``False``。

    返回
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        如果 `return_lse` 是 ``False``，返回注意力输出，形状：``[batch_size, num_qo_heads, head_dim]``。
        如果 `return_lse` 是 ``True``，返回一个包含两个张量的元组：

        * 注意力输出，形状：``[batch_size, num_qo_heads, head_dim]``
        * 注意力分数的 logsumexp，形状：``[batch_size, num_qo_heads]``。
```

`class flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(workspace_buffer: torch.Tensor, indptr_buffer: torch.Tensor, indices_buffer: torch.Tensor, last_page_len_buffer: torch.Tensor, kv_layout: str = 'NHD', use_tensor_cores: bool = False)`

与 CUDAGraph 兼容的解码注意力包装类，用于处理 Paged KV Cache （首次在 `vLLM <https://arxiv.org/abs/2309.06180>`_ 中提出）的批量请求。

请注意，此包装类可能不如 `BatchDecodeWithPagedKVCacheWrapper` 高效，因为为了适应 CUDAGraph 的要求，我们不会为不同的批量大小、序列长度等分派不同的kernel。

> plan() 方法无法被 CUDAGraph 捕获。

Constructor of `BatchDecodeWithPagedKVCacheWrapper`.

```python
Parameters
    workspace_buffer : torch.Tensor
        用户预留的 GPU 工作区缓冲区，用于存储辅助数据结构，建议大小为 128MB，工作区缓冲区的设备应与输入张量的设备相同。

    indptr_buffer : torch.Tensor
        用户预留的 GPU 缓冲区，用于存储分页 kv cache 的 indptr，应足够大以存储此包装器生命周期内的最大批量大小（``[max_batch_size + 1]``）的 indptr。

    indices_buffer : torch.Tensor
        用户预留的 GPU 缓冲区，用于存储分页 kv cache 的页索引，应足够大以存储此包装器生命周期内的最大页索引数（``max_num_pages``）。

    last_page_len_buffer : torch.Tensor
        用户预留的 GPU 缓冲区，用于存储每页的条目数，应足够大以存储此包装器生命周期内的最大批量大小（``[max_batch_size]``）。

    use_tensor_cores : bool
        是否使用张量核心进行计算。对于大组大小的分组查询注意力，使用张量核心会更快。默认为 ``False``。

    kv_layout : str
        输入 k/v 张量的布局，可以是 ``NHD`` 或 ``HND``。
```

## flashinfer.prefill

Attention kernels for prefill & append attention in both single request and batch serving setting.

### Single Request Prefill/Append Attention

- `single_prefill_with_kv_cache(q, k, v[, ...])`: 单请求的预填充/追加注意力，使用 kv cache ，返回注意力输出。

```python
Parameters
    参数
    ----------
    q : torch.Tensor
        查询张量，形状：``[qo_len, num_qo_heads, head_dim]``。
    k : torch.Tensor
        键张量，形状：``[kv_len, num_kv_heads, head_dim]`` 如果 `kv_layout` 是 ``NHD``，或 ``[num_kv_heads, kv_len, head_dim]`` 如果 `kv_layout` 是 ``HND``。
    v : torch.Tensor
        值张量，形状：``[kv_len, num_kv_heads, head_dim]`` 如果 `kv_layout` 是 ``NHD``，或 ``[num_kv_heads, kv_len, head_dim]`` 如果 `kv_layout` 是 ``HND``。
    custom_mask : Optional[torch.Tensor]
        自定义布尔掩码张量，形状：``[qo_len, kv_len]``。
        掩码张量中的元素应为 ``True`` 或 ``False``，其中 ``False`` 表示注意力矩阵中对应的元素将被屏蔽。

        当提供 `custom_mask` 且未提供 `packed_custom_mask` 时，函数会将自定义掩码张量打包成 1D 打包掩码张量，这会引入额外的开销。
    packed_custom_mask : Optional[torch.Tensor]
        1D 打包的 uint8 掩码张量，如果提供，`custom_mask` 将被忽略。
        打包的掩码张量由 :func:`flashinfer.quantization.packbits` 生成。
    causal : bool
        是否对注意力矩阵应用因果掩码。
        仅在未提供 `custom_mask` 时有效。
    kv_layout : str
        输入 k/v 张量的布局，可以是 ``NHD`` 或 ``HND``。
    pos_encoding_mode : str
        在注意力内核中应用的位置编码，可以是 ``NONE``/``ROPE_LLAMA``（LLAMA 风格的旋转嵌入）/``ALIBI``。
        默认为 ``NONE``。
    allow_fp16_qk_reduction : bool
        是否使用 f16 进行 qk reduction（更快但精度略有损失）。
    window_left : int
        注意力窗口的左（包含）窗口大小，当设置为 ``-1`` 时，窗口大小将设置为序列的全长。默认为 ``-1``。
    logits_soft_cap : Optional[float]
        注意力 logit 的软上限值（用于 Gemini, Grok 和 Gemma-2 等），如果未提供，将设置为 ``0``。如果大于 0，logits 将根据公式进行上限：
        $\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})$，
        其中 $x$ 是输入 logits。
    sm_scale : Optional[float]
        用于 softmax 的缩放因子，如果未提供，将设置为 ``1.0 / sqrt(head_dim)``。
    rope_scale : Optional[float]
        用于 RoPE 插值的缩放因子，如果未提供，将设置为 1.0。
    rope_theta : Optional[float]
        用于 RoPE 的 theta，如果未提供，将设置为 1e4。
    return_lse : bool
        是否返回注意力 logit 的 logsumexp 值。

    返回
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        如果 `return_lse` 是 ``False``，返回注意力输出，形状：``[qo_len, num_qo_heads, head_dim]``。
        如果 `return_lse` 是 ``True``，返回一个包含两个张量的元组：

        * 注意力输出，形状：``[qo_len, num_qo_heads, head_dim]``。
        * 注意力 logit 的 logsumexp 值，形状：``[qo_len, num_qo_heads]``。

    示例
    --------

    import torch
    import flashinfer
    qo_len = 128
    kv_len = 4096
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,
            allow_fp16_qk_reduction=True)
    o.shape
    torch.Size([128, 32, 128])
    mask = torch.tril(
        torch.full((qo_len, kv_len), True, device="cuda:0"),
        diagonal=(kv_len - qo_len),
    )
    print(mask)
    tensor([[ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            ...,
            [ True,  True,  True,  ...,  True, False, False],
            [ True,  True,  True,  ...,  True,  True, False],
            [ True,  True,  True,  ...,  True,  True,  True]], device='cuda:0')
    o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    assert torch.allclose(o, o_custom, rtol=1e-3, atol=1e-3)
    True

    注意
    ----
    ``num_qo_heads`` 必须是 ``num_kv_heads`` 的倍数。如果 ``num_qo_heads`` 不等于 ``num_kv_heads``，函数将使用 `分组查询注意力 <https://arxiv.org/abs/2305.13245>`_。
```

- `single_prefill_with_kv_cache_return_lse(q, k, v)`: 单请求的预填充/追加注意力，使用 kv cache ，返回注意力输出。

### Batch Prefill/Append Attention

`class flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, qo_indptr_buf: torch.Tensor | None = None, paged_kv_indptr_buf: torch.Tensor | None = None, paged_kv_indices_buf: torch.Tensor | None = None, paged_kv_last_page_len_buf: torch.Tensor | None = None, custom_mask_buf: torch.Tensor | None = None, qk_indptr_buf: torch.Tensor | None = None)`

用于批量请求的prefill/append注意力的 Paged  kv-cache 包装器类。

示例：

```python
import torch
import flashinfer

# 定义模型参数
num_layers = 32  # 模型层数
num_qo_heads = 64  # 查询/输出头数
num_kv_heads = 16  # 键/值头数
head_dim = 128  # 每个头的维度
max_num_pages = 128  # 最大页数
page_size = 16  # 每页的大小

# 分配128MB的工作区缓冲区
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# 创建BatchPrefillWithPagedKVCacheWrapper实例
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

# 定义批量大小和非零查询/输出数量
batch_size = 7
nnz_qo = 100

# 创建查询/输出的indptr数组
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)

# 创建分页键/值的索引数组
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")

# 创建分页键/值的indptr数组
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)

# 创建分页键/值的最后一页长度数组
paged_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)

# 创建查询张量
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")

# 创建键/值缓存张量
kv_cache_at_layer = torch.randn(
    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)

# 创建批量预填充注意力的辅助数据结构
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
)

# 计算每层的批量预填充注意力
outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    # 计算批量预填充注意力，重用辅助数据结构
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)

print(outputs[0].shape)  # 输出形状: torch.Size([100, 64, 128])

# 下面是创建自定义掩码的另一个示例
mask_arr = []
qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
for i in range(batch_size):
    mask_i = torch.tril(
        torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
        diagonal=(kv_len[i] - qo_len[i]),
    )
    mask_arr.append(mask_i.flatten())

mask = torch.cat(mask_arr, dim=0)

# 重新计划批量预填充注意力，使用自定义掩码
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=mask,
)

# 计算每层的批量预填充注意力，使用自定义掩码
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    # 计算批量预填充注意力，重用辅助数据结构
    o_custom = prefill_wrapper.run(q, kv_cache)
    assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
```


> 注意：为了加速计算，FlashInfer 的批量预填充/追加注意力操作符会创建一些辅助数据结构，这些数据结构可以在多次预填充/追加注意力调用中重用（例如，不同的 Transformer 层）。此包装类管理这些数据结构的生命周期。

`__init__(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, qo_indptr_buf: torch.Tensor | None = None, paged_kv_indptr_buf: torch.Tensor | None = None, paged_kv_indices_buf: torch.Tensor | None = None, paged_kv_last_page_len_buf: torch.Tensor | None = None, custom_mask_buf: torch.Tensor | None = None, qk_indptr_buf: torch.Tensor | None = None) → None`

Constructor of `BatchPrefillWithPagedKVCacheWrapper`.

```python
参数
    ----------
    float_workspace_buffer : torch.Tensor
        用户预留的工作区缓冲区，用于在 split-k 算法中存储中间注意力结果。推荐大小为 128MB，工作区缓冲区的设备应与输入张量的设备相同。

    kv_layout : str
        输入 k/v 张量的布局，可以是 ``NHD`` 或 ``HND``。

    use_cuda_graph : bool
        是否启用 CUDA 图捕获以加速预填充内核，如果启用，辅助数据结构将存储在提供的缓冲区中。当启用 CUDAGraph 时，此包装器的生命周期内 ``batch_size`` 不能改变。

    qo_indptr_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储 ``qo_indptr`` 数组，缓冲区的大小应为 ``[batch_size + 1]``。仅当 ``use_cuda_graph`` 为 ``True`` 时此参数才有效。

    paged_kv_indptr_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储 ``paged_kv_indptr`` 数组，此缓冲区的大小应为 ``[batch_size + 1]``。仅当 ``use_cuda_graph`` 为 ``True`` 时此参数才有效。

    paged_kv_indices_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储 ``paged_kv_indices`` 数组，应足够大以存储此包装器生命周期内 ``paged_kv_indices`` 数组的最大可能大小。仅当 ``use_cuda_graph`` 为 ``True`` 时此参数才有效。

    paged_kv_last_page_len_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储 ``paged_kv_last_page_len`` 数组，缓冲区的大小应为 ``[batch_size]``。仅当 ``use_cuda_graph`` 为 ``True`` 时此参数才有效。

    custom_mask_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储自定义掩码张量，应足够大以存储此包装器生命周期内打包的自定义掩码张量的最大可能大小。仅当 ``use_cuda_graph`` 设置为 ``True`` 且自定义掩码将用于注意力计算时此参数才有效。

    qk_indptr_buf : Optional[torch.Tensor]
        用户预留的缓冲区，用于存储 ``qk_indptr`` 数组，缓冲区的大小应为 ``[batch_size + 1]``。仅当 ``use_cuda_graph`` 为 ``True`` 且自定义掩码将用于注意力计算时此参数才有效。
```


`plan(qo_indptr: torch.Tensor, paged_kv_indptr: torch.Tensor, paged_kv_indices: torch.Tensor, paged_kv_last_page_len: torch.Tensor, num_qo_heads: int, num_kv_heads: int, head_dim: int, page_size: int, custom_mask: torch.Tensor | None = None, packed_custom_mask: torch.Tensor | None = None, causal: bool = False, pos_encoding_mode: str = 'NONE', allow_fp16_qk_reduction: bool = False, sm_scale: float | None = None, window_left: int = -1, logits_soft_cap: float | None = None, rope_scale: float | None = None, rope_theta: float | None = None, q_data_type: str | torch.dtype = 'float16', kv_data_type: str | torch.dtype | None = None) → None`

Plan batch prefill/append attention on Paged KV-Cache for given problem specification.

```python
Parameters
    ----------
    qo_indptr : torch.Tensor
        查询/输出张量的 indptr，形状: ``[batch_size + 1]``。
    paged_kv_indptr : torch.Tensor
        分页的 kv cache 的 indptr，形状: ``[batch_size + 1]``。
    paged_kv_indices : torch.Tensor
        分页的 kv cache 的页索引，形状: ``[qo_indptr[-1]]``。
    paged_kv_last_page_len : torch.Tensor
        每个请求在分页的 kv cache 中的最后一页的条目数，形状: ``[batch_size]``。
    num_qo_heads : int
        查询/输出头的数量。
    num_kv_heads : int
        键/值头的数量。
    head_dim : int
        头的维度。
    page_size : int
        分页的 kv cache 中每个页的大小。
    custom_mask : Optional[torch.Tensor]
        布尔掩码张量的展平形式，形状: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``。
        掩码张量中的元素应为 ``True`` 或 ``False``，其中 ``False`` 表示注意力矩阵中的相应元素将被屏蔽。

        有关掩码张量展平布局的详细信息，请参阅 `mask layout <mask-layout>`。

        当提供 `custom_mask` 且未提供 `packed_custom_mask` 时，函数会将自定义掩码张量打包成 1D 打包掩码张量，这会引入额外的开销。
    packed_custom_mask : Optional[torch.Tensor]
        1D 打包的 uint8 掩码张量，如果提供，`custom_mask` 将被忽略。
        打包的掩码张量由 :func:`flashinfer.quantization.packbits` 生成。
    causal : bool
        是否对注意力矩阵应用因果掩码。
        仅当在 `plan` 中未提供 `custom_mask` 时，此参数才有效。
    pos_encoding_mode : str
        在注意力内核中应用的位置编码，可以是 ``NONE``/``ROPE_LLAMA`` (LLAMA 风格的旋转嵌入) /``ALIBI``。
        默认值为 ``NONE``。
    allow_fp16_qk_reduction : bool
        是否使用 f16 进行 qk 降维（更快但精度略有损失）。
    window_left : int
        注意力窗口的左（包含）窗口大小，当设置为 ``-1`` 时，窗口大小将设置为序列的全长。默认值为 ``-1``。
    logits_soft_cap : Optional[float]
        注意力 logit 的软上限值（用于 Gemini, Grok 和 Gemma-2 等），如果未提供，将设置为 ``0``。如果大于 0，logit 将根据以下公式进行上限：
        $\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})$，
        其中 $x$ 是输入 logit。
    sm_scale : Optional[float]
        用于 softmax 的缩放因子，如果未提供，将设置为 ``1.0 / sqrt(head_dim)``。
    rope_scale : Optional[float]
        用于 RoPE 插值的缩放因子，如果未提供，将设置为 ``1.0``。
    rope_theta : Optional[float]
        用于 RoPE 的 theta，如果未提供，将设置为 ``1e4``。
    q_data_type : Union[str, torch.dtype]
        查询张量的数据类型，默认为 torch.float16。
    kv_data_type : Optional[Union[str, torch.dtype]]
        键/值张量的数据类型。如果为 None，将设置为 `q_data_type`。

    注意
    ----
    在调用任何 `run` 或 `run_return_lse` 之前，应调用 `plan` 方法，辅助数据结构将在此调用期间创建并缓存以供多次内核运行使用。

    ``num_qo_heads`` 必须是 ``num_kv_heads`` 的倍数。如果 ``num_qo_heads`` 不等于 ``num_kv_heads``，函数将使用 `分组查询注意力 <https://arxiv.org/abs/2305.13245>`_。
```

`reset_workspace_buffer(float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor) → None`

Reset the workspace buffer.

```python
Parameters
    float_workspace_buffer : torch.Tensor
        新的浮点工作区缓冲区，其设备应与输入张量的设备相同。

    int_workspace_buffer : torch.Tensor
        新的整数工作区缓冲区，其设备应与输入张量的设备相同。
```

`run(q: torch.Tensor, paged_kv_cache: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], k_scale: float | None = None, v_scale: float | None = None, return_lse: bool = False) → torch.Tensor | Tuple[torch.Tensor, torch.Tensor]`

计算查询和paged kv-cache 之间的批量预填充/追加注意力。

```python
Parameters
    ----------
    q : torch.Tensor
        查询张量，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``
    paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        分页的KV缓存，存储为张量的元组或单个张量：

        * 一个包含4维张量的元组 ``(k_cache, v_cache)``，每个张量的形状为：
            ``[max_num_pages, page_size, num_kv_heads, head_dim]`` 如果 `kv_layout` 是 ``NHD``,
            和 ``[max_num_pages, num_kv_heads, page_size, head_dim]`` 如果 `kv_layout` 是 ``HND``。

        * 一个5维张量，形状为：
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` 如果
            `kv_layout` 是 ``NHD``，和
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` 如果
            `kv_layout` 是 ``HND``。其中 ``paged_kv_cache[:, 0]`` 是键缓存，``paged_kv_cache[:, 1]`` 是值缓存。

    k_scale : Optional[float]
        fp8输入的键校准比例，如果未提供，将设置为 ``1.0``。
    v_scale : Optional[float]
        fp8输入的值校准比例，如果未提供，将设置为 ``1.0``。
    return_lse : bool
        是否返回注意力输出的对数和指数。

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        如果 `return_lse` 是 ``False``，返回注意力输出，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``。
        如果 `return_lse` 是 ``True``，返回一个包含两个张量的元组：

        * 注意力输出，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``。
        * 注意力输出的对数和指数，形状：``[qo_indptr[-1], num_qo_heads]``。
```

`class flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, qo_indptr_buf: torch.Tensor | None = None, kv_indptr_buf: torch.Tensor | None = None, custom_mask_buf: torch.Tensor | None = None, qk_indptr_buf: torch.Tensor | None = None)`

用于批量请求的预填充/追加注意力的包装类，支持不规则（张量）kv-cache。

EXAMPLE:

```python
import torch
import flashinfer

# 定义模型层数
num_layers = 32
# 定义查询/输出头数
num_qo_heads = 64
# 定义键/值头数
num_kv_heads = 16
# 定义每个头的维度
head_dim = 128

# 分配128MB的工作区缓冲区
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# 创建BatchPrefillWithRaggedKVCacheWrapper实例
prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace_buffer, "NHD"
)

# 定义批量大小和非零键/值数量
batch_size = 7
nnz_kv = 100
# 定义非零查询/输出数量
nnz_qo = 100

# 创建查询/输出的indptr数组
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)

# 创建键/值的indptr数组
kv_indptr = qo_indptr.clone()

# 创建查询张量
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")

# 创建键张量
k_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")

# 创建值张量
v_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")

# 创建批量预填充注意力的辅助数据结构
prefill_wrapper.plan(
    qo_indptr,
    kv_indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal=True,
)

# 存储输出结果
outputs = []

# 遍历每一层计算批量预填充注意力
for i in range(num_layers):
    q = q_at_layer[i]
    k = k_at_layer[i]
    v = v_at_layer[i]
    # 计算批量预填充注意力，重用辅助数据结构
    o = prefill_wrapper.run(q, k, v)
    outputs.append(o)

# 打印第一个输出的形状
print(outputs[0].shape)
# torch.Size([100, 64, 128])

# 下面是创建自定义掩码的另一个示例
mask_arr = []
# 计算每个查询/输出的长度
qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
# 计算每个键/值的长度
kv_len = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()

# 为每个批次创建自定义掩码
for i in range(batch_size):
    mask_i = torch.tril(
        torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
        diagonal=(kv_len[i] - qo_len[i]),
    )
    mask_arr.append(mask_i.flatten())

# 将所有掩码连接成一个张量
mask = torch.cat(mask_arr, dim=0)

# 使用自定义掩码创建辅助数据结构
prefill_wrapper.plan(
    qo_indptr,
    kv_indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    custom_mask=mask
)

# 存储使用自定义掩码的输出结果
outputs_custom_mask = []

# 遍历每一层计算批量预填充注意力
for i in range(num_layers):
    q = q_at_layer[i]
    k = k_at_layer[i]
    v = v_at_layer[i]
    # 计算批量预填充注意力，重用辅助数据结构
    o_custom = prefill_wrapper.run(q, k, v)
    # 确保自定义掩码的输出与默认掩码的输出一致
    assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    outputs_custom_mask.append(o_custom)

# 打印第一个使用自定义掩码的输出的形状
print(outputs_custom_mask[0].shape)
# torch.Size([100, 64, 128])
```

> 为了加速计算，FlashInfer 的批量预填充/追加注意力操作符创建了一些辅助数据结构，这些数据结构可以在多个预填充/追加注意力调用中重用（例如不同的 Transformer 层）。这个包装类管理这些数据结构的生命周期。

`__init__(float_workspace_buffer: torch.Tensor, kv_layout: str = 'NHD', use_cuda_graph: bool = False, qo_indptr_buf: torch.Tensor | None = None, kv_indptr_buf: torch.Tensor | None = None, custom_mask_buf: torch.Tensor | None = None, qk_indptr_buf: torch.Tensor | None = None) → None`

Constructor of `BatchPrefillWithRaggedKVCacheWrapper`.

```python
Parameters
    ----------
    float_workspace_buffer : torch.Tensor
        用户预留的浮点工作空间缓冲区，用于存储分割-k算法中的中间注意力结果。
        推荐大小为128MB，工作空间缓冲区的设备应与输入张量的设备相同。

    kv_layout : str
        输入k/v张量的布局，可以是 ``NHD`` 或 ``HND``。

    use_cuda_graph : bool
        是否为预填充内核启用CUDA图捕获，如果启用，辅助数据结构将存储为提供的缓冲区。

    qo_indptr_buf : Optional[torch.Tensor]
        用户预留的GPU缓冲区，用于存储 ``qo_indptr`` 数组，缓冲区的大小应为 ``[batch_size + 1]``。
        此参数仅在 ``use_cuda_graph`` 为 ``True`` 时有效。

    kv_indptr_buf : Optional[torch.Tensor]
        用户预留的GPU缓冲区，用于存储 ``kv_indptr`` 数组，缓冲区的大小应为 ``[batch_size + 1]``。
        此参数仅在 ``use_cuda_graph`` 为 ``True`` 时有效。

    custom_mask_buf : Optional[torch.Tensor]
        用户预留的GPU缓冲区，用于存储自定义掩码张量，应足够大以存储包装器生命周期内打包的自定义掩码张量的最大可能大小。
        此参数仅在 ``use_cuda_graph`` 为 ``True`` 且在注意力计算中使用自定义掩码时有效。

    qk_indptr_buf : Optional[torch.Tensor]
        用户预留的GPU缓冲区，用于存储 ``qk_indptr`` 数组，缓冲区的大小应为 ``[batch_size]``。
        此参数仅在 ``use_cuda_graph`` 为 ``True`` 且在注意力计算中使用自定义掩码时有效。
```

`plan(qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, num_qo_heads: int, num_kv_heads: int, head_dim: int, custom_mask: torch.Tensor | None = None, packed_custom_mask: torch.Tensor | None = None, causal: bool = True, pos_encoding_mode: str = 'NONE', allow_fp16_qk_reduction: bool = False, window_left: int = -1, logits_soft_cap: float | None = None, sm_scale: float | None = None, rope_scale: float | None = None, rope_theta: float | None = None, q_data_type: str = 'float16', kv_data_type: str | None = None) → None`

Plan batch prefill/append attention on Ragged KV-Cache for given problem specification.

```python
Parameters
    ----------
    qo_indptr : torch.Tensor
        查询/输出张量的indptr，形状：``[batch_size + 1]``。
    kv_indptr : torch.Tensor
        键/值张量的indptr，形状：``[batch_size + 1]``。
    num_qo_heads : int
        查询/输出头的数量。
    num_kv_heads : int
        键/值头的数量。
    head_dim : int
        头的维度。
    custom_mask : Optional[torch.Tensor]
        展平的布尔掩码张量，形状：``(sum(q_len[i] * k_len[i] for i in range(batch_size)))``。
        掩码张量中的元素应为 ``True`` 或 ``False``，
        其中 ``False`` 表示注意力矩阵中相应的元素将被
        屏蔽。
        
        当提供 `custom_mask` 且未提供 `packed_custom_mask` 时，
        函数将把自定义掩码张量打包成一个1D打包掩码张量，这会引入
        额外的开销。
    packed_custom_mask : Optional[torch.Tensor]
        1D打包的uint8掩码张量，如果提供，将忽略 `custom_mask`。
        打包的掩码张量由 :func:`flashinfer.quantization.packbits` 生成。

        如果提供，自定义掩码将在softmax之前和缩放之后
        添加到注意力矩阵中。掩码张量应与输入张量在同一设备上。
    causal : bool
        是否对注意力矩阵应用因果掩码。
        如果在 `plan` 中提供了 ``mask``，则忽略此参数。
    pos_encoding_mode : str
        在注意力内核中应用的位置编码，可以是
        ``NONE``/``ROPE_LLAMA`` (LLAMA风格的旋转嵌入) /``ALIBI``。
        默认为 ``NONE``。
    allow_fp16_qk_reduction : bool
        是否使用f16进行qk归约（以轻微的精度损失为代价提高速度）。
    window_left : int
        注意力窗口的左侧（包含）窗口大小，当设置为 ``-1`` 时，窗口
        大小将设置为序列的全长。默认为 ``-1``。
    logits_soft_cap : Optional[float]
        注意力logits软上限值（用于Gemini、Grok和Gemma-2等），如果未
        提供，将设置为 ``0``。如果大于0，logits将根据以下公式进行上限处理：
        `\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`，
        其中 `x` 是输入logits。
    sm_scale : Optional[float]
        softmax中使用的缩放，如果未提供，将设置为
        ``1.0 / sqrt(head_dim)``。
    rope_scale : Optional[float]
        RoPE插值中使用的缩放，如果未提供，将设置为
        ``1.0``。
    rope_theta : Optional[float]
        RoPE中使用的theta，如果未提供，将设置为 ``1e4``。
    q_data_type : Union[str, torch.dtype]
        查询张量的数据类型，默认为torch.float16。
    kv_data_type : Optional[Union[str, torch.dtype]]
        键/值张量的数据类型。如果为None，将设置为 `q_data_type`。

    注意
    ----
    在任何 `run` 或 `run_return_lse` 调用之前应调用 `plan` 方法，
    辅助数据结构将在此plan调用期间创建并缓存以供多次内核运行使用。

    ``num_qo_heads`` 必须是 ``num_kv_heads`` 的倍数。如果 ``num_qo_heads``
    不等于 ``num_kv_heads``，函数将使用
    `分组查询注意力 <https://arxiv.org/abs/2305.13245>`_。
```

`reset_workspace_buffer(float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor) → None`

Reset the workspace buffer.

```python
Parameters
    float_workspace_buffer : torch.Tensor
        新的浮点工作区缓冲区，其设备应与输入张量的设备相同。

    int_workspace_buffer : torch.Tensor
        新的整数工作区缓冲区，其设备应与输入张量的设备相同。
```

`run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False) → torch.Tensor | Tuple[torch.Tensor, torch.Tensor]`

Compute batch prefill/append attention between query and kv-cache stored as ragged tensor.

```python
Parameters
    ----------
    q : torch.Tensor
        查询张量，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``
    k : torch.Tensor
        键张量，形状：``[kv_indptr[-1], num_kv_heads, head_dim]``
    v : torch.Tensor
        值张量，形状：``[kv_indptr[-1], num_kv_heads, head_dim]``
    return_lse : bool
        是否返回注意力输出的对数和指数

    返回
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        如果 `return_lse` 为 ``False``，返回注意力输出，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``。
        如果 `return_lse` 为 ``True``，返回两个张量的元组：

        * 注意力输出，形状：``[qo_indptr[-1], num_qo_heads, head_dim]``。
        * 注意力输出的对数和指数，形状：``[qo_indptr[-1], num_qo_heads]``。
```

