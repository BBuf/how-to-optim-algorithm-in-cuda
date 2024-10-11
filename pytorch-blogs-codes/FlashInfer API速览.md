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

当请求数量大于 1 时，不同的请求可能具有不同的查询长度和 kv 长度。为了避免填充，我们使用 2D ragged tensor 来存储注意力掩码。输入的 ``qo_indptr`` 和 ``kv_indptr`` 数组（长度均为 ``num_requests+1``）用于存储每个请求的可变序列长度信息，``qo_indptr[i+1]-qo_indptr[i]`` 是请求 ``i`` 的查询长度（``qo_len[i]``），``kv_indptr[i+1]-kv_indptr[i]`` 是请求 ``i`` 的 kv 长度（``kv_len[i]``）。

所有请求的掩码数组被展平（查询作为第一维度，kv 作为最后一维）并连接成一个 1D 数组：``mask_data``。FlashInfer 会隐式创建一个 ``qk_indptr`` 数组来存储每个请求的掩码在展平的掩码数组中的起始偏移量：``qk_indptr[1:] = cumsum(qo_len * kv_len)``。

``mask_data`` 的形状为 ``(qk_indptr[-1],)``，我们可以使用 ``mask_data[qk_indptr[i]:qk_indptr[i+1]]`` 来切片请求 ``i`` 的展平掩码。

为了节省内存，我们可以进一步将布尔展平的布尔掩码数组打包成位打包数组（每个元素 1 位，8 个元素打包成一个 `uint8`），使用“little”位序（详见 `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_）。FlashInfer 接受布尔掩码和位打包掩码。如果提供布尔掩码，FlashInfer 会将其内部打包成 bit-packed 数组。

### FlashInfer APIs

`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` 和 `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` 允许用户在 `begin_forward` 函数中指定 ``qo_indptr``、``kv_indptr`` 和自定义注意力掩码 ``custom_mask``，掩码数据将在注意力 kernel 中的 softmax 之前（以及 softmax 缩放之后）添加到注意力分数中。

`flashinfer.quantization.packbits` 和 `flashinfer.quantization.segment_packbits` 是用于将布尔掩码打包成 bit-packed 数组的工具函数。

## Page Table 布局

当 KV-Cache 是动态的（例如在 append 或 decode 阶段），打包所有键/值是不高效的，因为每个请求的序列长度会随时间变化。`vLLM <https://arxiv.org/pdf/2309.06180.pdf>`_ 
提出将 KV-Cache 组织为Page Table。在 FlashInfer 中，我们将 Page Table 视为一个块稀疏矩阵（每个使用的页面可以视为块稀疏矩阵中的一个非零块）并使用 `CSR 格式 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>` 来索引 KV-Cache 中的Page。

![](https://files.mdnice.com/user/59/d14c4007-6568-4039-9733-ac5fd069ecb7.png)

对于每个请求，我们记录其 ``page_indices`` 和 ``last_page_len``，分别跟踪该请求使用的页面和最后一个页面中的条目数量。请求 ``i`` 的 KV 序列长度为 ``page_size * (len(page_indices[i]) - 1) + last_page_length[i]``。

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

其中，``max_num_pages`` 是所有请求使用的最大页面数，``page_size`` 是每个页面中容纳的 token 数量。在单个张量存储中，``2`` 表示 K/V（第一个用于Key，第二个用于Value）。

### FlashInfer APIs

:meth:`flashinfer.page.append_paged_kv_cache` can append a batch of keys/values (stored as ragged tensors) to the paged KV-Cache
(the pages for these appended keys/values must be allocated prior to calling this API).

:class:`flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` and :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` implements the decode attention
and prefill/append attention between queries stored in ragged tensors and keys/values stored in paged KV-Cache.

.. _cascade-inference-data-layout:

Multi-level Cascade Inference Data Layout
-----------------------------------------

When using multi-level `cascade inference <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_,
the query and output are stored in ragged tensors, and KV-Cache of all levels are stored
in a unified Paged KV-Cache. Each level has a unique ``qo_indptr`` array which is the prefix sum of the
accumulated number of tokens to append in the subtree, as well as ``kv_page_indptr``, ``kv_page_indices``, and
``kv_last_page_len`` which has same semantics as in :ref:`page-layout` section. The following figure
introduce how to construct these data structures for append attention operation for 8 requests where we
treat their KV-Cache as 3 levels for prefix reuse:

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/cascade_inference_data_layout.png
  :width: 800
  :align: center
  :alt: Cascade inference data layout.

Note that we don't have to change the data layout of ragged query/output tensor or paged kv-cache for each level.
All levels share the same underlying data layout, but we use different ``qo_indptr`` / ``kv_page_indptr`` arrays
so that we can view them in different ways.

FlashInfer APIs
~~~~~~~~~~~~~~~

FlashInfer provides :class:`flashinfer.cascade.MultiLevelCascadeAttentionWrapper` to compute
the cascade attention.

FAQ
---

How do FlashInfer manages KV-Cache?
  FlashInfer itself is not responsible for managing the page-table (pop and allocate new pages, etc.) and we leave the strategy
  to the user: different serving engine might have different strategies to manage the page-table. FlashInfer is only responsible
  for computing the attention between queries and keys/values stored in KV-Cache.
