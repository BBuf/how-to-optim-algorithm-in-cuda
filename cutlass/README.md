# cutlass

CUTLASS / CuTe DSL 学习笔记与代码实现。

## 目录结构

```
cutlass/
├── code/                    # 【代码】CUTLASS / CuTe 实现代码
│   ├── cfx-article-src/     #   cfx 系列博客配套源码（GEMM、TMA、StreamK、EVT 等）
│   ├── cute-examples/       #   CuTe 入门示例（vector_add、gemm-simple、s2r_copy）
│   ├── swizzle/             #   Swizzle 性能测试与分析代码
│   └── mma_vs_wgmma.cu      #   MMA vs WGMMA 性能对比实现
├── cute/                    # 【笔记】CuTe Layout 代数与编程模型笔记
├── gemm/                    # 【笔记】CUTLASS GEMM 实现（Hopper/Ampere Mixed GEMM）
├── tma/                     # 【笔记】Tensor Memory Accelerator (TMA) 教程
├── wgmma/                   # 【笔记】WGMMA 指令与 Hopper 快速矩阵乘
├── swizzle/                 # 【笔记】Swizzle 机制学习笔记
├── instructions/            # 【笔记】CUDA 指令详解（ldmatrix 等）
├── tutorials/               # 【笔记】CUTLASS 翻译教程（矩阵转置、Python bindings）
└── CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记.md   # 综述性入门笔记
```

## 内容说明

### 代码 (`code/`)

| 子目录 | 说明 |
|--------|------|
| [cfx-article-src/](code/cfx-article-src/) | cfx 博客系列配套 CUDA/Python 源码：GEMM、TMA copy、StreamK、EVT、transpose-cute 等 |
| [cute-examples/](code/cute-examples/) | CuTe 入门示例：`vector_add.cu`、`gemm-simple.cu`、`s2r_copy.cu` |
| [swizzle/](code/swizzle/) | Swizzle 访存模式 benchmark 与分析（含 CMake 工程）|
| [mma_vs_wgmma.cu](code/mma_vs_wgmma.cu) | Ampere MMA 与 Hopper WGMMA 性能对比完整实现 |

### 笔记 (主题子目录)

| 子目录 | 说明 |
|--------|------|
| [cute/](cute/) | CuTe Layout 代数、Tensor 抽象、MMA 抽象学习笔记 |
| [gemm/](gemm/) | TRT-LLM Hopper Mixed GEMM（CUTLASS 3.x）与 Quantization GEMM（CUTLASS 2.x）解析 |
| [tma/](tma/) | Mastering NVIDIA TMA 单元 CUTLASS 教程翻译 |
| [wgmma/](wgmma/) | 在 Hopper GPU 上使用 WGMMA 实现快速矩阵乘教程 |
| [swizzle/](swizzle/) | 实用 Swizzle 教程（一）（二）|
| [instructions/](instructions/) | `ldmatrix` 指令详细解析 |
| [tutorials/](tutorials/) | CUTLASS 矩阵转置教程翻译、PyTorch 中为 CUDA 库绑定 Python 接口教程 |
