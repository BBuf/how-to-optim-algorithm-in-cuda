# CuTe Transpose

CuTe Transpose 教程的示例代码。

# 安装

此示例需要 `nvcc` 编译器来编译 CUDA 应用程序。
此外，它需要 [CUTLASS](https://github.com/NVIDIA/cutlass/tree/44dae8b90ef232ea663727470dfbbe9daff6972d) 仓库。
但由于 CUTLASS 是一个仅包含头文件的库，因此不需要安装它。

CUTLASS 作为此仓库的子模块添加。
如果你递归克隆了仓库，它应该会下载到父目录中。
否则，请使用以下命令指定 CUTLASS 目录：

```
export CUTLASS_DIR=/path/to/cutlass
```

编译并运行 C++ 示例：
```
make
./transpose
```

编译 Python 模块并运行 Python 示例：
```
make python -B
python3 torch_benchmark.py
```
