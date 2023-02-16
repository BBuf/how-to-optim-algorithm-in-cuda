## SoftMax 的不同实现

### OneFlow Softmax

OneFlow 已经有SoftMax的实现介绍：https://zhuanlan.zhihu.com/p/341059988 。我这里会更加详细的介绍其中的代码，最后和 Faster Transformer 的实现做一个对比。

OneFlow 的 softmax 实现被独立到了一个头文件中方便广大开发者使用或者改进，地址为：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh 。

我这里也直接将其搬过来使用并更详细的了解了其中的原理最后测试下性能，需要注意的是oneflow版本的softmax实现依赖了nvidia cub的block reduce，因为要在外面单独运行这个实现需要手动编译下 cub 才可以。接下来展示一下cub的完全编译流程：

1. 构建trust

```shell
# Clone Thrust and CUB from Github. CUB is located in Thrust's
# `dependencies/cub` submodule.
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust
 
# Create build directory:
mkdir build
cd build
 
# Configure -- use one of the following:
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..   # Command line interface.
 
 
# Build:
#cmake --build . -j <num jobs>   # invokes make (or ninja, etc)
make -j16
 
# Run tests and examples:
ctest
```

注意如果 cmake 时出现 `  Failed to detect a default CUDA architecture.` 错误，需要我们自己手动指定一下自己的 nvcc 编译器以及 GPU 架构，也就是在 thrust 的顶层 `CMakeLists.txt` 加入：

```
set(CMAKE_CUDA_ARCHITECTURES 80) # 80对应修改为你自己的 GPU 架构
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```

2. 构建 cub

构建完 thrust 之后我们才可以构建cub。按照上述构建 thrust 的流程走完之后在 `thrust/dependencies` 这个文件夹下会有 cub 文件夹，我们需要进到这个文件夹里面执行：

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j16
```

注意如果这里 cmake 时出现 `  Failed to detect a default CUDA architecture.` 错误，需要我们自己手动指定一下自己的 nvcc 编译器以及 GPU 架构，也就是在 `thrust/dependencies/CMakeLists.txt` 的顶层加入：

```
set(CMAKE_CUDA_ARCHITECTURES 80) # 80对应修改为你自己的 GPU 架构
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```



### FasterTransformer

