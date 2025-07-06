> 博客来源：https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# 构建和开发CUTLASS CUDA kernel

## 简介

CUTLASS(https://github.com/NVIDIA/cutlass) 是一个仅包含头文件的库，由一系列CUDA C++模板抽象组成，用于在CUDA的所有层次和规模上实现高性能的矩阵-矩阵乘法（GEMM）和相关计算。

在这篇博客文章中，我们将在CUDA Docker容器中使用CMake构建CUTLASS和CuTe CUDA kernel。

## CUDA Docker容器

当涉及到为CUTLASS kernel开发创建CUDA Docker容器时，我们会遇到一个选择。要么我们在Docker容器内git clone CUTLASS仅头文件库，要么将CUTLASS仅头文件库作为CUDA kernel源代码的一部分。

起初，我在Docker容器内克隆了CUTLASS仅头文件库。然而，当我尝试从Docker容器检查仅头文件库实现时，这变得不可行。虽然如果Docker容器是VS Code开发容器，我仍然可以尝试从Docker容器检查CUTLASS仅头文件库实现，但如果我想修改并为CUTLASS仅头文件库做贡献，这就变得不友好了。因此，我决定将CUTLASS仅头文件库作为CUDA kernel源代码的一部分。

### 构建Docker镜像

以下CUDA Dockerfile将用于CUTLASS kernel开发。它也可以在我的CUTLASS Examples GitHub仓库中找到。

```shell
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG CMAKE_VERSION=3.30.5
ARG GOOGLETEST_VERSION=1.15.2
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive

# 安装包依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        wget \
        git && \
    apt-get clean

# 系统语言环境
# 对UTF-8很重要
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# 安装CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# 安装GoogleTest
RUN cd /tmp && \
    wget https://github.com/google/googletest/archive/refs/tags/v${GOOGLETEST_VERSION}.tar.gz && \
    tar -xzf v${GOOGLETEST_VERSION}.tar.gz && \
    cd googletest-${GOOGLETEST_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# 为Nsight Compute GUI安装QT6及其依赖
# https://leimao.github.io/blog/Docker-Nsight-Compute/
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libgl1-mesa-glx \
        libxkbfile-dev \
        openssh-client \
        xcb \
        xkb-data \
        libxcb-cursor0 \
        qt6-base-dev && \
    apt-get clean

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel
```

要在本地构建CUTLASS Docker镜像，请运行以下命令。

```shell
$ docker build -f docker/cuda.Dockerfile --no-cache --tag cuda:12.4.1 .
```

### 运行Docker容器

要运行自定义Docker容器，请运行以下命令。

```shell
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt cuda:12.4.1
```

要运行带有NVIDIA Nsight Compute的自定义Docker容器，请运行以下命令。

```shell
$ xhost +
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --network host cuda:12.4.1
$ xhost -
```

## CUTLASS示例

为了证明我们安装的CUTLASS在Docker容器内工作，我们将构建并运行两个从CUTLASS GitHub仓库(https://github.com/NVIDIA/cutlass/tree/v3.5.0) 复制的CUTLASS C++示例，没有任何修改。

CUTLASS是仅头文件的。每个CUTLASS构建目标需要包含两个关键的头文件目录，包括cutlass/include和`cutlass/tools/util/include`。

```shell
cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 设置CUTLASS包含目录
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

add_subdirectory(examples)
```

对于每个构建目标，NVCC编译器需要实验性标志`--expt-relaxed-constexpr`才能在设备代码中使用主机代码的某些`constexpr`。

```shell
cmake_minimum_required(VERSION 3.28)

project(CUTLASS-GEMM-API-V3 VERSION 0.0.1 LANGUAGES CXX CUDA)

# 设置编译代码的CUDA架构
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
add_executable(${PROJECT_NAME} main.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${PROJECT_NAME} PRIVATE --expt-relaxed-constexpr)
```

### 构建示例

要使用CMake构建CUTLASS示例(https://github.com/leimao/CUTLASS-Examples/tree/f93e9d7bfa60ddc631b90d2f96be7bc036cb3e10/examples)，请运行以下命令。

```shell
$ cmake -B build
$ cmake --build build --config Release --parallel
```

### 运行示例

要运行CUTLASS示例，请运行以下命令。

```shell
$ ./build/examples/gemm_api_v2/CUTLASS-GEMM-API-V2
$ echo $?
0
```

```shell
$ ./build/examples/gemm_api_v3/CUTLASS-GEMM-API-V3
10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

Basic data-parallel GEMM
  Disposition: Passed
  Avg runtime: 0.175606 ms
  GFLOPs: 97831.9

StreamK GEMM with default load-balancing
  Disposition: Passed
  Avg runtime: 0.149729 ms
  GFLOPs: 114740
  Speedup vs Basic-DP: 1.173

StreamK emulating basic data-parallel GEMM
  Disposition: Passed
  Avg runtime: 0.177553 ms
  GFLOPs: 96759.2
  Speedup vs Basic-DP: 0.989

Basic split-K GEMM with tile-splitting factor 2
  Disposition: Passed
  Avg runtime: 0.183542 ms
  GFLOPs: 93601.7

StreamK emulating Split-K GEMM with tile-splitting factor 2
  Disposition: Passed
  Avg runtime: 0.173763 ms
  GFLOPs: 98869.8
  Speedup vs Basic-SplitK: 1.056
```

## 参考资料

- CUTLASS(https://github.com/NVIDIA/cutlass)
- CUTLASS Examples - GitHub(https://github.com/leimao/CUTLASS-Examples)
- Nsight Compute In Docker(https://leimao.github.io/blog/Docker-Nsight-Compute/)







