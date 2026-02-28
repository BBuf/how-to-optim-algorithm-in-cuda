> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/predication-in-cutlass/

# Cutlass中的预测

Cutlass文档中关于CuTe的部分简要提到了预测(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0y_predication.md)这个主题,但没有给出完整的代码示例。在这篇博文中,我将解释如何在CuTe程序中使用预测来执行适当的边界检查。

## 介绍

我们从CuTe教程(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu)中的一个kernel 开始,它执行高效的分块复制。在开始讨论预测这个主题之前,让我们先简要关注一下非向量化版本的分块复制。

```c++
/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  # SEE BELOW
}

/// Main function
int main(int argc, char** argv) {
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  int M = 32768;
  int N = 16384;

  auto tensor_shape = make_shape(M, N);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Tensor tensor_S =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                  make_layout(tensor_shape));
  Tensor tensor_D =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                  make_layout(tensor_shape));

  auto block_shape = make_shape(Int<256>{}, Int<128>{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape);  // ((M, N), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape);  // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout =
      make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  dim3 gridDim(
      size<1>(tiled_tensor_D),
      size<2>(tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  copy_kernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                     thr_layout);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
              << std::endl;
    return -1;
  }

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i
                << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}
```

这个例子是从cutlass repo中的CuTe教程中采用的。这些都是我们在主函数中调用kernel 之前采取的步骤。我们将在下面逐步解释它们。

```c++
  Tensor tensor_S =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                  make_layout(tensor_shape));
  Tensor tensor_D =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                  make_layout(tensor_shape));
```

简单地初始化张量。

```c++
  auto block_shape = make_shape(Int<256>{}, Int<128>{});

  
  if ((size<0>(tensor_shape) % size<0>(block_shape)) ||
      (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape."
              << std::endl;
    return -1;
  }

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape);  
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape);  
```

这里我们简单地分块张量。这将把`(M,N) -> ((blkM, blkN), ceil(M/blkM), ceil(N/blkN)`, i.e.我们将初始矩阵分块成更小的矩阵,形状为`(blkM, blkN)`.形状的最后两个维度对应于我们在x和y维度中创建的块的数量。对于上面的例子,我们将有`(32768, 16384) -> ((256, 128), 128, 128)`

```c++
 Layout thr_layout =
      make_layout(make_shape(Int<32>{}, Int<8>{}));  

  dim3 gridDim(
      size<1>(tiled_tensor_D),
      size<2>(tiled_tensor_D));  
  dim3 blockDim(size(thr_layout));

  copy_kernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                     thr_layout);
```

这里我们制作一个线程布局。这将进一步将我们的块分块到kernel 中的线程中,如下所示。`(256, 128) -> (256/32, 128/8) = (8, 16)`然后我们用分块布局给出的块数启动kernel (即x方向上有256个块,y方向上有128个块)和每个块所需的线程数(即`32 * 8 = 256`)。

非预测的kernel 如下所示:

```c++
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  using namespace cute;
  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)

  Tensor fragment = make_tensor_like(thr_tile_S);  // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}
```

它简单地取整个矩阵块,然后像上面描述的那样创建一个局部分区。然后每个线程从`GMEM -> RMEM -> GMEM`复制这些元素。这个过程是非常高效的,并且可以在H100上实现`~3 TB/s`的带宽。我们可以通过调整不同的块大小来进一步增加这个值, 但这不是本篇博文的重点。

## 为什么我们需要预测?

让我们想象一下,我们想要处理一个维度为`(M, N) = (32768 + 1, 16384 + 1)`的矩阵,上面的kernel 将无法工作。为什么?因为我们的分块将导致一个布局`(32768 + 1, 16384 + 1) -> ((256, 128), 129, 129)`.这里的问题是,在最后一个块中,我们将尝试复制我们不应该复制的数据。这是因为x方向上的最后一个块中只有一个元素需要处理,y方向上的最后一个块也是如此。我们不想复制这些块的整个线程块!你可以通过调整M和N来尝试运行程序,你会得到一个像这样的错误:

```c++
CUDA Runtime error: an illegal memory access was encountered
terminate called after throwing an instance of 'thrust::THRUST_200700_900_NS::system::system_error'
  what():  CUDA free failed: cudaErrorIllegalAddress: an illegal memory access was encountered
Aborted (core dumped)
```

这应该对任何使用CUDAkernel 的人来说都不应该感到惊讶。我们经常需要进行适当的边界检查。

## 使用CuTe进行预测

在这篇博文中,我们将给出解决上述问题的kernel。

```c++
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel_predicate(TensorS S, TensorD D, ThreadLayout, int M,
                                      int N) {
  using namespace cute;

  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)

  auto identity_tensor = make_identity_tensor(make_shape(
      size<0>(tile_S), size<1>(tile_S)));  // (BlockShape_M, BlockShape_N)
  auto thread_identity_tensor = local_partition(
      identity_tensor, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  Tensor fragment = make_tensor_like(thr_tile_S);  // (ThrValM, ThrValN)
  auto predicator = make_tensor<bool>(
      make_shape(size<0>(fragment), size<1>(fragment)));  // (ThrValM, ThrValN)

  CUTE_UNROLL
  for (int i = 0; i < size<0>(predicator); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(predicator); ++j) {
      auto thread_identity = thread_identity_tensor(i, j);
      int global_row = blockIdx.x * size<0>(tile_S) + get<0>(thread_identity);
      int global_col = blockIdx.y * size<1>(tile_S) + get<1>(thread_identity);
      predicator(i, j) = (global_row < M) && (global_col < N);
    }
  }

  // Copy from GMEM to RMEM and from RMEM to GMEM with predicate
  copy_if(predicator, thr_tile_S, fragment);
  copy_if(predicator, fragment, thr_tile_D);
}
```

我们可以看到这个kernel与没有预测的版本非常相似。预测的逻辑是从Lei Mao(https://leimao.github.io/article/CuTe-Matrix-Transpose/)那里借鉴的,他在矩阵转置任务中使用了类似的技术。强烈推荐阅读他的博客文章,写得非常好!现在我们将解释为了让kernel能够处理那些不能被block tile维度整除的矩阵,我们需要做哪些改动。

```c++
  auto identity_tensor = make_identity_tensor(make_shape(
      size<0>(tile_S), size<1>(tile_S)));  // (BlockShape_M, BlockShape_N)
  auto thread_identity_tensor = local_partition(
      identity_tensor, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
```

我们创建一个与要复制的张量具有完全相同分块的恒等张量。这个恒等张量将简单地映射 `(x,y)->(x,y)`。


```c++
  auto predicator = make_tensor<bool>(
      make_shape(size<0>(fragment), size<1>(fragment)));  // (ThrValM, ThrValN)
```

我们初始化一个预测矩阵。这个矩阵将对于所有在边界`[0, M] x [0, N]`内的元组`(x,y)`为1, 即当问题中的元素在矩阵内时,它将为1。


```c++
 CUTE_UNROLL
  for (int i = 0; i < size<0>(predicator); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(predicator); ++j) {
      auto thread_identity = thread_identity_tensor(i, j);
      int global_row = blockIdx.x * size<0>(tile_S) + get<0>(thread_identity);
      int global_col = blockIdx.y * size<1>(tile_S) + get<1>(thread_identity);
      predicator(i, j) = (global_row < M) && (global_col < N);
    }
  }
```

我们遍历所有线程块。我们为每个元组`(i, j)`计算对应的行和列。我们可以简单地通过将`blockIdx`乘以对应维度中的块tile长度,并加上由于线程分块引起的相应偏移量来实现。

```c++
  copy_if(predicator, thr_tile_S, fragment);
  copy_if(predicator, fragment, thr_tile_D);
```

这将简单地复制那些在矩阵边界内的元素。kernel 将愉快地复制形状为`(M, N) = (32768 + 1, 16384 + 1)`的矩阵,而不会报错,并给出正确的结果。对于那些可以被块tile维度整除的矩阵,性能与上面的复制kernel相当。我猜这是因为编译器识别出它可以优化掉预测。对于那些矩阵,kernel 由于warp divergence而稍微不那么优化。

我希望这篇博文能帮助你更好地理解CuTe中的预测。






