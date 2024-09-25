#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"

// 定义一个矩阵转置的CUDA内核函数
// 使用模板参数来灵活处理不同的张量和布局类型
template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    transposeKernelSmem(TensorS const S, TensorD const D,
                        SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                        SmemLayoutD const smemLayoutD, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;  // 获取输入张量的元素类型

  // 声明并分配共享内存
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // 创建共享内存的两个视图：一个用于输入，一个用于输出
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // 输入视图 (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // 输出视图 (bN, bM)

  // 获取全局内存中当前块的输入和输出张量
  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // 输入 (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // 输出 (bN, bM)

  // 将全局内存和共享内存的张量划分给每个线程
  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // 全局内存输入 (ThrValM, ThrValN)
  Tensor tSsS = local_partition(sS, tS, threadIdx.x); // 共享内存输入 (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x); // 全局内存输出
  Tensor tDsD = local_partition(sD, tD, threadIdx.x); // 共享内存输出

  // 将数据从全局内存复制到共享内存
  cute::copy(tSgS, tSsS); // LDGSTS (Load Global, Store Shared)

  // 确保所有线程完成共享内存的写入
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // 将转置后的数据从共享内存复制回全局内存
  cute::copy(tDsD, tDgD);
}

// 定义矩阵转置的主函数
template <typename Element, bool isSwizzled = true> void transpose_smem(TransposeParams<Element> params) {

  using namespace cute;

  // 创建输入和输出张量
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  // 定义块的大小和形状
  using bM = Int<64>;
  using bN = Int<64>;
  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  // 将输入和输出张量划分为块
  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  // 定义块的布局
  auto tileShapeS = make_layout(block_shape, LayoutRight{});
  auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});

  // 定义共享内存的布局，包括非交错和交错（swizzled）版本
  auto smemLayoutS = tileShapeS;
  auto smemLayoutD = composition(smemLayoutS, tileShapeD);
  auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileShapeS);
  auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileShapeD);

  // 定义线程块内的线程布局
  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  // 计算所需的共享内存大小
  size_t smem_size = int(
      sizeof(SharedStorageTranspose<Element, decltype(smemLayoutS_swizzle)>));

  // 确定网格和块的维度
  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // 网格形状对应于模式 m' 和 n'
  dim3 blockDim(size(threadLayoutS)); // 256 线程

  // 根据是否使用交错（swizzled）布局来启动相应的内核
  if constexpr (isSwizzled) {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle, threadLayoutS,
        smemLayoutD_swizzle, threadLayoutD);
  } else {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS, threadLayoutS,
        smemLayoutD, threadLayoutD);
  }
}
