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
#include "util.h"

using namespace cute;

// 定义一个简单的转置内核函数
template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelNaive(TensorS const S, TensorD const DT,
                ThreadLayoutS const tS, ThreadLayoutD const tD) {
  // 定义元素类型
  using Element = typename TensorS::value_type;

  // 获取输入张量的当前块
  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  // 获取输出张量的当前块（已转置）
  Tensor gDT = DT(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  // Concept:                   Tensor  ThrLayout       ThrIndex
  // 将输入块划分给每个线程
  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  // 将输出块划分给每个线程
  Tensor tDgDT = local_partition(gDT, tD, threadIdx.x); // (ThrValM, ThrValN)

  // 创建一个与输入相同形状的寄存器内存张量
  Tensor rmem = make_tensor_like(tSgS);

  // 将输入数据复制到寄存器内存
  copy(tSgS, rmem);
  // 将寄存器内存中的数据复制到输出（实现转置）
  copy(rmem, tDgDT);
}

// 定义转置函数模板
template <typename Element> void transpose_naive(TransposeParams<Element> params) {
  
  //
  // 创建张量
  //
  // 定义输入张量的形状
  auto tensor_shape = make_shape(params.M, params.N);
  // 定义输出张量的形状（已转置）
  auto tensor_shape_trans = make_shape(params.N, params.M);
  // 创建输入张量的内存布局（行主序）
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  // 创建输出张量的内存布局（行主序）
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  // 创建输入张量
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  // 创建输出张量
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
  
  // 创建输出张量的转置视图（列主序）
  auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
  Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutDT);
  
  //
  // 对张量进行分块
  //
  
  // 定义块的大小
  using bM = Int<64>;
  using bN = Int<64>;
  
  // 定义块的形状
  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)
  
  // 将输入张量划分为块
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  // 将输出张量划分为块（已转置）
  Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape_trans); // ((bN, bM), n', m')
  
  // 定义线程布局
  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  
  // 定义网格维度
  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // 网格形状对应于模式 m' 和 n'
  // 定义块维度（256个线程）
  dim3 blockDim(size(threadLayoutS)); 
  // 启动内核
  transposeKernelNaive<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_DT,
                                            threadLayoutS, threadLayoutD);
};
