// 对于该向量类问题，如果有丰富的CUDA 开发的经验，我们可以通过如下优化手段提供高效的实现：
// - 单个线程处理多个数据，通过数据预取和指令并行，提升数据读取销量、提升执行单元的流水线效率；
// - 对global内存进行大字长读写，减少数据IO所需要的指令数目，减少调度开销，提升程序运行效率；
// - 使用Half2类型，减少half类型引入的PRMT指令的转换和开销；
// - 使用FMA（fused multiply accumulate）指令完成计算，减少FMUL、FADD指令数，提升计算精度；
// 具体地，我们实现了如下函数，
// 其中template行通过编译时常量指定每个线程处理8个数据，避免运行时常量不能利用寄存器（寄存器不可以寻址）所带来的Local Memory问题。由于一个线程处理8个元素，并且单个数据的大小为sizeof(half) = 2, 这样一个线程所需要的数据量为8 x 2 = 16byte，该大小的数据可以通过LDG.128指令实现一条指令完成对数据从全局内存到寄存器到加载；

// 其中函数声明行通过 __global__ 关键字指明该函数为cuda kernel函数，且通过const等信息提示了输入输出数据；

// using行引入cute的名字空间，以便使用cute名字空间中的Tensor工具和其他函数；

// idx和if判断行，通过使用cuda提供的内建threadIdx、blockIdx、blockDim变量来定位线程在网格中的位置；

// Tensor tz、tx、ty行，通过利用make_tensor 接口将kernel参数中的裸指针和维度信息包装成tensor表达；

// Tensor tzr、txr、tyr行，通过local_tile方法实现对tz/tx/ty Tensor的分块和选择（利用idx），在这之后我们只需要关注我们需要处理的局部的tensor即可，无需关注大的全局的tensor。同时在进行局部切块的时候，我们使用Int<>{}形式实现形状的编译常量表示，避免了运行时的量及其会引入的local memory；

// Tensor txR、tyR、tzR行通过make_tensor_like接口实现栈上tensor的定义（GPU表现为寄存器空间）；

// copy行通过调用cute提供的copy函数实现全局内存数据读入到寄存器空间，此处会生成LDG.128指令；

// half2 a2、b2、c2行，重复系数a、b、c构造half2类型的系数，以利用的HFMA2的指令完成后续计算；

// auto tzR2等行通过recast指令实现连续的half类型到half2类型的转换，以便能利用更高效的HFMA2指令；

// pragma 及后续for行实现了多个元素的z = ax + by + c的计算，并且通过括号将该计算通过两个HFMA2指令实现，如果没有括号，则其会生成 HMUL2 + HMUL2 + HADD2 + HADD2指令（由于乘法不满足结合律，且IEEE规定了浮点数计算的顺序需按照代码书写顺序）；

// 最后我们将结果cast回来并通过copy接口将计算结果存储到全局内存；

// 值得注意的是，Tensor tz; Tensor tzr; 行看似是生成Tensor但实际其并没有涉及到全局内存到读写（并没有Tensor被拷贝），只是利用Layout进行tensor的表达和变换数，数据实体没有移动，只有在copy的时候才有实际的数据读写。这也回应了文章开头的对cute中Tensor和深度学习框架中的Tensor不一样，大部分时间，我们在cute中使用Tensor只是使用Tensor的逻辑语义和变换，并没有实质的触发Tensor的搬运。同时我们也看到我们使用Tensor语义和工具能够更形象化的表达我们的逻辑，方便我们的思考，而CUDA的优化思路和技巧并不会因为Tensor的引入而变简单或困难。Tensor只是工具，可以方便我们的表达，至于深层次的优化思路那还是对经验的挑战。

//  nvcc -arch=sm_89 -I/mnt/bbuf/fused_moe_cutlass/third_party/cutlass/include -I/mnt/bbuf/fused_moe_cutlass/third_party/cutlass/tools/util/include -I/mnt/bbuf/fused_moe_cutlass/third_party/cub  vector_add.cu -o vector_add
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

using namespace cute;

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) { // 未处理非对齐问题
    return;
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma2 instruction
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

int main(int argc, char** argv)
{
    cute::device_init(0);

    thrust::host_vector<half> h_x(256);
    thrust::host_vector<half> h_y(256);
    thrust::host_vector<half> h_z(256);

    for (int j = 0; j < 256; ++j) h_x[j] = static_cast<half>(j);
    for (int j = 0; j < 256; ++j) h_y[j] = static_cast<half>(j);
    for (int j = 0; j < 256; ++j) h_z[j] = static_cast<half>(-1);

    thrust::device_vector<half> d_x = h_x;
    thrust::device_vector<half> d_y = h_y;
    thrust::device_vector<half> d_z = h_z;

    dim3 dimBlock(32);
    dim3 dimGrid(1);
    vector_add_local_tile_multi_elem_per_thread_half<<<dimGrid, dimBlock>>>(
        d_z.data().get(), 256, d_x.data().get(), d_y.data().get(), (half)10.0f, (half)1.0f, (half)0.5f);

    h_z = d_z;
    for (int i = 0; i < 256; ++i) {
        printf("%f\t", static_cast<float>(h_z[i]));
        if (i % 8 == 7) printf("\n");
    }

    return 0;
}

