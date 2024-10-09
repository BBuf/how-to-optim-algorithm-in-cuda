#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// 包含CUTLASS部分的代码文件
#include "include/copy.h"
#include "include/util.h"

// 一旦数据类型已知，获取大小和指针并调用CUTLASS部分的代码
template<typename T> void copy_cute_unpack(torch::Tensor input, torch::Tensor output) {
  // 获取输入形状
  const int M = input.sizes()[0];
  const int N = input.sizes()[1];

  // 我们将指针转换为我们需要的类型。我们使用指针而不是访问器。
  T *input_ptr  = reinterpret_cast<T*>(input.data_ptr());
  T *output_ptr = reinterpret_cast<T*>(output.data_ptr());
  TransposeParams<T> params = TransposeParams<T>(input_ptr, output_ptr, M, N);
  copy_baseline<T>(params);
}

// 这个函数绑定到 "copy_cute.copy"
torch::Tensor copy_cute(torch::Tensor input,
                             c10::optional<torch::Tensor> output) {

  // 处理可选的输出矩阵
  torch::Tensor _output;
  if(output.has_value()) {  // 提供了输出张量。所以我们使用它。
    _output = output.value();
  } else {               // 没有提供输出张量。创建一个空张量。
    const int M = input.sizes()[0];
    const int N = input.sizes()[1];

    // 我们将在GPU上分配矩阵，并将其数据类型设置为与输入相同。
    auto output_options = torch::TensorOptions().device(torch::kCUDA).dtype(input.dtype());
    _output = torch::empty({M, N}, output_options);
  }

  // 确保矩阵是连续的。
  torch::Tensor _input  = input.contiguous();
  _output = _output.contiguous();

  // 检查所有张量是否分配在GPU设备上。
  if(!(_input.device().is_cuda() && _output.device().is_cuda()))
    throw std::invalid_argument("copy_cute 仅支持GPU设备。使用 .to(device=torch.device('cuda'))");

  // 根据Torch输入数据类型选择要使用的CUTLASS精度类型。
  if(_input.dtype() == torch::kFloat16)
    copy_cute_unpack<cutlass::half_t>(_input, _output);
  else if(_input.dtype() == torch::kFloat32)
    copy_cute_unpack<float>(_input, _output);
  else
    throw std::invalid_argument("不支持的精度类型");

  // 将Torch张量返回给PyTorch
  return _output;
}

// 将函数绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", py::overload_cast<torch::Tensor,c10::optional<torch::Tensor>>(&copy_cute), py::arg("input"), py::arg("output") = py::none());
}
