#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// File containing the CUTLASS portion of the code.
#include "include/copy.h"
#include "include/util.h"

// Once the datatypes are known, get the sizes and the pointers and call the CUTLASS part of the code.
template<typename T> void copy_cute_unpack(torch::Tensor input, torch::Tensor output) {
  // Get the input shapes
  const int M = input.sizes()[0];
  const int N = input.sizes()[1];

  // We cast the pointers to the type we need. We work with pointers instead of accessors.
  T *input_ptr  = reinterpret_cast<T*>(input.data_ptr());
  T *output_ptr = reinterpret_cast<T*>(output.data_ptr());
  TransposeParams<T> params = TransposeParams<T>(input_ptr, output_ptr, M, N);
  copy_baseline<T>(params);
}

// This function is bound to "copy_cute.copy". 
torch::Tensor copy_cute(torch::Tensor input,
                             c10::optional<torch::Tensor> output) {

  // Handling the optional output matrix.
  torch::Tensor _output;
  if(output.has_value()) {  // Output tensor was provided. So we will use it.
    _output = output.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = input.sizes()[0];
    const int N = input.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input.
    auto output_options = torch::TensorOptions().device(torch::kCUDA).dtype(input.dtype());
    _output = torch::empty({M, N}, output_options);
  }

  // Ensuring that the matrices are contiguous. 
  torch::Tensor _input  = input.contiguous();
  _output = _output.contiguous();

  // Check that all tensors are allocated on GPU device.
  if(!(_input.device().is_cuda() && _output.device().is_cuda()))
    throw std::invalid_argument("copy_cute only supports GPU device. Use .to(device=torch.device('cuda'))");

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(_input.dtype() == torch::kFloat16)
    copy_cute_unpack<cutlass::half_t>(_input, _output);
  else if(_input.dtype() == torch::kFloat32)
    copy_cute_unpack<float>(_input, _output);
  else
    throw std::invalid_argument("Unsupported precision type");

  // Return the Torch tensor back to PyTorch
  return _output;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", py::overload_cast<torch::Tensor,c10::optional<torch::Tensor>>(&copy_cute), py::arg("input"), py::arg("output") = py::none());
}
