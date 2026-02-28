#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// File containing the CUTLASS portion of the code.
#include "include/transpose_naive.h"
#include "include/transpose_smem.h"
#include "include/transpose_tmastore_vectorized.h"
#include "include/util.h"

// Different versions of transpose
enum Version {
  naive=0,
  smem,
  swizzle,
  tma
};

// Once the datatypes are known, get the sizes and the pointers and call the CUTLASS part of the code.
template<typename T> void transpose_cute_unpack(torch::Tensor input, torch::Tensor output, Version ver) {
  // Get the input shapes
  const int M = input.sizes()[0];
  const int N = input.sizes()[1];

  // We cast the pointers to the type we need. We work with pointers instead of accessors.
  T *input_ptr  = reinterpret_cast<T*>(input.data_ptr());
  T *output_ptr = reinterpret_cast<T*>(output.data_ptr());
  TransposeParams<T> params = TransposeParams<T>(input_ptr, output_ptr, M, N);
  if(ver == naive) 
    transpose_naive<T>(params);
  else if(ver == smem) 
    transpose_smem<T, false>(params);
  else if(ver == swizzle) 
    transpose_smem<T, true>(params);
  else if(ver == tma) 
    transpose_tma<T>(params);
}

std::string get_version_info(Version const ver) {
  if(ver == naive) 
    return "Naive (no tma, no smem, not vectorized):";
  else if(ver == smem) 
    return "SMEM transpose (no tma, smem passthrough, not vectorized, not swizzled):"; 
  else if(ver == swizzle) 
    return "Swizzle (no tma, smem passthrough, not vectorized, swizzled):";
  else if(ver == tma) 
    return "TMA (tma, smem passthrough, vectorized, swizzled):";
}

// This function is bound to "transpose_cute.transpose". 
torch::Tensor transpose_cute(torch::Tensor input,
                             c10::optional<torch::Tensor> output,
                             Version const ver) {

  // Handling the optional output matrix.
  torch::Tensor _output;
  if(output.has_value()) {  // Output tensor was provided. So we will use it.
    _output = output.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = input.sizes()[0];
    const int N = input.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input.
    auto output_options = torch::TensorOptions().device(torch::kCUDA).dtype(input.dtype());
    _output = torch::empty({N, M}, output_options);
  }

  // Ensuring that the matrices are contiguous. 
  torch::Tensor _input  = input.contiguous();
  _output = _output.contiguous();

  // Check that all tensors are allocated on GPU device.
  if(!(_input.device().is_cuda() && _output.device().is_cuda()))
    throw std::invalid_argument("transpose_cute only supports GPU device. Use .to(device=torch.device('cuda'))");

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(_input.dtype() == torch::kFloat16)
    transpose_cute_unpack<cutlass::half_t>(_input, _output, ver);
  else if(_input.dtype() == torch::kFloat32)
    transpose_cute_unpack<float>(_input, _output, ver);
  else
    throw std::invalid_argument("Unsupported precision type");

  // Return the Torch tensor back to PyTorch
  return _output;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<Version>(m, "version")
      .value("naive", naive)
      .value("smem", smem)
      .value("swizzle", swizzle)
      .value("tma", tma)
      .export_values();
  m.def("transpose", py::overload_cast<torch::Tensor,c10::optional<torch::Tensor>,Version>(&transpose_cute), py::arg("input"), py::arg("output") = py::none(), py::arg("version")=swizzle);
  m.def("get_version_info",&get_version_info);
}
