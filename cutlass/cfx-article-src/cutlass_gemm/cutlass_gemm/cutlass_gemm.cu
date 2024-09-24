#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// File containing the CUTLASS portion of the code.
#include "cutlass_gemm.hpp"

// Not strictly necessary, but here for convenience.
template<typename DataType, typename OutputType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC);

// Once the datatypes are known, get the sizes and the pointers and call the CUTLASS part of the code.
template<typename DataType, typename OutputType> void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get the input shapes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0]; 
  const int N = B.sizes()[1];

  // We cast the pointers to the type we need. We work with pointers instead of accessors because CUTLASS requires pointers.
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  OutputType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}

// Intermediate function to get the output precision to use for the wrapper template. 
template<typename DataType> void cutlass_gemm_find_output_type(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  if(C.dtype() == torch::kFloat16)
    cutlass_gemm_unpack<DataType, cutlass::half_t>(A, B, C);
  else if(C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<DataType, float>(A, B, C);
  else
    throw std::invalid_argument("Unsupported precision type");
}

// This function is bound to "cutlass_gemm.mm". 
torch::Tensor cutlass_gemm(torch::Tensor A,  // A matrix (m x k)
                           torch::Tensor B,  // B matrix (k x n)
                           c10::optional<torch::Tensor> out) {   // optional out matrix (m x n)


  // Handling the optional C matrix.
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input.
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use .to(device=torch.device('cuda'))");



  // Ensuring that the matrices are contiguous. 
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(_A.dtype() == torch::kFloat16)
    cutlass_gemm_find_output_type<cutlass::half_t>(_A, _B, _C);
  else if(_A.dtype() == torch::kFloat32)
    cutlass_gemm_find_output_type<float>(_A, _B, _C);
  else
    throw std::invalid_argument("Unsupported precision type");

  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", py::overload_cast<torch::Tensor,torch::Tensor,c10::optional<torch::Tensor>>(&cutlass_gemm), py::arg("A"), py::arg("B"), py::arg("out") = py::none());
}
