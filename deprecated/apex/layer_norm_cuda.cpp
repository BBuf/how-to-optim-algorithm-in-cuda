#include <torch/extension.h>
#include <vector>
#include <cassert>
#include "compat.h"

namespace {
// 在 LayerNorm 中，通常将一个张量分为两部分：一部分进行标准化处理，另一部分则不受影响。n1 和 n2 分别代表这两部分的大小。
// 例如，如果你有一个形状为 [batch_size, channels, height, width] 的 4D 张量，并且你只想对最后两个维度进行 LayerNorm，
// 那么 n1 将是 batch_size * channels，而 n2 则是 height * width。
void compute_n1_n2(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input.sizes()[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input.sizes()[i];
    }
}

// 这个函数 check_args 是在执行 LayerNorm 之前对关键参数进行的一种安全检查，
// 确保 gamma 和 beta 的形状要么是未定义的，要么与正则化操作的目标形状相符。
void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

// 这个 check_args 函数通过确保 gamma 参数的形状正确，为 LayerNorm 操作的正确执行提供了一个安全检查
void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}


void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2
    )
{
    // 获取 normalized_shape 的维度数。
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    // 获取输入张量的形状。
    auto input_shape = input.sizes();
    // 获取输入张量的维度数。
    auto input_ndim = input.dim();

    // 这部分代码检查输入张量的维度数是否至少与 normalized_shape 一样多，
    // 并且在最后 normalized_ndim 个维度上与 normalized_shape 相匹配。
    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    // 调用 compute_n1_n2 函数，根据输入张量 input 和 normalized_shape 计算 n1 和 n2 的值。
    // n1 代表在 LayerNorm 中不受正则化影响的部分张量的元素总数。
    // n2 代表将要进行正则化的部分张量的元素总数。
    compute_n1_n2(input,normalized_shape,n1,n2);
}

// 这个 check_args 函数是 LayerNorm 操作的一个关键组成部分，用于确保所有输入参数都是正确的，
// 并且准备了执行 LayerNorm 所需的必要信息。通过组合两个不同的 check_args 调用，这个函数有效地验证了输入张量、
// 正则化形状以及可选的 gamma 和 beta 参数，同时计算出关键的 n1 和 n2 参数，从而为 LayerNorm 操作提供了必要的前置条件。
void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}

// 和前面一个 check_args 类似，不过只检查gamma
void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma);
}
}

//  at::Tensor* output, mean, invvar：分别指向输出张量、均值张量和逆方差张量的指针。
// at::Tensor* input: 指向输入张量的指针。
// int n1, n2: 与 LayerNorm 操作相关的维度参数。
// at::IntArrayRef 或 at::IntList normalized_shape: 表示在哪些维度上进行标准化的形状参数。
// at::Tensor* gamma, beta: 指向缩放（gamma）和偏移（beta）张量的指针。
// double epsilon: 用于数值稳定性的小量，避免除以零。
void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon);

// 这几个宏用来检查张量是否为cuda张量以及是否连续
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// layer_norm 函数是一个对 Layer Normalization 过程的封装，它在 CUDA 环境中处理输入张量，并返回标准化后的结果。
// 这个函数简化了在神经网络模型中使用 LayerNorm 的过程，尤其是在需要 GPU 加速时。通过预先计算必要的参数（如 n1 和 n2），
// 并将实际的计算工作委托给 cuda_layer_norm函数，该函数提供了一个高效且易于使用的接口来执行 LayerNorm 操作。
std::vector<at::Tensor> layer_norm(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor output = at::empty_like(input);
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type()==at::ScalarType::Half || input.scalar_type()==at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,NULL,NULL,epsilon);
  return {output, mean, invvar};
}

// affine表示有gamma和beta，这也是和上面的函数唯一的区别
std::vector<at::Tensor> layer_norm_affine(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input,normalized_shape,gamma,beta,n1,n2);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor mean = at::empty({n1}, input.options().dtype(stats_dtype));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,&gamma,&beta,epsilon);
  return {output, mean, invvar};
}

// 这里对half和bf16类型的input做了一个强制类型转换，转换为float
std::vector<at::Tensor> layer_norm_affine_mixed_dtypes(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  int n1, n2;
  check_args(input, normalized_shape, n1, n2);
  at::Tensor output = at::empty_like(input, gamma.options().dtype(gamma.scalar_type()));
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
   cuda_layer_norm(&output, &mean, &invvar, &input, n1, n2,
      normalized_shape, &gamma, &beta, epsilon);
  return {output, mean, invvar};
}

// at::Tensor* dout: 指向上游（后一层）的梯度张量。
// at::Tensor* mean, at::Tensor* invvar: 分别指向 LayerNorm 操作中计算得到的均值和逆方差张量。
// at::Tensor* input_or_output: 指向输入或输出张量，这取决于 memory_efficient 标志的设置。
// int n1, n2: 与 LayerNorm 操作相关的维度参数。
// at::IntArrayRef 或 at::IntList normalized_shape: 表示在哪些维度上进行标准化的形状参数。
// at::Tensor* gamma, at::Tensor* beta: 指向缩放（gamma）和偏移（beta）张量。
// double epsilon: 用于数值稳定性的小量，避免除以零。
// at::Tensor* grad_input, at::Tensor* grad_gamma, at::Tensor* grad_beta: 分别指向输入、gamma、beta 的梯度张量。
// bool memory_efficient: 标志，指示是否以内存效率优先的方式进行计算。
void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input_or_output,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta,
    bool memory_efficient
    );

at::Tensor layer_norm_gradient(
    at::Tensor dout,
    c10::optional<at::Tensor> mean_,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon,
    bool memory_efficient) {
  // 使用 CHECK_INPUT 宏确保 dout、invvar 和 input_or_output 是 CUDA 张量且在内存中连续。
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  int n1,n2;
  // 调用 check_args(input_or_output, normalized_shape, n1, n2) 函数计算 n1 和 n2 的值。
  check_args(input_or_output,normalized_shape,n1,n2);
  // 创建一个与 input_or_output 形状相同的空张量用于存储计算出的梯度。
  at::Tensor grad_input = at::empty_like(input_or_output);
  // 据 mean_ 是否有值，选择不同的方式调用 cuda_layer_norm_gradient。
  if (mean_.has_value()) {
    cuda_layer_norm_gradient(&dout,&mean_.value(),&invvar,&input_or_output,n1,n2,
        normalized_shape,NULL,NULL,epsilon,
        &grad_input,NULL,NULL,memory_efficient);
  } else {
    cuda_layer_norm_gradient(&dout,NULL,&invvar,&input_or_output,n1,n2,
        normalized_shape,NULL,NULL,epsilon,
        &grad_input,NULL,NULL,memory_efficient);
  }
  return grad_input;
}

std::vector<at::Tensor> layer_norm_gradient_affine(
    at::Tensor dout,
    c10::optional<at::Tensor> mean_,
    at::Tensor invvar,
    at::Tensor input_or_output,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon,
    bool memory_efficient) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input_or_output);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input_or_output,normalized_shape,gamma,beta,n1,n2);
  at::Tensor grad_input = at::empty_like(input_or_output);
  at::Tensor grad_gamma = at::empty_like(gamma);
  at::Tensor grad_beta = at::empty_like(beta);
//   at::Tensor *mean = mean_.has_value() ? &mean_.value() : NULL;
  if (mean_.has_value()) {
    cuda_layer_norm_gradient(&dout,&mean_.value(),&invvar,&input_or_output,n1,n2,
        normalized_shape,&gamma,&beta,epsilon,
        &grad_input,&grad_gamma,&grad_beta,memory_efficient);
  } else {
    cuda_layer_norm_gradient(&dout,NULL,&invvar,&input_or_output,n1,n2,
        normalized_shape,&gamma,&beta,epsilon,
        &grad_input,&grad_gamma,&grad_beta,memory_efficient);
  }
  return {grad_input, grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_affine", &layer_norm_affine, "LayerNorm forward (CUDA)");
  m.def("forward", &layer_norm, "LayerNorm forward (CUDA)");
  m.def("backward_affine", &layer_norm_gradient_affine, "LayerNorm backward (CUDA)");
  m.def("backward", &layer_norm_gradient, "LayerNorm backward (CUDA)");

  m.def("forward_affine_mixed_dtypes", &layer_norm_affine_mixed_dtypes, "LayerNorm forward with mixed dtypes (CUDA) compatible with Megatron's implementation");
}
