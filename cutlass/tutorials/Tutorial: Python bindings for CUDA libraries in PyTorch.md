> 博客来源：https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/ ，这里做了一个翻译学习一下。

# 教程: 在PyTorch中为CUDA库绑定Python接口

PyTorch是当今最受欢迎的AI框架之一。它由Meta(前Facebook)开发并于2017年开源,具有简洁友好的"Python式"接口。这种易用性使其特别适合于研究和开发领域,研究人员可能需要多次迭代新的AI工作负载。然而,纯Python开发也存在一些缺陷,常见的主要缺陷之一就是性能问题。Python通常比诸如C++这样的语言运行速度较慢,尤其是当Python代码根本没有利用GPU硬件加速,或者只以较为简单的方式加速时(例如,未针对特定GPU架构的特殊特性进行优化)。

为了充分利用PyTorch针对NVIDIA®GPU的代码优化,最简单的方法之一是让PyTorch调用经过优化的GPU加速库。虽然PyTorch已经为许多常见的AI工作负载做了这些,但并非所有工作负载都已整合。对于某些工作负载,可能存在比PyTorch默认使用的库性能更好的CUDA®C++库。

此外,创建新的CUDA库的开发人员可能希望通过移植到PyTorch来提高库的可访问性。虽然像PyCUDA这样的库可以从Python调用CUDA,但C++仍然是CUDA开发的主要语言。因此,CUDA开发人员可能需要将其C++函数绑定到可与PyTorch一起使用的Python调用。

PyTorch网站已经有一个非常有用的指南（https://pytorch.org/tutorials/advanced/cpp_extension.html）,逐步介绍了编写C++扩展的过程。在本文中,我们将介绍一些在使用CUDA和CUDA库(如CUTLASS)时发现的补充信息。为了解释这些信息,我们将介绍一个PyTorch的C++扩展示例,该扩展使用NVIDIA的CUTLASS库进行通用矩阵乘法(GEMM)运算。我们将以torch.mm（https://pytorch.org/docs/stable/generated/torch.mm.html）为模板设计Python端接口,以便可以直接作为替换品使用。我们的目标是创建一个完整的可运行示例,作为未来开发的模板。

## 从Torch到CUTLASS的输入转换

我们将基于CUTLASS basic_gemm example 0（https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu）实现。对于熟悉CUTLASS的人,请注意这个例子使用了2.X语法。我们还将在本文的附录中提供一个使用3.X语法针对NVIDIA Hopper™架构的单独示例。

首先,为了简化,我们将该示例封装在一个单一函数调用中:

![](https://files.mdnice.com/user/59/21fe5ff0-64b8-4b52-a574-284f3d99f231.png)

然后我们将重点关注获取此调用所需的参数。具体来说,我们需要三样东西:

- 张量的形状,
- 张量的数据类型,和
- 指向数据的指针。

我们的目标是创建一个接收PyTorch输入的函数,提取上述信息,并调用CUTLASS包装器函数。

### 输入Torch张量

我们新函数cutlass_gemm的输入参数将采用`torch::Tensor`类的形式,这是Python中`torch.Tensor`类在C++的表示形式。例如,该函数的签名可以是:

```c++
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C)
```

注意在上面的代码中,矩阵C作为必需参数出现,尽管对于mm来说它是可选的。我们将在后面的部分解决这个问题。

### 张量形状

为了提取GEMM所需的数据,我们可以利用PyTorch ATen API。首先,我们可以使用`.sizes()`方法获取张量的形状:

`auto A_shape = A.sizes();`

这将返回一个包含张量形状的数组(具体来说,是Torch的`IntArrayRef`)。

### 张量数据类型

接下来是数据类型。Torch张量有多种可能的数据类型,可以使用`.dtype()`方法获取:

`auto A_type = A.dtype();`

然后可以将其与Torch数据类型进行比较:

`bool is_half = (A.dtype() == torch::kHalf);`

不同数据类型的完整列表可以在这里(https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h)找到。

### 张量数据指针

最后,我们可以使用张量的`.data_ptr()`方法提取数据的指针:

`float* A_ptr = A.data_ptr<float>();`

这里的`.data_ptr()`是模板化的,允许开发者将返回的指针转换为所需的数据类型。注意,如果你的应用程序只处理默认数据类型,这种模板化就足够了,但它不支持自定义数据类型。例如,在CUTLASS中,FP16数据类型是`cutlass::half_t`,而相应的FP16数据类型的`.data_ptr()`模板是torch::kFloat16。

因此,我们不使用模板化,而是使用`reinterpret_cast`来转换为所需的数据类型:

`float* A_ptr = reinterpret_cast<float*>(A.data_ptr());`

对于我们的示例,我们将让CUTLASS使用用户输入的任何数据类型。因此,我们可以使用在上一步找到的数据类型,并转换为正确的精度。为此,我们将`reinterpret_cast`放在一个中间函数中,并使用C++模板传递数据类型。

```c++
template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];
 
  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}
``

注意模板参数是在编译时解析的,但这里我们需要根据A和C的数据类型选择cutlass_gemm_unpack的正确模板实例,我们在运行时是知道的。为此,我们可以引入一些条件逻辑,例如:

```c++
if(A.dtype() == torch::kFloat16 && C.dtype() == torch::kFloat32)
cutlass_gemm_unpackcutlass::half_t,float(A, B, C);
// ...
```

实际上,我们不会完全按照这种方式编写代码。在讨论了一些其他重要点之后,我们将展示完整的程序。

## 输入验证

现在我们已经获取了输入及其相关信息,让我们检查这些输入是否有效。通过访问张量形状和数据类型,一些较为简单的检查(例如,矩阵乘法的维度兼容性)应该是不言自明的。因此,我们将重点关注一些更加特定于Torch和CUTLASS的主题。

CUTLASS对矩阵乘法的一个限制是,它必须是连续的,这意味着相邻元素在内存中也是相邻的。由于PyTorch张量按行优先存储,一个连续的张量就是其中元素在同一行和相邻列是在内存中彼此相邻的。我们可以用`.is_contiguous()`方法检查一个张量是否连续。

```c++
bool a_contiguous = A.is_contiguous();
```

如果一个张量不是连续的,可以使用`.contiguous()`方法使其变为连续。

```c++
torch::Tensor _A = A.contiguous();
```

如果原始张量已经是连续的,这个方法只是简单地返回原始张量。然而,如果不是连续的,它会创建一个新的连续张量。对于输入矩阵A和B,这不是问题,但对于C矩阵来说就是个问题,因为`torch.mm`支持inplace操作。因此对于C矩阵,如果必要的话,我们将使用`.copy_()`复制数据。

```c++
torch::Tensor _C = C.contiguous();

// ... GEMM 操作 ... //

if(!C.is_contiguous())
C.copy_(_C);
return C
```

另一个限制是数据必须在GPU设备上。我们可以轻松检查:

```c++
bool is_cuda = A.device().is_cuda();
```

我们的库只针对GPU构建。如果数据需要在主机上分配,我们会在Python中使用`.to()`方法将其移动到设备上。虽然在C++中使用`.to()`自动移动数据到设备是可能的,但这种行为与大多数其他PyTorch函数不一致,因此如果设备不是GPU,我们将直接抛出错误。

## 让C成为可选


与PyTorch的`mm`类似，我们的函数将会把C张量返回给PyTorch以供使用。我们还需要更新函数参数以将C标记为可选。Torch C++ API提供了一个工具`c10::optional<torch::Tensor>`来指定Tensor参数为可选。有了这个，我们可以用`.has_value()`方法检查是否提供了输入。如果这返回true，我们就可以用`.value()`方法获取值。

如果`.has_value()`返回false，那么我们需要创建一个新的张量。ATen有很多创建张量的选项，这些选项在这里有文档(https://pytorch.org/cppdocs/notes/tensor_creation.html)说明。对我们的目的来说，我们只需要一个空张量。综合起来，我们得到：

```c++
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> out) { 
 
  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];
 
    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }
 
  // ... Rest of the GEMM workload ...//
}
```

当创建新矩阵时，我们设置options以将设备设为GPU，并将数据类型设置为与输入张量相同。建议在创建新张量时使用ATen库。虽然可以从现有的数据指针创建新的torch::Tensor，但这意味着ATen不拥有该数据。这可能会限制某些操作，比如一旦张量传回Python后进行调整大小。所以虽然CUTLASS有特殊的分配器如`HostTensor`，我们不会使用它们。

## 放在一起

把上面讲到的放在一起得到完整代码：

```c++
template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];
 
  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}
 
// Intermediate function to get the output precision to use for the wrapper template. 
template<typename DataType>
void cutlass_gemm_find_output_type(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  if(C.dtype() == torch::kFloat16)
    cutlass_gemm_unpack<DataType, cutlass::half_t>(A, B, C);
  else if(C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<DataType, float>(A, B, C);
  else
    throw std::invalid_argument("Unsupported precision type");
} 
 
// This function is bound to "cutlass_gemm.mm". Takes torch::Tensors as inputs
torch::Tensor cutlass_gemm(torch::Tensor A,  // A matrix (m x k)
                           torch::Tensor B,  // B matrix (k x n)
                           c10::optional<torch::Tensor> out) {  // optional out matrix (m x n)
  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];
    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }
 
  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device.
                                 Use .to(device=torch.device('cuda'))");
 
  // Ensuring that the matrices are contiguous. 
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();
 
  // Select the CUTLASS precision type to use based on Torch input data type.
  if(A.dtype() == torch::kFloat16)
    cutlass_gemm_find_output_type<cutlass::half_t>(_A, _B, _C);
  else if(A.dtype() == torch::kFloat32)
    cutlass_gemm_find_output_type<float>(_A, _B, _C);
  else
    throw std::invalid_argument("Unsupported precision type");
 
  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);
 
  // Return the Torch tensor back to PyTorch
  return C;
}
```

在此代码中，我们采用了一种临时性的方法来处理根据A和C的数据类型向适当的模板函数dispatch所需的条件逻辑。显然，这种方法对于大量模板参数来说是不可扩展的。对于如何使用Python脚本来处理编写高度模板化的C++/CUDA函数（如CUTLASS中的那些）的包装器的示例，我们建议查看CUTLASS库中的_python_gemm(https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/emit/pytorch.py#L704)方法和EmitGemmUniversalInstance3x(https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/backend/gemm_operation.py#L1195)类。

## 绑定和编译

现在我们有了函数，让我们编译它并将其绑定到一个Python函数。我们将使用PyBind11结合setuptools(https://setuptools.pypa.io/en/latest/userguide/index.html)来完成这一步骤。我们不会对这些工具进行全面讨论，而只会涉及与我们直接相关的内容。

### PyBind11

我们函数的绑定如下：

```python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", 
        py::overload_cast<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>(
          &cutlass_gemm), 
        py::arg("A"), 
        py::arg("B"), 
        py::arg("out") = py::none());
}
```

我们还将第三个参数指定为关键字参数"out"，与torch.mm保持一致，并将其默认值设置为Python的None。

### setuptools

不幸的是，setuptools本身不支持nvcc，即CUDA编译器。虽然有一种解决方法，但可能相当复杂。幸运的是，PyTorch提供了一个名为CUDAExtension的实用工具，可以编译CUDA代码。

```c++
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
 
### ... set up lists cutlass_include_dirs, nvcc_flags, and ld_flags ... ###
setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(name="cutlass_gemm",
                      sources=["cutlass_gemm.cu"],
                      include_dirs=cutlass_include_dirs,
                      extra_compile_args={'nvcc': nvcc_flags},
                      libraries=ld_flags)
    ],
    cmdclass={'build_ext': BuildExtension})
```

参数的语法与基础Extension类相同。但是，它会自动添加所有必要的Torch库标志。因此，我们唯一需要做的就是添加CUTLASS的路径。由于CUTLASS是一个仅头文件的库，我们只需设置include_dir。一旦运行setup.py，我们现在就可以从PyTorch代码中访问新模块`cutlass_gemm`了。

## 使用PyTorch调用我们的新mm函数

这是一个简单的PyTorch脚本，使用我们的新函数执行CUTLASS GEMM。

```python
import math
import cutlass_gemm
 
M = K = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)
 
C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()
 
C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
```

我们指定`.to(device=cuda)`以使A和B在GPU上可访问，并对两个矩阵使用FP16精度。此外，我们还有一个与`torch.mm`的验证步骤，显示与Torch版本的最大偏差。

```shell
cutlass_gemm.mm 结果：
tensor([[-0.0045, -0.0139, 0.0109, ..., 0.0192, -0.0117, 0.0083],
[ 0.0110, 0.0005, -0.0079, ..., 0.0106, -0.0012, -0.0083]],
device='cuda:0', dtype=torch.float16)

torch.mm 结果：
tensor([[-0.0045, -0.0139, 0.0109, ..., 0.0192, -0.0117, 0.0083],
[ 0.0110, 0.0005, -0.0079, ..., 0.0106, -0.0012, -0.0083]],
device='cuda:0', dtype=torch.float16)

最大偏差：0.0000610352
```

在这里，我们可以看到结果矩阵确实使用了FP16精度格式，并且我们得到的结果（在误差范围内）与`torch.mm`相同。所以现在我们可以使用这个优化的GEMM来替代`torch.mm`。

## 代码下载

完整的代码请到这里查看：https://github.com/ColfaxResearch/cfx-article-src

## 附录A：AMP支持

PyTorch有一个名为自动混合精度（AMP）的功能，可用于简化混合精度工作负载。它围绕autocast上下文展开，在此上下文中，操作会在适当时自动使用较低精度。这可以带来显著的性能改进。

我们的示例不支持此功能，但你可以在这里找到更多关于C++包中AMP支持(https://pytorch.org/tutorials/advanced/dispatcher.html#autocast)的信息。

## 附录B：CUTLASS 3.X和Hopper架构

如前所述，上面的示例使用CUTLASS 2.X语法。在我们的github中，我们还提供了基于`hopper_warp_specialized_gemm`的CUTLASS 3.X示例（示例48(https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu)）。然而，在本文的范围内，2.X和3.X CUTLASS所需的内容没有区别。我们的3.X示例仍然在包装函数中包含所有CUTLASS代码。有关CUTLASS 3.X以及如何针对特定架构进行优化的更多信息，请参阅CUTLASS文档。

## 附录C：构建后端

在本文中，我们的重点是编写可与PyTorch一起使用的扩展。为此，我们使用setuptools作为构建后端，结合PyTorch的CUDAExtension实用类。但是，这会将PyTorch作为我们扩展的依赖项，如果该扩展不是为PyTorch开发的，这可能不理想。可以使用setuptools而不依赖CUDAExtension。有关示例，请参见CUTLASS的Python安装。

此外，还有其他与nvcc兼容的构建后端，可用于创建基于C/C++的Python扩展。例如，scikit-build-core(https://github.com/scikit-build/scikit-build-core)是一个基于cmake的后端，可以代替setuptools使用。在Nvidia开发者论坛上有一个关于在cmake中使用nvcc的指南(https://developer.nvidia.com/blog/building-cuda-applications-cmake/)。

最后，构建后端通常在`pyproject.toml`文件中指定，然后由Python打包软件使用。有关pyproject.toml及其用法的详细信息可以在这里找到(https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)。

## 完整代码补充

### cutlass_gemm.hpp

代码链接：https://github.com/ColfaxResearch/cfx-article-src/blob/master/cutlass_gemm/cutlass_gemm/cutlass_gemm.hpp ，这个代码是从cutlass那边拷贝过来的，具体见代码开头的注释，并且分别针对Hopper架构和Hopper之前的架构给出了不同的实现。

```c++
#ifndef COMPILE_3X_HOPPER

// CUTLASS 2.X syntax GEMM
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu

#include <cutlass/gemm/device/gemm.h>

template<typename DataType, typename OutputType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC) {
  using Gemm = cutlass::gemm::device::Gemm<
    DataType,                     // ElementA
    cutlass::layout::RowMajor,    // LayoutA
    DataType,                     // ElementB
    cutlass::layout::RowMajor,    // LayoutB
    OutputType,                     // ElementOutput
    cutlass::layout::RowMajor,    // LayoutOutput
    float                         // ElementAccumulator
  >;

  float alpha = 1.0f;
  float beta = 0.0f;

  int lda = M;
  int ldb = K;
  int ldc = M;

  Gemm gemm_op;
  gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrC, ldc},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });
}

#else

// CUTLASS 3.X syntax GEMM
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template<typename DataType, typename OutputType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC) {


  // A matrix configuration
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<OutputType>::value;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TilesShape          = Shape<_128,_128,_64>;                           // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
  using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder 

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      DataType, LayoutA, AlignmentA,
      DataType, LayoutB, AlignmentB,
      ElementAccumulator,
      TilesShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TilesShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    OutputType, LayoutC, AlignmentC,
    OutputType, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  float alpha = 1.00f;
  float beta = 0.0f;

  //
  // Allocate device memory
  //
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, Int<1>{}));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, Int<1>{}));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, Int<1>{}));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, Int<1>{}));

  //
  // Launch GEMM on the device
  //
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptrA, stride_A, ptrB, stride_B},
    {{alpha, beta}, ptrC, stride_C, ptrC, stride_D}
  };
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  gemm_op.can_implement(arguments);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  gemm_op.initialize(arguments, workspace.get());

  // Correctness / Warmup iteration
  gemm_op.run();

}
#endif
```

这篇博客讲到的Pybind相关的完整代码在上面《放在一起》那一节已经完整展示了。



 