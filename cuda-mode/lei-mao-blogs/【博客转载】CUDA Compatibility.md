> 博客来源：https://leimao.github.io/blog/CUDA-Compatibility/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 兼容性

## 介绍

创建可移植的CUDA应用程序和库，使其能够在各种NVIDIA平台和软件环境中工作，有时是很重要的。NVIDIA在不同层面提供CUDA兼容性。

在这篇博客文章中，我想从CUDA应用程序或库兼容性（与GPU架构）、CUDA运行时兼容性（与CUDA应用程序或库）和CUDA驱动程序兼容性（与CUDA运行时库）的角度来讨论CUDA的前向和后向兼容性。

## CUDA应用程序兼容性

为了简化，我们首先假设我们的CUDA应用程序或库没有依赖其他CUDA库，如cuDNN、cuBLAS等。如果我们有一台计算机，它有一个旧架构的NVIDIA GPU，我们想要构建一个CUDA应用程序或库，该应用程序或库也能在有新架构NVIDIA GPU的计算机上运行，甚至是更未来的架构。当我们在有旧架构NVIDIA GPU的计算机上构建CUDA应用程序或库时，NVCC编译可以生成PTX代码作为编译后的二进制文件的一部分。在有新架构NVIDIA GPU的计算机上，当执行CUDA应用程序或库时，PTX代码将由CUDA运行时JIT编译为新架构的二进制文件，因此在有旧架构NVIDIA GPU的计算机上构建的应用程序或软件可以在有新架构NVIDIA GPU的计算机上前向兼容。

当然，前向兼容的缺点是为旧架构生成的PTX代码无法利用新架构的新特性，这可能会带来巨大的性能提升。我们在本文中不讨论性能，因为兼容性是我们想要实现的目标。

相反，如果我们有一台计算机，它有一个新架构的NVIDIA GPU，我们想要构建一个CUDA应用程序或库，该应用程序或库也能在有旧架构NVIDIA GPU的计算机上运行。NVCC编译允许我们不仅指定PTX代码，还可以为旧架构生成编译后的二进制文件。在有旧架构NVIDIA GPU的计算机上，当执行CUDA应用程序或库时，如果已为该架构编译了二进制文件，它将直接执行，或者PTX代码将由CUDA运行时JIT编译为旧架构的二进制文件，因此在有新架构NVIDIA GPU的计算机上构建的应用程序或软件可以在有旧架构NVIDIA GPU的计算机上后向兼容。

可以通过禁用PTX代码生成作为二进制文件的一部分来禁用前向兼容性。可以通过禁用PTX代码和旧架构特定二进制文件的生成作为二进制文件的一部分来禁用后向兼容性。

现在，如果我们的CUDA应用程序或库依赖其他CUDA库，如cuDNN、cuBLAS，为了实现前向或后向兼容性，这些CUDA库也应该构建为具有与我们的CUDA应用程序或库相同的前向或后向兼容性。然而，这有时并非如此，这使得我们的CUDA应用程序或库无法前向或后向兼容。应用程序或开发人员应该始终事先仔细检查依赖库的兼容性。

## CUDA运行时兼容性

CUDA运行时库是CUDA应用程序或库在大多数情况下在构建期间总是必须链接的库，有时不需要明确指定。例外情况是有些CUDA应用程序或库链接到CUDA驱动程序库。因此，对于发布的CUDA软件，如cuDNN，它总是会提到它链接的CUDA（运行时库）的版本。有时，也会有针对不同版本CUDA运行时库的多个构建。所以CUDA应用程序兼容性也取决于CUDA运行时。

然而，有时我们的CUDA应用程序或库在构建时链接的CUDA运行时与执行环境中的CUDA运行时库不同。为了解决这些问题，CUDA运行时库提供次版本（前向和/或后向）兼容性，前提是满足NVIDIA驱动程序要求。

CUDA明确提到兼容性是次版本兼容性的原因是，不同主版本之间的CUDA运行时API可能不同，使用这些API并针对一个主版本的CUDA运行时构建的应用程序或库可能无法与另一个主版本的CUDA运行时一起运行。例如，用于CUDA 10.2的cuDNN 8.6无法与CUDA 11.2一起使用。实际上，如果我们使用`ldd`检查CUDA应用程序或库的链接共享库，我们经常会看到指定了CUDA运行时库的主版本而没有指定次版本。对于次版本不同的CUDA运行时库，CUDA运行时API通常是相同的，因此，实现次版本兼容性成为可能。

## CUDA驱动程序兼容性

CUDA运行时库是在运行前构建应用程序组件的库，而CUDA驱动程序库是实际运行应用程序的库。所以CUDA运行时兼容性也取决于CUDA驱动程序。尽管每个版本的CUDA工具包发布都附带彼此兼容的CUDA运行时库和CUDA驱动程序库，但它们可以来自不同的来源并单独安装。

CUDA驱动程序库总是后向兼容的。使用最新的驱动程序允许我们运行带有旧CUDA运行时库的CUDA应用程序。CUDA驱动程序库前向兼容性(https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward)更加复杂，需要安装额外的库，这有时是专注于稳定性的数据中心计算机所需要的。我们在本文中不会对此进行太多阐述。

## NVIDIA Docker

NVIDIA Docker是一个便捷的工具，允许用户以可移植、可复现和可扩展的方式开发和部署CUDA应用程序。使用NVIDIA Docker，我们可以在任何我们想要的CUDA环境中构建和运行任何CUDA应用程序，前提是满足驱动程序和GPU架构要求。

![](https://files.mdnice.com/user/59/53ae4202-5972-4499-bd4f-6ec4c8d66571.png)

## 参考文献

- [CUDA Compilation](https://leimao.github.io/blog/CUDA-Compilation/)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [CUDA Application Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#application-compatibility)
- [CUDA Version and Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#versioning-and-compatibility)
- NVIDIA Docker: GPU Server Application Deployment Made Easy(https://developer.nvidia.com/blog/nvidia-docker-gpu-server-application-deployment-made-easy/)
- NVIDIA NGC User Guide(https://docs.nvidia.com/ngc/)


