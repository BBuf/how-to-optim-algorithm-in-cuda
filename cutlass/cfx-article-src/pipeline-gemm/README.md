# CUTLASS Hopper GEMMs with pipelined kernel design

Example code for the GEMM pipelining tutorial.
Contains both multistage and warp-specialized kernel designs.

# Installation

This example requires the `nvcc` compiler to compile the CUDA application.
Additionally, it needs the [CUTLASS]() repo. 
But because CUTLASS is a header-only library, no installation is needed for it.

CUTLASS is added as a submodule for this repo.
If you cloned recursively, it should be downloaded in the parent directory.
Otherwise, specify the CUTLASS directory with:

```
export CUTLASS_DIR=/path/to/cutlass
```

To compile and run the warp-specialized example:
```
make
./sm90_gemm_ws
```

To compile and run the multistage example, change 'APP' in the Makefile to point to the other .cu file.

# FP32 vs FP16 accumulation

Kernels are compiled to use FP32 accumulation by default.

To try out a version with FP16 accumulation, comment in the relevant option directly in the code.

In the WS kernel, this is done at the top of "hopper_gemm_kernel_launch.h".

In the multistage kernel, this is done in the host function "gemm_tn" within the single file.