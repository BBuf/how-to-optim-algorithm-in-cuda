# CUTLASS GEMM

An example implementation for integrating CUTLASS GEMM into PyTorch. 

# Installation

This example requires the `nvcc` compiler to compile the CUDA application.
Additionally, it needs the [CUTLASS]() repo. 
But because CUTLASS is a header-only library, no installation is needed for it.

To compile, run:
```
export CUTLASS_DIR=/path/to/cutlass
make
```

And you can run the example with:
```
python3 gemm.py
```

If you have a device with Compute Capability 9.0 or above, you can compile the NVIDIA Hopper enabled version with:
```
export CUTLASS_DIR=/path/to/cutlass
make hopper

``` 
