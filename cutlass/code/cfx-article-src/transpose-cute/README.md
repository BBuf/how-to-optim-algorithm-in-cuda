# CuTe Transpose

Example code for the CuTe Transpose tutorial.

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

To compile and run the C++ example:
```
make
./transpose
```

To compile the python module and run the python example:
```
make python -B
python3 torch_benchmark.py
```
