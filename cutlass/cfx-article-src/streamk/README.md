# CUTLASS Stream-K

Example code for the CUTLASS Stream-K tutorial.

# Installation

This example requires the `nvcc` compiler to compile the CUDA application.
Additionally, it needs the CUTLASS repo.
But because CUTLASS is a header-only library, no installation is needed for it.

CUTLASS is added as a submodule for this repo.
If you cloned recursively, it should be downloaded in the parent directory.
Otherwise, specify the CUTLASS directory with:

```
export CUTLASS_DIR=/path/to/cutlass
```

To compile and run the example:
```
make
./streamk
```

`./streamk --help` displays command-line options for choosing different schedulers, tasks, and problem shapes.
