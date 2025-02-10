# CUTLASS EVT

Example code for the CUTLASS EVT tutorial.

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

To compile and run the custom EVT example:
```
make
./evt_gemm_cute 
```

# Appendix on EVT Node Types

The supplementary material on EVT node types can be found in the `node_types.md` file located in this directory.