# TMA examples

Example code for the TMA tutorial. There are three kernels of interest:

1) GMEM -> GMEM copy kernel with TMA load and store.

2) GMEM -> GMEM scale and copy kernel with TMA load and store.

3) GMEM -> multiple GMEM copy kernel with clusters and optional TMA multicast.

# Building and running

Run `git submodule update --init --recursive` once to pull the CUTLASS submodule.

Then to compile and run the example kernels, do the following:

```
make
./main
```
