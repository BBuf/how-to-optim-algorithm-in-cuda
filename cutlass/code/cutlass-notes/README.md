# CUTLASS Notes

The CUTLASS notes series will begin with a minimal GEMM implementation, gradually expand to incorporate CuTe and various CUTLASS components, as well as features of new architectures, e.g. Hopper and Blackwell, ultimately achieving a high-performance fused GEMM operator.


## Usage

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git

make update  # clone cutlass
```

## Run sample code

All example code in this GitHub repository can be compiled and run by simply executing the Python script. For example:

```bash
cd 01-minimal-gemm
python minimal_gemm.py
```

## Note list

| Notes                     | Summary                                                                                              | Links                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-Intro**              | Brief introduction to CUTLASS | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963) |
| **01-minimal-gemm**       | - Introduces CuTe fundamentals<br>- Implements 16x8x8 GEMM kernel using single MMA instruction from scratch<br>- Python kernel invocation, precision validation & performance benchmarking<br>- Profiling with Nsight Compute (ncu) | [minimal-gemm](https://zhuanlan.zhihu.com/p/1937517614084650073) |
| **02-mixed-precision-gemm** | - Implements mixed-precision GEMM supporting varying input/output/accumulation precisions<br>- Explores technical details for numerical precision conversion within kernels<br>- Demonstrates custom FP8 GEMM kernel implementation via PTX instructions (for CUTLASS-unsupported MMA ops) | [mixed-precision-gemm](https://zhuanlan.zhihu.com/p/1940158874255602181) |
| **03-tiled-mma** | - Introduces the key conceptual model of GEMM operator: Three-Level Tiling<br>- Details the implementation of Tiled MMA operations in CUTLASS CuTe<br>- Explains the usage and semantics of various parameters in the Tiled MMA API<br>- Extends the GEMM kernel from single instruction to single tile operation | [tiled-mma](https://zhuanlan.zhihu.com/p/1950555644814946318) |
| **04-tiled-copy** | - Explains the core principles of CuTe TiledCopy and its role in data movement between global and shared memory<br>- Describes the API parameters and semantics of TiledCopy<br>- Demonstrates how to implement data copying at the Tile level<br>- Introduces foundational knowledge of GPU global memory access characteristics | [tiled-copy](https://zhuanlan.zhihu.com/p/1968745447741972494) |
| **05-block-mma** | - Extends Tiled MMA to the Block level for larger-scale GEMM computations<br>- Explains how multiple Tiled MMA operations are combined within a thread block<br>- Describes the tiling and coordination of TiledCopy and TiledMMA at the Block level<br>- Illustrates the hierarchical dataflow from global memory to shared memory to registers for Block-level MMA | [block-mma](https://zhuanlan.zhihu.com/p/1970162570636816559) |
| **06-block-copy** | *Coming soon* | *Stay tuned* |


## License

This project is licensed under the MIT License - see the LICENSE file for details.