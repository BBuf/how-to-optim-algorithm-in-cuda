# how-to-optim-algorithm-in-cuda

CUDA, GPU kernel, and AI infrastructure optimization notes.

This repository collects hands-on CUDA kernels, CUTLASS/CuTe notes, Triton examples, PTX ISA notes, PyTorch internals notes, and LLM inference/training optimization material. It is one of my main public study and engineering notebooks for GPU systems work.

## Repository Map

- `cuda-kernels/`: handwritten CUDA kernels for reduce, softmax, elementwise, GEMV, indexing, atomic add, upsampling, and linear attention.
- `cuda-mode/`: notes and code from the CUDA-MODE lecture series.
- `cutlass/`: CUTLASS and CuTe DSL notes, including GEMM, TMA, WGMMA, swizzling, and instruction-level material.
- `triton/`: Triton kernels, PyTorch interop examples, and meetup notes.
- `large-language-model/`: LLM serving, training, and systems optimization notes.
- `pytorch/`: PyTorch internals and CUDA-related notes.
- `papers/`: GPU architecture and ML systems paper notes.
- `ptx-isa/`: PTX ISA study notes.
- `tools/`: small helper scripts.
- `deprecated/`: older material kept for reference.

## Related Repositories

- CUDA and GPU optimization: this repository.
- Deep learning compiler notes: https://github.com/BBuf/tvm_mlir_learn
- Deep learning framework notes: https://github.com/BBuf/how-to-learn-deep-learning-framework

## Status

Actively curated around CUDA kernels, LLM inference optimization, and AI infrastructure. Older Chinese-language notes are being consolidated or replaced with English entry points.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BBuf/how-to-optim-algorithm-in-cuda&type=Date)](https://star-history.com/#BBuf/how-to-optim-algorithm-in-cuda&Date)
