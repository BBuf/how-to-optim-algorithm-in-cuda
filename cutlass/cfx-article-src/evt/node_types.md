## Appendix: EVT node types
- All node types define a `ProducerLoadCallbacks` and/or `ConsumerStoreCallbacks` which inherit from the ones defined in `Sm90VisitorImpl`. They will overload one or more of the methods that appear in the epilogue.
	- For producers, these methods are `.begin(), .step(), .end()`.
	- For consumers, they are `.begin(), .begin_loop(), .previsit(), .visit(), .reduce(), .postreduce(), .tma_store(), .end_loop(), .end()`.
	- The base implementations of each of these methods just call them recursively on all children.
	- The exception is `.visit()`, which must be overloaded by each consumer node type.
- For each node type, I'll say which methods it overloads and what Arguments are needed to construct it. Any data not coming from Arguments must come from the GEMM-submitted ConsumerStoreArgs or ProducerLoadArgs.
### Sm90TreeVisitor = Sm90EVT
```cpp
template<class NodeOp, class... ChildOps> struct Sm90TreeVisitor;
template <class NodeOp, class... ChildOps>
using Sm90EVT = Sm90TreeVisitor<NodeOp, ChildOps...>;
struct Arguments {
  child_1_args,
  ...,
  child_n_args,
  node_op_args
}
```
These nodes are used to form the tree structure: when visited, they recursively visit each of their children.

The Arguments to a tree visitor node consist of the Arguments to each of its ChildOps, followed by the Arguments to its NodeOp.

**Methods overloaded:** `.visit()`.

### Sm90Compute
```cpp
template<template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  class FloatRoundStyle>
struct Sm90Compute;
```
Both `ElementOutput` and `ElementCompute` are dtypes (as is any template argument prefixed by `Element` in this appendix). `FloatRoundStyle` is a [rounding style](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h) for floating-point arguments. `ComputeFn` is a template parametrized by the dtype `ElementCompute` -- examples can be found in `cutlass/epilogue/thread` and in `cutlass/functional.h`.

The `Arguments` to an `Sm90Compute` node are the same as the `Arguments` to its `ComputeFn`. For example, `cutlass::epilogue::thread::Clamp` requires two arguments, `lower_bound` and `upper_bound`.

**Methods overloaded:** `.visit()`.

### Sm90AccFetch
```cpp
struct Sm90AccFetch;
```
This takes no arguments. Its only purpose is to return values from the accumulator.

**Methods overloaded:** `.visit()`.

### Sm90SrcFetch
```cpp
template <class ElementC> struct Sm90SrcFetch;
```
This also takes no arguments. It returns values from C. Note that the location and layout of C are assumed known to the kernel, and the kernel is assumed to have loaded the appropriate fragment of C into SMEM by the time this node is accessed. To allow C to be loaded into RMEM, a TiledCopy is passed to ConsumerStoreCallbacks at initialization.

`ElementC` isn't used to do dtype conversion. It's only used to check if it's `void`. If it is, then the node instead treats C as 0 (so we don't need to load C).

**Methods overloaded:** `.visit()`.
On `.visit()`, return the value of `tCrC` at index `epi_v`. The presence of this node in the tree signals that the producer warps have to load C. (Again, C is handled specially, so this node does not have to define producer callbacks!)

### Sm90AuxLoad
```cpp
template <int Stages, // load pipeline stages
  class EpilogueTile,
  class ElementAux,
  class StrideMNL, // stride of aux tensor
  class SmemLayoutAtom,
  class CopyOpS2R,
  int Alignment = 128 / sizeof_bits_v<Element>, // must be multiple of 16B for TMA
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Sm90AuxLoad;
struct Arguments {
  ElementAux const* ptr_aux = nullptr;  // pointer to aux tensor
  ElementAux null_default = Element(0); // fallback scalar
  StrideMNL dAux = {};               // stride of aux tensor
};
```
This loads the auxiliary tensor pointed to by `ptr_aux`, using TMA to load into shared memory and the copy instruction `CopyOpS2R` to load into registers. If `EnableNullptr` is true and `ptr_aux == nullptr`, the operation instead falls back to a scalar broadcast of `null_default`. (Since this is a runtime check, the presence of this node in the tree still signals to the epilogue that a producer load is needed.)

**Methods overloaded:** `pld_callbacks.step()` (does TMA load), `cst_callbacks.previsit()` (does SMEM->RMEM copy), `cst_callbacks.visit()`. 

### Sm90ScalarBroadcast
```cpp
template<
  class ElementScalar,
  class StrideMNL = Stride<_0,_0,_0>, // must be <_0, _0, dL>
  int BroadcastCount = 1,
  template <class> class ReductionFn = multiplies // depends on dtype, as with Sm90Compute
>
struct Sm90ScalarBroadcast;
struct Arguments {
  ElementScalar scalars[BroadcastCount] = {};
  ElementScalar const* scalar_ptrs[BroadcastCount] = {};
  StrideMNL dScalar = {};
};
```
This broadcasts a scalar, i.e., gives each thread a fragment sized equally to its accumulator size and filled with the scalar. In the batched case (L > 1), a different scalar can be used for each index in the batch dimension.

We actually supply not just a single scalar, but an array of scalars of size `BroadcastCount`. If `BroadcastCount > 1`, this array is then reduced using `ReductionFn` before broadcasting. As with `Sm90AuxLoad`, we can supply either an array of scalars or an array of pointers to scalars. The second form is useful in the batched case, where we assume that each pointer points to the start of an array, and index into it according to the value of `l` (the batch dimension).

**Methods overloaded:** `.visit()`.

### Sm90RowBroadcast and Sm90ColBroadcast
```cpp
template<int Stages, // currently must be 0
  class CtaTileShapeMNK,
  class ElementAux,
  class StrideMNL = Stride<_0, _1, _0>, // need <_0, _1, dL>
  // for Sm90ColBroadcast: StrideMNL = Stride<_1, _0, _0>, need <_1, _0, dL>
  int Alignment = 16B,
  bool EnableNullptr = true>
struct Sm90RowBroadcast; // Sm90ColBroadcast is similar
struct Arguments {
  ElementAux const* ptr_row = nullptr;
  ElementAux null_default = Element(0);
  StrideMNL dRow = {};
};
```
These broadcast a row or column in GMEM to the fragment size, similar to Sm90AuxLoad. Note that (as of CUTLASS 3.5) these operations do not support a multistage pipelined load. Again, there is a fallback option to instead broadcast the scalar given in null_default.

**Methods overloaded:** `cst_callbacks.begin()` (tiled copy row from GMEM->SMEM, or column from GMEM->RMEM), `cst_callbacks.begin_loop()` (tiled copy row SMEM->RMEM), `.visit()`.

### Sm90AuxStore
```cpp
template<int Stages,
  class EpilogueTile,
  class ElementAux,
  FloatRoundStyle RoundStyle,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpR2S,
  int Alignment = 16B,
  bool EnableNullptr = true>
struct Sm90AuxStore;
struct Arguments {
  ElementAux* ptr_aux = nullptr;
  StrideMNL dAux = {};
};
```
This stores its input to GMEM at ptr_aux, also returning this input down the tree as output. The TiledCopy `CopyOpR2S` is used to store the input to SMEM, and TMA is used to store it to GMEM. If `EnableNullptr` is true and `ptr_aux == nullptr`, no SMEM or GMEM stores are performed.

**Methods overloaded:** `.visit()` (store input in register-backed aux fragment), `.postreduce()`  (tiled copy RMEM->SMEM), `.tma_store()` (TMA store SMEM->GMEM).

### Sm90ScalarReduction
```cpp
template <template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementOutput, // dtype to store result in
  class ElementCompute, // dtype to hold intermediate computations
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_0,_0>, // need <_0, _0, dL>
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90ScalarReduction;
struct Arguments {
  ElementOutput* ptr_scalar = nullptr;
  ElementCompute reduction_identity = ElementCompute(0);
  StrideMNL dScalar = {};
};
```
This reduces an input matrix to a scalar, using `RegReduceFn` to do thread-level reductions and `GmemReduceFn` to do CTA- and grid-level reductions. If `EnableNullptr` is true, then the operation does nothing if `ptr_scalar == nullptr`.

**Methods overloaded:** `cst_callbacks.visit()` (thread-level reduction), `cst_callbacks.end()` (grid-level reduction).

### Sm90RowReduction and Sm90ColReduction
```cpp
template<template <class> class RegReduceFn,
  template <class> class ShuffleReduceFn,
  template <class> class GmemReduceFn,
  int Stages, // must be 0
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_1,_0>, // must be <_0, _1, dL>
  // for Sm90ColReduction, StrideMNL = Stride<_1, _0, _0>, must be <_1, _0, dL>
  int Alignment = 16B,
  bool EnableNullptr = true,
  bool FinalReduction = true, 
  bool VisitCheckOOB = true>
struct Sm90RowReduction;
struct Arguments {
  void* ptr_row = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
  ElementCompute reduction_identity = 0;
  StrideMNL dRow = {};
};
```

Reduce the matrix along, respectively, the row or column dimension. This proceeds in several stages:

1. Reduce across the thread using `RegReduceFn`,
2. Reduce across the warp using warp shuffle and `ShuffleReduceFn`,
3. Reduce across the CTA using `sD` as workspace and `GmemReduceFn`,
4. Move results to GMEM workspace (thus, the presence of this node requires an additional GMEM workspace to be allocated),
5. Atomically increment a tile counter; if we reach the expected tile count, then this CTA was the last to complete.
6. If the CTA is the last to complete and `FinalReduction` is true, then use `GmemReduceFn` to produce an output.

If `FinalReduction` is false, the result remains in dtype `ElementCompute`, and only CTA-level reductions get performed -- thus, the output may not be a row. A comment says:
```cpp
// If this is false, ptr_row is assumed to point to a compact n-major (ceil_div(M,CTA_M), round_nearest(N,CTA_N), L)
// tensor of ElementCompute. It is the user's responsibility to reduce this to a (N, L) tensor of ElementOutput
```

**Methods overloaded:** `cst_callbacks.visit()` (thread-level reduction), `cst_callbacks.reduce()` (warp- and CTA-level reductions), `cst_callbacks.end()` (final GMEM reduction).
