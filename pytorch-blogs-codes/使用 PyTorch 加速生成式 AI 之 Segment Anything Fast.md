> 来源：https://pytorch.org/blog/accelerating-generative-ai/

This post is the first part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance features alongside practical examples of how these features can be combined to see how far we can push PyTorch native performance.

As announced during the PyTorch Developer Conference 2023, the PyTorch team rewrote Meta’s Segment Anything (“SAM”) Model(https://github.com/facebookresearch/segment-anything) resulting in 8x faster code than the original implementation, with no loss of accuracy, all using native PyTorch optimizations. We leverage a breadth of new PyTorch features:


- Torch.compile: A compiler for PyTorch models
- GPU quantization(https://github.com/pytorch/ao/tree/main#torchao): Accelerate models with reduced precision operations
- Scaled Dot Product Attention (SDPA): Memory efficient attention implementations
- Semi-Structured (2:4) Sparsity(https://pytorch.org/tutorials/prototype/semi_structured_sparse.html): A GPU optimized sparse memory format
- Nested Tensor(https://pytorch.org/tutorials/prototype/nestedtensor.html): Batch together non-uniformly sized data into a single Tensor, such as images of different sizes.
- **Custom operators with Triton**: Write GPU operations using Triton Python DSL and easily integrate it into PyTorch’s various components with custom operator registration.

We encourage readers to copy-paste code from our implementation of SAM on Github(https://github.com/pytorch-labs/segment-anything-fast) and ask us questions on Github.


![A quick glimpse of increasing throughput and decreasing memory overhead with our newly released, PyTorch native, features. Benchmarks run on p4d.24xlarge instance (8x A100s).](https://files.mdnice.com/user/59/41d3ccfe-07eb-4b49-8d6d-18cdc8dd4699.png)

# SegmentAnything Model

SAM is a zero-shot vision model for generating promptable image masks.

![](https://files.mdnice.com/user/59/b9b480e2-961c-4a40-9452-44b67d5b9a6d.jpg)

The SAM architecture [described in its paper(https://arxiv.org/abs/2304.02643)] includes multiple prompt and image encoders based on the Transformer architecture. Of this, we measured performance across the smallest and largest vision transformer backbones: ViT-B and ViT-H. And for simplicity, we only show traces for the ViT-B model.

# Optimizations

Below we tell the story of optimizing SAM: profiling, identifying bottlenecks, and building new features into PyTorch that solve these problems. Throughout, we showcase our new PyTorch features: torch.compile, SDPA, Triton kernels, Nested Tensor and semi-structured sparsity. The following sections are progressively built upon each other, ending with our SAM-fast, now available on Github(https://github.com/pytorch-labs/segment-anything-fast). We motivate each feature using real kernel and memory traces, using fully PyTorch native tooling, and visualize these traces with Perfetto UI(https://perfetto.dev/#viewer).


## Baseline

Our SAM baseline is Facebook Research’s unmodified model, using float32 dtype and a batch size of 1. After some initial warmup, we can look at a kernel trace using the PyTorch Profiler:

![](https://files.mdnice.com/user/59/f83a0658-0355-42fc-bd06-e0ba48658919.png)

We notice two areas ripe for optimization.

The first is long calls to aten::index, the underlying call resulting from a Tensor index operation (e.g., []). While the actual GPU time spent on aten::index is relatively low. aten::index is launching two kernels, and a blocking cudaStreamSynchronize is happening in between. This means the CPU is waiting for the GPU to finish processing until it launches the second kernel. To optimize SAM, we should aim to remove blocking GPU syncs causing idle time.

The second is significant time spent on GPU in matrix multiplication (dark green on stream 7 7 above). This is common in Transformers. We can significantly speed up SAM if we can reduce the amount of GPU time spent on matrix multiplication.

We can measure the throughput (img/s) and memory overhead (GiB) from out of the box SAM to establish a baseline:

![](https://files.mdnice.com/user/59/965b7f81-e87a-4e0a-951b-334d36a75091.png)

## Bfloat16 Half precision (+GPU syncs and batching)

To address the first issue of less time spent in matrix multiplication, we can turn to bfloat16. Bfloat16 is a commonly used half-precision type. Through less precision per parameter and activations, we can save significant time and memory in computation. With reducing precision of parameters, it’s critical to validate end to end model accuracy.

![](https://files.mdnice.com/user/59/496022b4-f412-46fb-bee5-51af1eab027c.png)

Shown here is an example of replacing padding dtypes with half precision, bfloat16. Code is here(https://github.com/pytorch-labs/segment-anything-fast/blame/main/segment_anything_fast/modeling/prompt_encoder.py#L86).

Next to simply setting `model.to(torch.bfloat16)` we have to change a few small places that assume the default dtype.

Now, in order to remove GPU syncs we need to audit operations that cause them. We can find these pieces of code by searching the GPU traces for calls to `cudaStreamSynchronize`. In fact, we found two locations that we were able to rewrite to be sync-free.

![](https://files.mdnice.com/user/59/7a9acc28-8c65-46ec-8b59-0a953361791c.jpg)

Specifically, we see that within SAM’s image encoder, there are variables acting as coordinate scalers, q_coords and k_coords. These are both allocated and processed on the CPU. However, once these variables are used to index in rel_pos_resized, the index operation automatically moves these variables to the GPU. This copy over causes the GPU sync we’ve observed above. We notice a second call to index in SAM’s prompt encoder: We can use torch.where to rewrite this as shown above.

### Kernel trace

After applying these changes, we begin to see significant time between individual kernel calls. This is typically observed with small batch sizes (1 here) due to the GPU overhead of launching kernels. To get a closer look at practical areas for optimization, we can start to profile SAM inference with batch size 8:

![](https://files.mdnice.com/user/59/47e36556-1212-4cf4-9d54-77f8ae565ce0.png)

Looking at the time spent per-kernel, we obverse most of SAM’s GPU time spent on elementwise kernels and softmax operation. With this we now see that matrix multiplications have become a much smaller relative overhead.

![](https://files.mdnice.com/user/59/003c375a-fce8-4f50-bb48-fed65d4d5c82.png)

Taken the GPU sync and bfloat16 optimizations together, we have now pushed SAM performance by up to 3x

![](https://files.mdnice.com/user/59/1e1353b9-4487-42c0-96d2-68dc0cca27e0.png)

## Torch.compile (+graph breaks and CUDA graphs)

When observing a large number of small operations, such as the elementwise kernels profiled above, turning to a compiler to fuse operations can have strong benefits. PyTorch’s recently released torch.compile does a great job optimizing by:

- Fusing together sequences of operations such as nn.LayerNorm or nn.GELU into a single GPU kernel that is called and
- Epilogues: fusing operations that immediately follow matrix multiplication kernels to reduce the number of GPU kernel calls.

Through these optimizations, we reduce the number of GPU global memory roundtrips, thus speeding up inference. We can now try torch.compile on SAM’s image encoder(https://github.com/pytorch-labs/segment-anything-fast/blob/3bd74614fe7285de4de3d763d8ec2e951c4c589c/experiments/eval_combo.py#L196-L201). To maximize performance we use a few advanced compile techniques such as:

- using torch.compile’s max-autotune mode enables CUDA graphs and shape-specific kernels with custom epilogues
- By setting TORCH_LOGS=”graph_breaks,recompiles” we can manually verify that we are not running into graph breaks(https://pytorch.org/docs/main/torch.compiler_faq.html#graph-breaks) or recompiles.
- Padding the batch of images input to the encoder with zeros ensures compile accepts static shapes thus being able to always use shape-specific optimized kernels with custom epilogues without recompilations.

```python
predictor.model.image_encoder = \
    torch.compile(predictor.model.image_encoder, mode=use_compile)
```


### Kernel trace

![](https://files.mdnice.com/user/59/b2423571-bd3b-4b50-b0f9-e03d77a64ef7.jpg)

torch.compile is working beautifully. We launch a single CUDA graph, which makes up a significant portion of GPU time within the timed region. Let’s run our profile again and look at the percentage of GPU time spent in specific kernels:

![](https://files.mdnice.com/user/59/05f590ab-6c31-4e87-89a9-45fbd2799b9f.jpg)

We now see softmax makes up a significant portion of the time followed by various GEMM variants. In summary we observe the following measurements for batch size 8 and above changes.

![](https://files.mdnice.com/user/59/4bbc6e30-613a-4c76-a34d-09cdabb214c7.png)

## SDPA: scaled_dot_product_attention

