> 来源： https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487

# [Distributed w/ TorchTitan] Introducing Async Tensor Parallelism in PyTorch

## TL;DR

- We implemented experimental async tensor parallelism support in PyTorch.
- We integrated it in TorchTitan and observed:
    - Up to ~29% forward pass speedup and ~8% E2E speedup in Llama3 7B.
    - Up to ~20% forward pass speedup and ~8% E2E speedup in Llama3 70B.
- We briefly discuss the performance challenges and the solutions we devised.

