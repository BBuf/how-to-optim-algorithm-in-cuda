# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any

import torch

from fairscale.nn.model_parallel.initialize import get_model_parallel_group

# from float8_tensor import Float8Tensor
from torchao.float8.float8_tensor import Float8Tensor

# additional differentiable distributed primitives for SP which are not in
# the Fairscale codebase


def _gather_along_first_dim(input_: torch.Tensor):
    # same as https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/mappings.py#L67,
    # but gather along first dim instead of last dim
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    first_dim = 0
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    # If the input is a float8 tensor, we need to do the transformation on the
    # inner tensor and then return a new wrapper.
    def _transform(t):
        # tensors must be contiguous for all_gather to work
        input_contig = t.contiguous()

        tensor_list = [torch.empty_like(input_contig) for _ in range(world_size)]
        tensor_list[rank] = input_contig
        torch.distributed.all_gather(tensor_list, input_contig, group=group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=first_dim).contiguous()
        return output

    if isinstance(input_, Float8Tensor):
        new_data = input_._data
        new_data = new_data.view(torch.int8)
        new_data = _transform(new_data)
        new_data = new_data.view(input_._data.dtype)
        output = Float8Tensor(new_data, input_._scale, input_._orig_dtype)
    else:
        output = _transform(input_)

    return output


def _reduce_scatter(ctx: Any, input_: torch.Tensor):
    group = get_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)

    assert input_.shape[0] % world_size == 0
    output_shape = (input_.shape[0] // world_size, *input_.shape[1:])
    output = torch.empty(*output_shape, device=input_.device, dtype=input_.dtype)

    torch.distributed.reduce_scatter_tensor(output, input_, group=group)
    return output


def _split_along_first_dim(input_: torch.Tensor):
    # this is needed for testing

    # like fairscale.nn.model_parallel.mappings._split, but
    # along the first dim instead of last dim

    group = get_model_parallel_group()
    local_rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)

    assert input_.shape[0] % world_size == 0
    input_list = torch.split(input_, input_.shape[0] // world_size)
    return input_list[local_rank]


class _AllGatherFloat8FwReduceScatterBw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter(ctx, grad_output)


class _ReduceScatterFwAllGatherFloat8Bw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _AllGatherFwSplitBw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_first_dim(grad_output)
