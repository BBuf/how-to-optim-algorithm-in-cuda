# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

BYTES_PER_EL_FLOAT8 = 1
BYTES_PER_EL_BF16 = 2

# https://www.nvidia.com/en-us/data-center/h100/, divide by 2 because no sparsity
H100_BF16_PEAK_TOPS = 989e12
H100_FP8_PEAK_TOPS = 1979e12

# 2.4 TB per second, custom to Meta's H100 variant
H100_PEAK_MEM_BW_BYTES_SEC = 2.4e12

# based on quick experimental observation with sample large inputs
H100_PCT_ACHIEVABLE_GEMM_TOPS = 0.6

# based on previous experience looking at pointwise triton kernels with large inputs,
# which would hit about 2.2k GBPS on Meta's H100 variant
H100_PCT_ACHIEVABLE_MEM_BW = 0.92

# Source: run a triton kernel with a single element read/write on an H100 and 
# measure GPU time from the trace
TRITON_KERNEL_1_ELEMENT_TIME_SEC = 0.002 * 0.001


def get_tensor_memory_traffic_bytes(
    dim0, 
    dim1,
    scaling_type: str,
    fuse_with_prev=False,
    model_torch_compile_limitations=False,
):
    # assumes input bf16, output f8
    numel = dim0 * dim1

    if scaling_type == "dynamic":
        # x_bf16 = ...
        # kernel 1:               x_bf16 -> max_abs_stage_1 -> tmp
        # kernel 2 (not modeled): tmp -> max_abs_stage_2 -> max_abs
        # kernel 3:               x_bf16, max_abs -> to_float8 -> x_fp8

        if fuse_with_prev:
            kernel_1_rw = 0
        else:
            # kernel 1: read numel, write 0 (assume size(tmp) ~ 0)
            kernel_1_rw = BYTES_PER_EL_BF16 * numel

        # kernel 3: read in bf16, write twice in float8 (row-major and col-major)
        kernel_3_rw = BYTES_PER_EL_BF16 * numel + 2 * BYTES_PER_EL_FLOAT8 * numel

        if model_torch_compile_limitations:
            # today, the kernel to do cast_to_fp8_row_major_and_col_major(input_bf16, ...)
            # has an extra memory read of the input in fp8
            # context: https://github.com/pytorch/pytorch/issues/130015
            tc_adjustment = numel * BYTES_PER_EL_FLOAT8
        else:
            tc_adjustment = 0

        return kernel_1_rw + kernel_3_rw + tc_adjustment

    else:
        assert scaling_type == "delayed", "unsupported"
        # x_bf16 = ...
        # kernel 1:               x_bf16 -> max_abs_stage_1_and_to_float8 -> x_float8, tmp
        # kernel 2 (not modeled): tmp -> max_abs_stage_2 -> max_abs
        # kernel 3 (not modeled): scale -> reciprocal -> inv_scale

        if fuse_with_prev:
            kernel_1_r = 0
        else:
            kernel_1_r = numel * BYTES_PER_EL_BF16
        # write twice: once in row major, once in col-major
        kernel_1_w = numel * BYTES_PER_EL_FLOAT8 * 2

        if model_torch_compile_limitations:
            # today, the kernel to do cast_to_fp8_row_major_and_col_major(input_bf16, ...)
            # has an extra memory read of the input in fp8
            # context: https://github.com/pytorch/pytorch/issues/130015
            tc_adjustment = numel * BYTES_PER_EL_FLOAT8

            # https://github.com/pytorch/pytorch/issues/128063
            # instead of 
            #   kernel 1: x_bf16 -> max(abs(x)), x_fp8
            #   kernel 2: not modeled
            #   kernel 3: not modeled
            # we get
            #   kernel 1: x_bf16 -> max(abs(x))
            #     reads: same as before
            #     writes: 0
            #   ...
            #   kernel 4: x_bf16, scale -> x_fp8
            #     reads: numel * BYTES_PER_EL_BF16
            #     writes: 2 * numel * BYTES_PER_EL_FLOAT8
            # Note that assuming worst case, this issue brings the memory 
            # traffic for delayed scaling to be equal to that of dynamic scaling.
            tc_adjustment += (
                # subtract writes from kernel 1
                -1 * 2 * numel * BYTES_PER_EL_FLOAT8
                # add reads for kernel 4
                + numel * BYTES_PER_EL_BF16
                # add writes for kernel 4
                + 2 * numel * BYTES_PER_EL_FLOAT8
            )
        else:
            tc_adjustment = 0

        return kernel_1_r + kernel_1_w + tc_adjustment


def get_gemm_time_sympy(M, K, N, dtype):
    gemm_ops = 2 * M * K * N + 2 * M * N * K + 2 * K * M * N
    if dtype is torch.bfloat16:
        peak_tops = H100_BF16_PEAK_TOPS
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak_tops = H100_FP8_PEAK_TOPS
    gemm_time_s = gemm_ops / peak_tops / H100_PCT_ACHIEVABLE_GEMM_TOPS
    return gemm_time_s


def get_float8_mem_sympy(
    M, 
    K, 
    N,
    model_torch_compile_limitations: bool = False,
    scaling_type_input: str = "dynamic",
    scaling_type_weight: str = "dynamic",
    scaling_type_grad_output: str = "dynamic",
):

    assert scaling_type_input in ("dynamic", "delayed"), "unsupported"
    assert scaling_type_weight in ("dynamic", "delayed"), "unsupported"
    assert scaling_type_grad_output in ("dynamic", "delayed"), "unsupported"

    # there are three gemms in the fwd/bwd of a linear:
    #
    # input @ weight_t = output
    # MxK @ KxN => MxN
    #
    # grad_output @ weight = grad_input
    # MxN @ NxK => MxK
    #
    # input_t @ grad_output = grad_weight
    # KxM @ MxN => KxN

    #
    # forward - output
    #
    fwd_fp8_input_mem = get_tensor_memory_traffic_bytes(
        M, K, scaling_type_input, fuse_with_prev=True, 
        model_torch_compile_limitations=model_torch_compile_limitations)
    fwd_fp8_weight_mem = get_tensor_memory_traffic_bytes(
        K, N, scaling_type_weight, fuse_with_prev=False,
        model_torch_compile_limitations=model_torch_compile_limitations)
    fwd_fp8_total_mem = fwd_fp8_input_mem + fwd_fp8_weight_mem

    #
    # backward - grad_input
    #
    gi_fp8_grad_output_mem = get_tensor_memory_traffic_bytes(
        M, N, scaling_type_grad_output, fuse_with_prev=True,
        model_torch_compile_limitations=model_torch_compile_limitations)
    # already casted, assuming that we save weight from fw to bw
    # TODO: model this if FSDP float8 all-gather is on
    # TODO: model this if we don't save weight from fw to bw, and recompute instead
    gi_fp8_weight_mem = 0  

    #
    # backward - grad_weight
    #
    # TODO: model this if we don't save fp8 input from fw to bw
    gw_fp8_input_t_mem = 0  # already casted
    # this should be always 0
    gw_fp8_grad_output_mem = 0  # already casted

    bwd_fp8_total_mem = \
        gi_fp8_grad_output_mem + gi_fp8_weight_mem + \
        gw_fp8_input_t_mem + gw_fp8_grad_output_mem
    fp8_total_mem = fwd_fp8_total_mem + bwd_fp8_total_mem
    fp8_mem_time_s = (
        fp8_total_mem / H100_PEAK_MEM_BW_BYTES_SEC / H100_PCT_ACHIEVABLE_MEM_BW
    )

    # Adjust final estimate for small kernel launches
    # note that we do this adjustment here because we are assuming a minimal
    # kernel overhead in the units of seconds, and the per-gemm-input memory
    # estimations are in the units of bytes.
    num_extra_kernels = 0
    if scaling_type_input == "dynamic":
        # second stage of max-abs reduction
        num_extra_kernels += 1
    elif scaling_type_input == "delayed":
        # second stage of max-abs reduction
        num_extra_kernels += 1
        # reciprocal of scale
        num_extra_kernels += 1
    if scaling_type_weight == "dynamic":
        # second stage of max-abs reduction
        num_extra_kernels += 1
    elif scaling_type_weight == "delayed":
        # second stage of max-abs reduction
        num_extra_kernels += 1
        # reciprocal of scale
        num_extra_kernels += 1
    if scaling_type_grad_output == "dynamic":
        # second stage of max-abs reduction
        num_extra_kernels += 1
    elif scaling_type_grad_output == "delayed":
        # second stage of max-abs reduction
        num_extra_kernels += 1
        # reciprocal of scale
        num_extra_kernels += 1

    extra_kernel_overhead_s = num_extra_kernels * TRITON_KERNEL_1_ELEMENT_TIME_SEC

    return fp8_mem_time_s + extra_kernel_overhead_s
