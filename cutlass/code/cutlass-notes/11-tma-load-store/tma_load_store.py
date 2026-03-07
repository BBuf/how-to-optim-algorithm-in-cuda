import os

import torch
from torch.utils.cpp_extension import load


current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include_path = os.path.join(current_dir, "../third-party/cutlass/include")
sources = [os.path.join(current_dir, filename) for filename in ["tma_load_store.cu"]]

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))

# Load CUDA extension module
lib = load(
    name="tma_load_store",
    sources=sources,
    extra_cuda_cflags=[
        "-O3",
        f"-I{cutlass_include_path}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--resource-usage",  # printing out number of registers
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--generate-line-info",  # show PTX and SASS in ncu
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_ENABLE_GDC_FOR_SM100",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-Xfatbin",  # compress all binary sections
        "-compress-all",
        # for debug purpose
        # "-G", # device debug
        # "-g", # host debug
        # "-Xcompiler",
        # "-rdynamic",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True,  # show compile logs
)


ENABLE_PROF = os.environ.get("ENABLE_PROF", False)
ENABLE_MMA = os.environ.get("ENABLE_MMA", False)
PRINT_LENGTH = 100


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    diff = target - ref
    norm_diff = torch.norm(diff, p=2)
    norm_diff_ref = torch.norm(ref, p=2)

    return (norm_diff / (norm_diff_ref + eps)).item()


num_succeed = 0
num_failed = 0


def compare_matrix(kernel_output: torch.Tensor, torch_output: torch.Tensor):
    kernel_output = kernel_output.float()
    torch_output = torch_output.float()

    max_diff = torch.max(torch.abs(torch_output - kernel_output))
    mean_diff = torch.mean(torch.abs(torch_output - kernel_output))
    re = relative_error(kernel_output, torch_output)
    is_correct = re < 0.001

    global num_succeed, num_failed

    if not is_correct:
        num_failed += 1

        print(f" Kernel Output: {tuple(kernel_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(kernel_output[:8, :8])

        print(f" Torch Output: {tuple(torch_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(torch_output[:8, :8])
    else:
        num_succeed += 1

    print(
        f" Result: {'Success' if is_correct else 'Failed'}, Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, RE = {(re * 100):.2f}% ".center(
            PRINT_LENGTH, "-"
        )
    )


# Note: N and K must be multiples of 8 to remain compatible with the 128-bit vectorized copy op.
Ms = [16, 64, 128, 192, 256, 1024, 4096, 8192]
Ns = [16, 64, 128, 192, 256, 1024, 4096, 8192]
Ks = [16, 64, 128, 192, 256, 1024, 4096, 8192]
exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]


# ---------------- fp16 = fp16 * fp16 + fp32 ----------------

torch.cuda.manual_seed_all(9527)

for exp in exps:
    M, N, K = exp
    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float)

    # Case 1: MM
    _ = lib.tma_load_store_fp16_fp16_fp16_fp16(a, b, None)
    _ = lib.tma_load_store_fp16_fp16_fp16_fp16(a, b, c.half())
    kernel_output = lib.tma_load_store_fp16_fp16_fp16_fp32(a, b, None)
    if not ENABLE_PROF:
        # For fp16 input, `torch.matmul` uses fp32 as the accumulator precision
        torch_output = torch.matmul(a, b.T)
        compare_matrix(kernel_output, torch_output)

    # Case 2: MMA
    if ENABLE_MMA:
        kernel_output = lib.tma_load_store_fp16_fp16_fp16_fp32(a, b, c.clone())
        if not ENABLE_PROF:
            torch_output = torch.addmm(c, a.float(), b.T.float()).half()
            compare_matrix(kernel_output, torch_output)


# ---------------- bf16 = bf16 * bf16 + fp32 ----------------

torch.cuda.manual_seed_all(9527)

for exp in exps:
    M, N, K = exp
    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Case 1: MM
    kernel_output = lib.tma_load_store_bf16_bf16_bf16_fp32(a, b, None)
    if not ENABLE_PROF:
        torch_output = torch.matmul(a.float(), b.T.float()).bfloat16()
        compare_matrix(kernel_output, torch_output)

    # Case 2: MMA
    if ENABLE_MMA:
        kernel_output = lib.tma_load_store_bf16_bf16_bf16_fp32(a, b, c.clone())
        if not ENABLE_PROF:
            torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
            compare_matrix(kernel_output, torch_output)


print(f" Summary: {num_succeed} Succeed, {num_failed} Failed ".center(PRINT_LENGTH, "-"))
