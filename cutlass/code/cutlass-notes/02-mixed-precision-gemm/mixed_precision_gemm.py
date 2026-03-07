import os
import subprocess

import torch
from torch.utils.cpp_extension import load


current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include_path = os.path.join(current_dir, "../third-party/cutlass/include")
sources = [os.path.join(current_dir, filename) for filename in ["mixed_precision_gemm.cu"]]

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))

# Load CUDA extension module
lib = load(
    name="mixed_precision_gemm",
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


# ---------------- fp32 = bf16 * bf16 + fp32 ----------------

Ms = [16]
Ns = [8]
Ks = [8]
exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

torch.cuda.manual_seed_all(9527)

for exp in exps:
    M, N, K = exp
    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Case 1: MM
    kernel_output = lib.mixed_precision_gemm_fp32_bf16_bf16_fp32(a, b, None)
    if not ENABLE_PROF:
        # Note: For bf16 input, `torch.matmul` uses bf16 as the output precision,
        # which may lead to reduced numerical accuracy. To ensure higher precision,
        # we explicitly convert inputs `a` and `b` to fp32 before performing the multiplication.
        torch_output = torch.matmul(a.float(), b.T.float())
        compare_matrix(kernel_output, torch_output)

    # Case 2: MMA
    kernel_output = lib.mixed_precision_gemm_fp32_bf16_bf16_fp32(a, b, c.clone())
    if not ENABLE_PROF:
        torch_output = torch.addmm(c, a.float(), b.T.float())
        compare_matrix(kernel_output, torch_output)


# ---------------- bf16 = bf16 * bf16 + fp32 ----------------

Ms = [16]
Ns = [8]
Ks = [8]
exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

torch.cuda.manual_seed_all(9527)

for exp in exps:
    M, N, K = exp
    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Case 1: MM
    kernel_output = lib.mixed_precision_gemm_bf16_bf16_bf16_fp32(a, b, None)
    if not ENABLE_PROF:
        torch_output = torch.matmul(a, b.T)
        compare_matrix(kernel_output, torch_output)

    # # Case 2: MMA
    kernel_output = lib.mixed_precision_gemm_bf16_bf16_bf16_fp32(a, b, c.clone())
    if not ENABLE_PROF:
        torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
        compare_matrix(kernel_output, torch_output)


# ---------------- fp32 = e4m3 * e5m2 + fp32 ----------------


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split("\n"):
            if "release" in line:
                return line.split("release ")[1].split(",")[0]
    except Exception as e:
        return f"Error: {e}"


sm_version = torch.cuda.get_device_capability()
cuda_version = tuple(int(k) for k in get_cuda_version().split("."))

# Check if the hardware and software environment supports alternate floating point types (e4m3 and e5m2) for MMA operations.
# - SM version 8.9 or higher is required.
# - CUDA version 12.4 or higher (PTX version 8.4 or higher) is required.
if sm_version >= (8, 9) and cuda_version >= (12, 4):
    Ms = [16]
    Ns = [8]
    Ks = [32]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    torch.cuda.manual_seed_all(9527)

    for exp in exps:
        M, N, K = exp
        print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

        a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        a = a_fp32.to(torch.float8_e4m3fn)
        b_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
        b = b_fp32.to(torch.float8_e5m2)
        c = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # Case 1: MM
        kernel_output = lib.mixed_precision_gemm_fp32_e4m3_e5m2_fp32(a, b, None)
        if not ENABLE_PROF:
            torch_output = torch.matmul(a.float(), b.T.float())
            compare_matrix(kernel_output, torch_output)

        # Case 2: MMA
        kernel_output = lib.mixed_precision_gemm_fp32_e4m3_e5m2_fp32(a, b, c.clone())
        if not ENABLE_PROF:
            torch_output = torch.addmm(c, a.float(), b.T.float())
            compare_matrix(kernel_output, torch_output)


# ---------------- bf16 = e4m3 * e5m2 + fp32 ----------------

if sm_version >= (8, 9) and cuda_version >= (12, 4):
    Ms = [16]
    Ns = [8]
    Ks = [32]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    torch.cuda.manual_seed_all(9527)

    for exp in exps:
        M, N, K = exp
        print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

        a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        a = a_fp32.to(torch.float8_e4m3fn)
        b_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
        b = b_fp32.to(torch.float8_e5m2)
        c = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # Case 1: MM
        kernel_output = lib.mixed_precision_gemm_bf16_e4m3_e5m2_fp32(a, b, None)
        if not ENABLE_PROF:
            torch_output = torch.matmul(a.float(), b.T.float()).bfloat16()
            compare_matrix(kernel_output, torch_output)

        # Case 2: MMA
        kernel_output = lib.mixed_precision_gemm_bf16_e4m3_e5m2_fp32(a, b, c.clone())
        if not ENABLE_PROF:
            torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
            compare_matrix(kernel_output, torch_output)


print(f" Summary: {num_succeed} Succeed, {num_failed} Failed ".center(PRINT_LENGTH, "-"))
