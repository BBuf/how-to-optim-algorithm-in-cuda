import os

import click
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

from triton_barrier import get_flat_tid
from utils import benchmark_with_event, log_triton_kernel

def all_gather_with_progress(
    output: torch.Tensor,  # 输出张量,用于存储all-gather的结果
    inp: torch.Tensor,     # 输入张量,每个rank上的局部数据
    progress: torch.Tensor, # 进度张量,用于跟踪每个rank的完成状态
    splits_per_rank: int,  # 每个rank上的分片数量
):
    # 确保输入张量是连续的
    assert inp.is_contiguous()

    # 创建对称内存句柄,用于在不同rank之间共享内存
    symm_mem_hdl = symm_mem.rendezvous(inp, group=dist.group.WORLD)
    assert symm_mem_hdl is not None

    # 获取当前rank和总的rank数量
    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    # 确保输入可以被splits_per_rank整除
    assert inp.numel() % splits_per_rank == 0
    # 确保progress张量大小正确
    assert progress.numel() == world_size * splits_per_rank

    # 计算输出张量的形状
    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, (list(output.shape), output_shape)

    # 将输出张量分成world_size * splits_per_rank个块
    chunks = output.chunk(world_size * splits_per_rank)

    # 对每个rank进行循环
    for step in range(0, world_size):
        # 计算源rank,使用循环方式遍历所有rank
        src_rank = (rank + step + 1) % world_size
        # 对每个分片进行循环
        for split_id in range(splits_per_rank):
            # 从源rank获取对应的buffer
            src_buf = symm_mem_hdl.get_buffer(
                src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
            )
            # 将源buffer复制到对应的输出chunk
            chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
            # 写入进度值,表示该分片已完成
            # cuStreamWriteValue32 在写入前会发出系统级fence
            symm_mem_hdl.stream_write_value32(
                progress,
                offset=src_rank * splits_per_rank + split_id,
                val=1,
            )
    # 等待所有rank完成
    symm_mem_hdl.barrier()


def _matmul_launch_metadata(grid, kernel, args):
    # 初始化返回字典
    ret = {}
    
    # 从参数中获取矩阵维度
    M, N, K = args["M"], args["N"], args["K"]
    
    # 设置kernel名称,包含矩阵维度信息
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    
    # 计算浮点运算次数(8位精度)
    # 矩阵乘法每个元素需要2次运算(乘加),总次数为M*N*K*2
    ret["flops8"] = 2.0 * M * N * K
    
    # 计算每个元素占用的字节数
    if "c_ptr" in args:
        # 如果有输出指针,使用其元素大小
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        # 否则根据是否使用FP8输出决定大小(FP8=1字节,否则2字节)
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
        
    # 计算总内存访问字节数(输入矩阵A和B的大小之和)
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret

@triton.jit
def wait_signal(addr, flat_tid):
    # 只有第一个线程(flat_tid=0)执行等待操作
    if flat_tid == 0:
        # 使用内联汇编实现等待信号
        tl.inline_asm_elementwise(
            """
            {
                # 定义一个谓词寄存器
                .reg .pred  %p<1>;

                # 等待循环标签
                wait_block:
                    # 从全局内存加载32位无符号整数
                    ld.global.relaxed.gpu.u32 $0, [$1];
                    # 比较加载的值是否等于1
                    setp.eq.u32 %p0, $0, 1;
                    # 如果不等于1,跳回等待循环
                    @!%p0 bra wait_block;
            }
            """,
            "=r, l",
            [addr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    # 使用内联汇编实现线程块同步
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    a_shard_desc_ptr,  # A矩阵分片的TMA描述符指针
    a_desc_ptr,        # A矩阵的TMA描述符指针
    b_desc_ptr,        # B矩阵的TMA描述符指针
    c_desc_ptr,        # C矩阵的TMA描述符指针
    progress_ptr,      # 进度指针,用于同步
    M,                 # 矩阵A的行数
    N,                 # 矩阵B的列数
    K,                 # 矩阵A的列数/矩阵B的行数
    BLOCK_SIZE_M: tl.constexpr,  # 块大小M维度
    BLOCK_SIZE_N: tl.constexpr,  # 块大小N维度
    BLOCK_SIZE_K: tl.constexpr,  # 块大小K维度
    GROUP_SIZE_M: tl.constexpr,  # M维度的组大小
    COMM_BLOCK_SIZE_M: tl.constexpr,  # 通信块M维度大小
    RANK: tl.constexpr,          # 当前进程的rank
    WORLD_SIZE: tl.constexpr,    # 总进程数
    FP8_OUTPUT: tl.constexpr,    # 是否使用FP8输出
    NUM_SMS: tl.constexpr,       # SM数量
):
    """
    基于SM90 TMA persistent Triton教程修改的矩阵乘法kernel
    """
    # 获取线程ID
    flat_tid = get_flat_tid()

    # 根据是否使用FP8输出选择数据类型
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.bfloat16
    # 获取程序ID
    start_pid = tl.program_id(axis=0)
    # 计算M、N维度的程序ID数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 计算K维度的tile数量
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    # 计算总tile数量
    num_tiles = num_pid_m * num_pid_n

    # 计算每个SM处理的tile数量
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    # 初始化tile ID和其他变量
    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am_src = 0
    offs_bn = 0
    a_ptr = a_desc_ptr

    # 计算每组的程序ID数量
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 主循环,处理所有tile
    for _ in range(0, k_tiles * tiles_per_SM):
        # 更新K维度的tile索引
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            # 更新tile ID和相关索引
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # 计算通信相关的块大小和数量
            NUM_COMM_BLOCKS = M // COMM_BLOCK_SIZE_M
            NUM_COMM_BLOCKS_PER_RANK = NUM_COMM_BLOCKS // WORLD_SIZE
            NUM_PID_M_PER_COMM_BLOCK = COMM_BLOCK_SIZE_M // BLOCK_SIZE_M

            # 上面的pid_m是没有做分片的时候的pid_m，这里要考虑到分片的情况
            pid_m = (pid_m + NUM_PID_M_PER_COMM_BLOCK * RANK) % num_pid_m

            # 确定数据来源(本地或远程)
            comm_block_id = pid_m // NUM_PID_M_PER_COMM_BLOCK
            if comm_block_id // NUM_COMM_BLOCKS_PER_RANK == RANK:
                # 从本地分片读取
                offs_am_src = (pid_m * BLOCK_SIZE_M) % COMM_BLOCK_SIZE_M
                a_ptr = a_shard_desc_ptr
            else:
                # 等待并从远程分片读取
                wait_signal((progress_ptr + comm_block_id).to(tl.uint64), flat_tid)
                offs_am_src = pid_m * BLOCK_SIZE_M
                a_ptr = a_desc_ptr

        # 计算偏移量
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = ki * BLOCK_SIZE_K

        # 从全局内存加载数据块
        a = tl._experimental_descriptor_load(
            a_ptr, [offs_am_src, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        # 执行矩阵乘法
        accumulator = tl.dot(a, b.T, accumulator)

        # K维度处理完成后,存储结果
        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(
                c_desc_ptr, c, [pid_m * BLOCK_SIZE_M, offs_bn]
            )
            # 重置累加器
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


# TMA描述符缓存
_tma_desc_cache = {}


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    """创建2D TMA描述符并缓存
    
    Args:
        ptr: 数据指针
        dim1: 第一维度大小
        dim0: 第二维度大小 
        block_dim1: 块的第一维度大小
        block_dim0: 块的第二维度大小
        element_size: 元素大小(字节)
        
    Returns:
        TMA描述符对象
    """
    global _tma_desc_cache
    key = (ptr, dim1, dim0, block_dim1, block_dim0, element_size)
    if key in _tma_desc_cache:
        return _tma_desc_cache[key]
    desc = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        ptr,
        dim1,
        dim0,
        block_dim1,
        block_dim0,
        element_size,
    )
    _tma_desc_cache[key] = desc
    return desc


def all_gather_matmul_tma_persistent(
    a_shard, b, a_out, c_out, configs, mm_only: bool = False
):
    """使用TMA和persistent kernel实现all-gather矩阵乘法
    
    Args:
        a_shard: 分片的A矩阵
        b: B矩阵
        a_out: all-gather后的A矩阵输出
        c_out: 矩阵乘法结果输出
        configs: kernel配置参数
        mm_only: 是否只执行矩阵乘法(不执行all-gather)
    
    Returns:
        矩阵乘法结果
    """
    # 获取rank和world_size
    if mm_only:
        rank = 0
        world_size = int(os.environ.get("WORLD_SIZE", "8"))
    else:
        symm_mem_hdl = symm_mem.rendezvous(a_shard, group=dist.group.WORLD)
        assert symm_mem_hdl is not None, "a_shard must be allocated via SymmetricMemory"
        rank = symm_mem_hdl.rank
        world_size = symm_mem_hdl.world_size

    # 获取输入矩阵的基本信息
    dtype = a_shard.dtype
    M = a_shard.shape[0] * world_size
    N = b.shape[0]
    K = a_shard.shape[1]

    # 检查矩阵维度是否匹配
    assert b.shape[1] == K
    assert a_out.shape[0] == M
    assert a_out.shape[1] == K
    assert c_out.shape[0] == M
    assert c_out.shape[1] == N

    # 计算通信块大小
    SPLITS_PER_RANK = 1
    COMM_BLOCK_SIZE_M = M // world_size // SPLITS_PER_RANK
    assert COMM_BLOCK_SIZE_M % (configs["BLOCK_SIZE_M"] * configs["GROUP_SIZE_M"]) == 0

    # 设置后端流和进度数组
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    if mm_only:
        progress = torch.ones(world_size, dtype=torch.uint32, device="cuda")
    else:
        progress = torch.zeros(world_size, dtype=torch.uint32, device="cuda")
        symm_mem_hdl.barrier(0)
        backend_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(backend_stream):
            all_gather_with_progress(a_out, a_shard, progress, SPLITS_PER_RANK)

    # 创建TMA描述符
    desc_a_shard = create_2d_tma_descriptor(
        a_shard.data_ptr(),
        a_shard.shape[0],
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a_shard.element_size(),
    )
    desc_a = create_2d_tma_descriptor(
        a_out.data_ptr(),
        M,
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a_out.element_size(),
    )
    desc_b = create_2d_tma_descriptor(
        b.data_ptr(),
        N,
        K,
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        b.element_size(),
    )
    desc_c = create_2d_tma_descriptor(
        c_out.data_ptr(),
        M,
        N,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        c_out.element_size(),
    )
    # 获取GPU的SM数量
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # 定义kernel网格大小
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    
    # 启动kernel
    kernel = matmul_kernel_tma_persistent[grid](
        desc_a_shard,
        desc_a,
        desc_b,
        desc_c,
        progress,
        M,
        N,
        K,
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        COMM_BLOCK_SIZE_M=COMM_BLOCK_SIZE_M,
        RANK=rank,
        WORLD_SIZE=world_size,
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        NUM_SMS=NUM_SMS,
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )
    log_triton_kernel(kernel)
    torch.cuda.current_stream().wait_stream(backend_stream)
    return c_out


def all_gather_matmul(a_shard, b):
    from torch.distributed._functional_collectives import all_gather_tensor

    a = all_gather_tensor(a_shard, 0, "0")
    return torch.matmul(a, b)


@click.command()
@click.option("--M", default=4096)
@click.option("--N", default=6656)
@click.option("--K", default=16384)
@click.option("--BLOCK_SIZE_M", default=128)
@click.option("--BLOCK_SIZE_N", default=256)
@click.option("--BLOCK_SIZE_K", default=64)
@click.option("--GROUP_SIZE_M", default=4)
@click.option("--num_stages", default=3)
@click.option("--num_warps", default=8)
def main(
    m: int,
    n: int,
    k: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_stages: int,
    num_warps: int,
):
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 triton_all_gather_matmul.py
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)
    dist.init_process_group("nccl")

    a_shard = symm_mem.empty(
        m // world_size, k, dtype=torch.bfloat16, device=device
    ).normal_()
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).T.contiguous()
    c = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    # Autotuner does not work with TMA. Use manual config.
    configs = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_stages": num_stages,
        "num_warps": num_warps,
    }

    c0 = all_gather_matmul(a_shard, b.T)
    c1 = all_gather_matmul_tma_persistent(a_shard, b, a, c, configs)
    assert torch.allclose(c0, c1)

    def rank_0_print(msg):
        if rank == 0:
            print(msg)

    lat_cublas_mm = benchmark_with_event(
        lambda: torch.matmul(a, b.T, out=c), flush_l2=True
    )
    rank_0_print(f"cublas mm only:\t{round(lat_cublas_mm)} us")

    lat_triton_mm = benchmark_with_event(
        lambda: all_gather_matmul_tma_persistent(
            a_shard, b, a, c, configs, mm_only=True
        ),
        flush_l2=True,
    )
    rank_0_print(f"triton mm only:\t{round(lat_triton_mm)} us")

    lat_cublas_nccl = benchmark_with_event(
        lambda: all_gather_matmul(a_shard, b.T), flush_l2=True
    )
    rank_0_print(f"cublas + nccl:\t{round(lat_cublas_nccl)} us")

    lat_triton_fused = benchmark_with_event(
        lambda: all_gather_matmul_tma_persistent(a_shard, b, a, c, configs),
        flush_l2=True,
    )
    rank_0_print(f"triton fused:\t{round(lat_triton_fused)} us")
    rank_0_print(f"speedup:\t{lat_cublas_nccl / lat_triton_fused:.02f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
