"""
对单个静态批次在无服务器情况下运行延迟的基准测试。

此脚本不启动服务器，而是使用底层API。
它接受服务器参数（与launch_server.py相同）和基准测试参数（例如，批次大小，输入长度）。

# 使用方法（延迟测试）
## 使用虚拟权重：
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## 遍历多个数据点并将结果存储（追加）到jsonl文件中：
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run

# 使用方法（正确性测试）：
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## 参考输出（上述正确性测试的输出，可能因GPU而异）：
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server import _set_envs_and_config
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_process_tree, suppress_other_loggers

@dataclasses.dataclass
class BenchArgs:
    """基准测试参数类
    
    属性:
        run_name: 运行名称,默认为"default"
        batch_size: 批处理大小元组,默认为(1,)
        input_len: 输入长度元组,默认为(1024,) 
        output_len: 输出长度元组,默认为(16,)
        result_filename: 结果文件名,默认为"result.jsonl"
        correctness_test: 是否进行正确性测试,默认为False
        cut_len: 仅用于正确性测试的截断长度,默认为4
    """
    run_name: str = "default"  # 运行的名称标识
    batch_size: Tuple[int] = (1,)  # 批处理大小配置
    input_len: Tuple[int] = (1024,)  # 输入序列长度配置
    output_len: Tuple[int] = (16,)  # 输出序列长度配置
    result_filename: str = "result.jsonl"  # 结果保存的文件名
    correctness_test: bool = False  # 是否进行正确性测试的标志
    cut_len: int = 4  # 正确性测试时使用的截断长度

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """添加命令行参数
        
        Args:
            parser: ArgumentParser对象,用于添加命令行参数
        """
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """从命令行参数创建BenchArgs实例
        
        Args:
            args: 解析后的命令行参数

        Returns:
            BenchArgs实例
        """
        # 使用默认值的类型来转换参数为正确的类型
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    """加载模型和分词器
    
    Args:
        server_args: 服务器参数
        port_args: 端口参数
        tp_rank: 张量并行的rank编号
        
    Returns:
        model_runner: 模型运行器实例
        tokenizer: 分词器实例
    """
    # 抑制其他日志输出
    suppress_other_loggers()
    # 只在rank 0上打印信息
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # 创建模型配置
    model_config = ModelConfig(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.context_length,
        model_override_args=server_args.json_model_override_args,
        is_embedding=server_args.is_embedding,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
    )
    
    # 创建模型运行器
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    
    # 打印最大token数
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    
    # 获取分词器
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    # 如果使用张量并行,等待所有进程同步
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer

def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    """为正确性测试准备输入数据
    
    Args:
        bench_args: 基准测试参数
        tokenizer: 分词器实例
        
    Returns:
        input_ids: 输入文本的token id列表
        reqs: 请求对象列表
    """
    # 定义测试用的提示语
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is", 
        "Today is a sunny day and I like",
    ]
    # 将提示语转换为token id
    input_ids = [tokenizer.encode(p) for p in prompts]
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,  # 设置温度为0,使输出确定性
        max_new_tokens=BenchArgs.output_len,  # 设置最大生成token数
    )

    reqs = []
    for i in range(len(prompts)):
        # 确保输入长度大于截断长度
        assert len(input_ids[i]) > bench_args.cut_len

        # 截取指定长度的输入
        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        # 创建请求对象
        req = Req(
            rid=i,  # 请求id
            origin_input_text=prompts[i],  # 原始输入文本
            origin_input_ids=tmp_input_ids,  # 截断后的输入token ids
            sampling_params=sampling_params,  # 采样参数
        )
        # 初始化前缀索引为空列表
        req.prefix_indices = []
        # 设置填充ids为原始输入ids
        req.fill_ids = req.origin_input_ids
        # 计算需要扩展的输入长度
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    """为正确性测试准备扩展输入数据
    
    Args:
        bench_args: 基准测试参数
        input_ids: 输入文本的token id列表
        reqs: 请求对象列表
        model_runner: 模型运行器实例
        
    Returns:
        reqs: 更新后的请求对象列表
    """
    for i in range(len(reqs)):
        req = reqs[i]
        # 将截断后的输入补充完整
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        # 从token池中获取前缀token的索引
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        # 计算需要扩展的输入长度
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    """为延迟测试准备合成输入数据
    
    Args:
        batch_size: 批处理大小
        input_len: 输入序列长度
        
    Returns:
        reqs: 请求对象列表
    """
    # 创建全1的输入矩阵
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,  # 设置温度为0进行贪婪搜索
        max_new_tokens=BenchArgs.output_len,  # 设置最大生成token数
    )

    # 创建请求列表
    reqs = []
    for i in range(len(input_ids)):
        # 为每个输入创建请求对象
        req = Req(
            rid=i,  # 请求id
            origin_input_text="",  # 原始输入文本为空
            origin_input_ids=list(input_ids[i]),  # 输入token ids
            sampling_params=sampling_params,  # 采样参数
        )
        # 初始化前缀索引为空列表
        req.prefix_indices = []
        # 设置填充ids为原始输入ids
        req.fill_ids = req.origin_input_ids
        # 计算需要扩展的输入长度
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    """扩展生成新的token
    
    Args:
        reqs: 请求对象列表
        model_runner: 模型运行器对象
        
    Returns:
        next_token_ids: 下一个token的id
        next_token_logits: 下一个token的logits
        batch: 批处理对象
    """
    # 初始化调度批处理
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,  # token池映射
        token_to_kv_pool=model_runner.token_to_kv_pool,    # KV缓存池映射
        tree_cache=None,                                    # 树缓存为空
        model_config=model_runner.model_config,             # 模型配置
        enable_overlap=False,                               # 禁用重叠计算
    )
    
    # 准备扩展操作
    batch.prepare_for_extend()
    
    # 获取model_worker_batch
    model_worker_batch = batch.get_model_worker_batch()
    
    # 创建前向传播批处理
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    
    # 执行模型前向传播
    logits_output = model_runner.forward(forward_batch)
    
    # 采样获取下一个token
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    """解码生成下一个token
    
    Args:
        input_token_ids: 输入token的id
        batch: 批处理对象
        model_runner: 模型运行器对象
        
    Returns:
        next_token_ids: 下一个token的id
        next_token_logits: 下一个token的logits
    """
    # 设置批处理的输出ids为输入token ids
    batch.output_ids = input_token_ids
    
    # 准备解码操作
    batch.prepare_for_decode()
    
    # 获取model_worker_batch
    model_worker_batch = batch.get_model_worker_batch()
    
    # 创建前向传播批处理
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    
    # 执行模型前向传播
    logits_output = model_runner.forward(forward_batch)
    
    # 采样获取下一个token
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    
    return next_token_ids, logits_output.next_token_logits

def correctness_test(
    server_args,  # 服务器参数
    port_args,    # 端口参数
    bench_args,   # 基准测试参数
    tp_rank,      # 张量并行的rank
):
    """正确性测试函数
    
    Args:
        server_args: 服务器参数,包含模型配置等
        port_args: 端口参数,用于分布式通信
        bench_args: 基准测试参数,包含输入长度、输出长度等
        tp_rank: 张量并行的rank编号
    """
    # 配置日志记录器
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # 只在rank 0上打印日志
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # 加载模型和分词器
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # 准备输入数据
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    # 如果设置了截断长度,先进行prefill
    if bench_args.cut_len > 0:
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # 准备扩展输入
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # 解码生成文本
    # 将输入token和第一个生成的token拼接
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    # 自回归生成剩余的token
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        # 将生成的token添加到输出序列
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # 打印每个prompt的生成结果
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len, device
):
    """执行一次延迟测试
    
    Args:
        run_name: 运行名称
        model_runner: 模型运行器实例
        rank_print: 打印函数
        reqs: 请求对象列表
        batch_size: 批处理大小
        input_len: 输入序列长度
        output_len: 输出序列长度
        device: 运行设备
        
    Returns:
        measurement_results: 测量结果字典,包含延迟和吞吐量等指标
    """
    # 检查批处理大小是否超过最大限制
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # 清空token池和KV缓存池
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool.clear()

    # 初始化测量结果字典
    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    # Prefill阶段
    synchronize(device)
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode阶段
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.time()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5:
            rank_print(
                f"Decode.  latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    # 记录从第二个输出token开始的解码时间
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    # 计算总体延迟和吞吐量
    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results

def latency_test(
    server_args,
    port_args, 
    bench_args,
    tp_rank,
):
    """执行延迟测试的主函数
    
    Args:
        server_args: 服务器参数
        port_args: 端口参数
        bench_args: 基准测试参数
        tp_rank: 张量并行的rank
    """
    # 配置日志记录器,添加TP rank前缀
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # 只在rank 0上打印日志
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # 加载模型和分词器
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # 准备预热阶段的输入数据
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # 执行预热,使用较短的输出长度加速预热过程
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        8,  # 使用较短的解码长度加速预热
        server_args.device,
    )
    rank_print("Benchmark ...")

    # 遍历所有参数组合进行测试
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        # 为每个参数组合准备输入数据
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        # 执行一次延迟测试
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
        )
        if ret is not None:
            result_list.append(ret)

    # 在rank 0上将结果以jsonlines格式写入文件
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def main(server_args, bench_args):
    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
