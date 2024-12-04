# 0x0. 背景

我们最近在用 SGLang 部署模型，当我们使用 TP2 的并行方式和 BF16 的dtype部署一个微调后的 LLama3 8B模型时发现了一个诡异的现象，最终确认为是掉入了一个RMS Norm的精度陷阱。

在SGLang启动的服务上重复对一个prompt请求1000次，模型会偶然出现3-5次奇怪的输出，这个输出不是乱码，但是显然不符合模型的效果。相反，如果在HuggingFace上请求1000次，则模型的输出一直都是正常的。

这个bug困扰了挺久，主要是觉得debug起来很费劲，复现不稳定，最开始心态已经摆了。

如果你有类似的bug，大模型推理存在精度问题，那这篇小记的处理方法可能会适合你。

**叠个甲**：这个bug的出现取决于模型训练框架的RMSNorm实现方式以及自定义数据长什么样。所以可能对于官方模型，使用你的自定义数据集微调的模型在推理时均不受这个问题的影响或者影响忽略不计，如果你不存在类似问题的话就图一乐就行。

# 0x1. 怀疑&准备工作

大模型的精度问题整体上要么是模型的问题，要么是sampling的问题，由于SGLang 1000次的结果中会出现几次不符合预期的输出，我首先猜测了sampling可能会有问题。

为了更好的定位bug，需要准备2个复现的脚本，首先是HF的：

```python
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "填入你的模型路径"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

generate_config = GenerationConfig(**{
    "do_sample": True,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 1
    })
def generate_from_prompt_and_compare(prompt, num_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        inputs=input_ids,
        generation_config=generate_config,
        )   

    for _ in range(num_sequences):
        outputs = model.generate(**gen_kwargs)
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        print(response)

prompt1 = "填入prompt"
generate_from_prompt_and_compare(prompt1, num_sequences=1000)
```

我们可以把它的输出重定向到log文件中。然后是SGLang的：

```python
mport requests
import json
import time

# API端点
url = "http://127.0.0.1:8000/v1/completions"

# 请求头
headers = {
    "Content-Type": "application/json"
}

# 请求体
data = {
    "model": "填入你的模型路径",
    "prompt": "填入prompt",
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_k": 1,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "stop": ["<|im_end|>", "<|endoftext|>"]
}

for i in range(1000):
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            text = result['choices'][0].get('text', '')
            with open('log_sglang_v0_1_6.txt', 'a', encoding='utf-8') as f:
                f.write(text + "\n")
        time.sleep(0.5)
        
    except Exception as e:
        print(f"第{i+1}次请求失败: {str(e)}")
        continue
```

注意把prompt和模型，包括sampling params都完全对齐。此外，我们需要针对SGLang启动一个serving服务，才能使用上面的脚本来发起请求执行推理。

为了验证对Sampling的怀疑，我使用了SGLang提供的naive PyTorch sampling来做推理（启动服务的时候指定`sampling_backend=pytorch`即可），但从输出结果来看和默认用flashinfer做sampling的结果几乎一致，仍然会出现不符合预期的输出。

# 0x2. 定位

既然如此，那有可能是模型推理出了问题。模型推理如果有问题一般在第一次prefill的结果就可以观测到差异。因此，我打算记录一下每一次推理的prefill之后的模型输出在过lm head之前的张量记为hidden_states，只要找到SGLang的对应bad prediction的hidden_states，然后和HF的对比一下可能就能看到区别。

HF的比较好处理，只需要把transformers仓库中llama3的model和config实现文件复制到模型仓库中，然后使用automap指定从当前模型仓库中加载模型实现，最后`AutoModelForCausalLM.from_pretrained`初始化模型的时候指定`trust_remote_code=True`就可以用上当前模型仓库中的model实现了。然后修改一下代码：

![](https://files.mdnice.com/user/59/47930b88-5276-43a7-9e21-9bf67093296c.png)

在SGLang的模型实现文件中也使用类似的代码记录每一次推理时prefill的lm head之前的hidden_states。需要注意2个点：
- 禁用chunked prefill。
- 由于warmup的存在，我们需要在发起请求之前先清空sglang保存每一次推理的hidden_states的文件夹，这样保存下来的Tensor对应的txt标号和HF才能对上。

这里只看第一次推理的HF和SGlang的hidden_states的前几个值：

HF:

```shell
[2.96875, -1.8046875, 1.8515625, 3.46875, -1.6484375, 2.125, 2.359375, 0.640625
```

SGLang:

```shell
[2.90625, -1.796875, 1.828125, 3.359375, -1.609375, 2.125, 2.34375, 0.6484375
```

可以看出hidden_states的差距已经比较大了，来到了1e-1的级别。

然后经过类似的方法往前找有差距的Layer，我最终发现是RMSNorm这个层在每一个Transformer Block都在放大精度差距。

所以，我尝试把RMSNorm模块的cuda实现替换成HF的naive实现，然后发现hidden_states的精度获得了明显的提升，并且重复运行1000次原始的prompt也不会再出现奇怪的输出了。

# 0x3. FlashInfer RMSNorm实现问题在哪？

在 https://github.com/sgl-project/sglang/issues/2258 提出了这个问题，很快得到了响应，问题指向flashinfer的`FusedAddRMSNormKernel`算子在v0.1.6存在精度低(1e-2)的问题，并且在近期获得了精度提升。

按照开发者的建议安装了flashinfer的nightly，对上面的prompt运行1000次，输出中均符合预期，问题得到解决。

# 0x4. FlashInfer RMSNorm 精度提升原理

我们可以看一下flashinfer里面`FusedAddRMSNormKernel`精度提示的原理，它在这个PR：https://github.com/flashinfer-ai/flashinfer/pull/587

原理为：

当 `sizeof(T) = 2` 时，读取的输入和残差（浮点数 x）的和被分成高位和低16位两部分，分别保存到 input 和 residual 中。之后，input 和 residual 被读出并组合成 x，目的是为了提高后续 `x * rms_rcp` 运算的精度。

这样可以将精度从1e-2提高到1e-3。

对应的代码实现为：

![](https://files.mdnice.com/user/59/2f9e5d24-24ae-45af-a1bf-0f9d7784a949.png)

那么，VLLM在输入为FP16时是否存在这个问题呢？我们可以使用下面的测试验证：

```python
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    """Root mean square normalization.
    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.randn(hidden_size, dtype=dtype))
    
    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual
    
    def forward_with_vllm_op(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm import _custom_ops as ops
        if residual is not None:
            ops.fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out
    
    def forward_with_flashinfer(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from flashinfer.norm import (
                fused_add_rmsnorm,
                rmsnorm,
            )
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

if __name__ == "__main__":
    def test_rmsnorm(dtype: torch.dtype, hidden_size: int = 2048, seq_len: int = 1024):
        print(f"\n测试 RMSNorm - dtype: {dtype}")
        
        # 初始化模型和测试数据
        rms_norm = RMSNorm(hidden_size, dtype=dtype).cuda()
        input_tensor = torch.randn(seq_len, hidden_size, device="cuda:0", dtype=dtype)
        
        # 测试有残差连接的情况
        print("\n场景1: 有残差连接")
        residual = torch.randn(seq_len, hidden_size, device="cuda:0", dtype=dtype)
        
        native_output, _ = rms_norm.forward_native(x=input_tensor, residual=residual)
        vllm_output, _ = rms_norm.forward_with_vllm_op(
            x=input_tensor.clone(), residual=residual.clone()
        )
        flashinfer_output, _ = rms_norm.forward_with_flashinfer(
            x=input_tensor.clone(), residual=residual.clone()
        )
        
        vllm_match = torch.allclose(native_output, vllm_output, rtol=1e-03, atol=1e-03)
        flashinfer_match = torch.allclose(native_output, flashinfer_output, rtol=1e-03, atol=1e-03)
        
        print(f"VLLM 实现与原生实现匹配: {vllm_match}")
        print(f"FlashInfer 实现与原生实现匹配: {flashinfer_match}")

    
    test_rmsnorm(dtype=torch.float16)  
```

结果：

```shell
测试 RMSNorm - dtype: torch.float16

场景1: 有残差连接
VLLM 实现与原生实现匹配: False
FlashInfer 实现与原生实现匹配: True
```

VLLM的`FusedAddRMSNormKernel`精度和flashinfer v0.1.6一样在FP16数据类型上只能维持到1e-2的级别，在我们的模型上使用VLLM来serving和测试时也能较低频率的观测到不符合预期的输出。

如果测试torch.bfloat16，我发现即使是flashinfer nightly也无法保持1e-3的精度。

# 0x5. 当前限制

当切换到flashinfer nightly后，目前我们需要在serving模型时指定`--dtype=float16`来避免flashinfer的Attention在创建辅助数据结构也就是调用`plan`相关函数时没有显式指定dtype的问题。否则使用`--dtype=bfloat16`会报这个错误：https://github.com/sgl-project/sglang/pull/2295#issuecomment-2509682065

还有一个限制是，当使用bfloat16 dtype来serving时，目前即使是flashinfer nightly也无法保持`FusedAddRMSNormKernel`算子的精度在1e-3级别。


# 0x6. 总结

总结一下，目前SGLang可以正确serving上述自定义llama3-8b tp2的方式有：

- flashinfer nightly + fused_rms_norm + torch.float16
- rms_norm_naive + torch.bfloat16/torch.float16

正确性未知：

- flashinfer nighly + fused_rmns_norm + torch.bfloat16

但大概率不可以，因为上面的单测验证`FusedAddRMSNormKernel`的精度还是不够高。

大模型推理框架查精度问题其实挺费力的，希望这里提到的步骤可以给有相同困惑的读者一些启发。我也要推荐一下SGLang框架，从我的个人的使用场景来看，它的性能已经是目前工业界最SOTA的一档，开发者团队也很热心，负责。在SGLang仓库下也可以看到非常创新的点子例如推理框架自己的Expert Parallel，Fused MoE的torch compile应用，SDPA/Flashinfer Backend，支持Embeeding输入进行Serving等等，后续也学习分享下。



