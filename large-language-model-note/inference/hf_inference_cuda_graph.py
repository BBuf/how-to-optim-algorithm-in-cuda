# /opt/nvidia/nsight-systems/2024.5.1/bin/nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o hf_qwen2.5_7b_flash_attn_fused_rope_rmsnorm_8_core python3 debug.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nvtx
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/bbuf/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 设置静态缓存实现
# model.generation_config.cache_implementation = "static"

# 添加torch.compile优化
# model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.generation_config.cache_implementation = "static"

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

prompt = "帮我计划一次去北京的旅行，我想明年春天出发，大概五天的行程。"

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# warmup
for _ in range(2):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

torch.cuda.synchronize()
# profile

for i in range(1):
    with nvtx.annotate(f"step={i}", color="blue"):
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)