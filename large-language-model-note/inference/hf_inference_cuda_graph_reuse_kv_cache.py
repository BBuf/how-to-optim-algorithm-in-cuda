# /opt/nvidia/nsight-systems/2024.5.1/bin/nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o hf_qwen2.5_7b_sdpa_compile_new python3 debug.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import torch
import nvtx

model_name = "/mnt/bbuf/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "介绍一下大熊猫"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
prompt_length = input_ids.input_ids.shape[1]
model.generation_config.max_new_tokens = 256

past_key_values = StaticCache(
    config=model.config,
    batch_size=1,
    # If you plan to reuse the cache, make sure the cache length is large enough for all cases
    max_cache_len=32768,
    device=model.device,
    dtype=model.dtype
)
with nvtx.annotate(f"step=1", color="blue"):
    outputs = model.generate(**input_ids, past_key_values=past_key_values)
print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# pass in the generated text and the same cache object to continue generation from where it left off. Optionally, in a
# multi-turn conversation, append the new user input to the generated text.
new_input_ids = outputs
with nvtx.annotate(f"step=2", color="red"):
    outputs = model.generate(new_input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# pass in the generated text and the same cache object to continue generation from where it left off. Optionally, in a
# multi-turn conversation, append the new user input to the generated text.
new_input_ids = outputs
with nvtx.annotate(f"step=3", color="green"):
    outputs = model.generate(new_input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


for i in range(10):
    new_input_ids = outputs
    with nvtx.annotate(f"step={i+4}", color="yellow"):
        outputs = model.generate(new_input_ids, past_key_values=past_key_values)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
