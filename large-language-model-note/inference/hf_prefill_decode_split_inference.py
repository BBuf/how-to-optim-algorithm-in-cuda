# /opt/nvidia/nsight-systems/2024.5.1/bin/nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o hf_qwen2.5_7b_flash_attn_fused_rope_rmsnorm_8_core python3 debug.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nvtx
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/bbuf/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", 
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "帮我计划一次去北京的旅行，我想明年春天出发，大概五天的行程。"

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

def sample_next_token(logits, temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.05):
    # 应用temperature
    logits = logits / temperature
    
    # 应用repetition_penalty
    # 这里需要对已生成的token施加惩罚,但由于是单token生成,暂时略过
    
    # 计算top_k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        logits[..., :] = float('-inf')
        logits[..., indices] = values
    
    # 计算top_p (nucleus sampling)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # 计算概率分布并采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

for i in range(1):
    with nvtx.annotate(f"step={i}", color="blue"):
        # prefill阶段
        with nvtx.annotate("prefill", color="green"):
            outputs = model(
                **model_inputs,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # 使用新的采样函数
            next_token = sample_next_token(
                next_token_logits,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                repetition_penalty=1.05
            )
            
            generated_tokens = [next_token]
            cur_len = 1
            max_new_tokens = 512

        # decode阶段
        with nvtx.annotate("decode", color="red"):
            while cur_len < max_new_tokens:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # 使用采样函数
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.8,
                    repetition_penalty=1.05
                )
                
                generated_tokens.append(next_token)
                cur_len += 1
                
                # 检查是否生成了结束符(注意Qwen2.5有两个可能的EOS token)
                if next_token.item() in [151645, 151643]:  # eos_token_ids
                    break
                    
        # 将所有生成的token拼接起来
        generated_ids = torch.cat(generated_tokens, dim=1)

# 只保留新生成的token
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
