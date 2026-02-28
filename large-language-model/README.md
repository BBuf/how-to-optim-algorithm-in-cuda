# large-language-model

大语言模型与 AI Infra 相关学习笔记与技术文章。

## 目录结构

```
large-language-model/
├── sglang/          # SGLang 推理框架相关（优化技巧、MLA、PD分离、Diffusion 等）
├── vllm/            # vLLM 相关（PagedAttention、CUDA Graph、P2P 等）
├── moe/             # MoE 模型推理优化（DeepEP、EP并行、moe kernel等）
├── diffusion/       # Diffusion 模型推理（Cache-DiT、LightX2V 等）
├── flash-attention/ # Flash Attention 相关（dispatch逻辑、FlashInfer 等）
├── trt-llm/         # TensorRT-LLM（量化、应用与部署）
├── kernels/         # 具体 kernel 优化分析与 debug 经验
├── infra/           # 通用 Infra 经验（ZeRO、开源框架参与记录等）
└── rl/              # 强化学习（veRL、GRPO 等）
```

## 外部参考资源

博客链接汇总见 [RESOURCES.md](RESOURCES.md)。
