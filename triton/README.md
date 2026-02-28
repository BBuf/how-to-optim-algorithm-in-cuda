# Triton 学习笔记

## 目录结构

```
triton/
├── code/        # Triton / PyTorch 代码实现
└── meetup/      # Triton 中国 Meetup slides 汇总
```

## code/

依赖：

```
torch
triton
flash_attn
apex
```

| 文件 | 内容 |
|------|------|
| `layernorm.py` | Triton LayerNorm 实现 |
| `benchmark_layernorm.py` | PyTorch / Apex / Triton 三种实现基准测试 |
| `attention_in_pytorch.py` | PyTorch Attention 实现 |
| `flash_attention_v1_in_pytorch.py` | Flash Attention V1 PyTorch 实现 |
| `flash_attention_v2_in_pytorch.py` | Flash Attention V2 PyTorch 实现 |
| `flash_attention_v2_in_triton.py` | Flash Attention V2 Triton 实现 |

## meetup/

Triton 中国生态 Meetup 历期 slides，视频回放参见各 README 中的链接。
