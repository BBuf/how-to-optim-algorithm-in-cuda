import torch

N, d = 1024, 64  # 更新N和d的值

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

def standard_softmax_attention(Q, K, V):
    """
    执行标准的PyTorch softmax和attention计算。
    """
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_softmax, expected_attention

def flash_attention_v2(Q, K, V, B_r=64, B_c=768):
    """
    使用分块计算和在线softmax校正执行flash attention v2算法。
    """
    O = torch.zeros((N, d))  # 初始化输出矩阵
    l = torch.zeros((N, 1))  # 存储softmax分母
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值

    for j in range(0, N, B_c):  # 外循环遍历K和V的块
        Kj = K[j:j+B_c, :]
        Vj = V[j:j+B_c, :]

        for i in range(0, N, B_r):  # 内循环遍历Q的块
            Qi = Q[i:i+B_r, :]
            Sij = Qi @ Kj.T  # 计算得分矩阵
            mi_new = torch.max(torch.column_stack([m[i:i+B_r], torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
            Pij_hat = torch.exp(Sij - mi_new)  # 校正后的概率
            l[i:i+B_r] = torch.exp(m[i:i+B_r] - mi_new) * l[i:i+B_r] + torch.sum(Pij_hat, dim=1)[:, None]
            O[i:i+B_r] = O[i:i+B_r] * torch.exp(m[i:i+B_r] - mi_new) + Pij_hat @ Vj
            m[i:i+B_r] = mi_new

    O = O / l  # 根据softmax分母校正输出
    return O

# 执行flash attention计算
flash_attention_v2_output = flash_attention_v2(Q_mat, K_mat, V_mat)

# 执行标准的PyTorch softmax和attention计算
_, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)

# 断言flash attention计算的结果与标准计算结果是否接近
assert torch.allclose(flash_attention_v2_output, expected_attention), "Error in flash attention calculation"
