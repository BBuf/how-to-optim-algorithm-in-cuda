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
    O = torch.zeros((N, d))  # 初始化O为(N, d)的形状，实际上对应伪代码第5行的O初始化
    l = torch.zeros((N, 1))  # 初始化l为(N)的形状，实际上对应伪代码第5行的l初始化
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值，初始化为负无穷大，对应伪代码的第5行

    # 对应伪代码的第3行，for 1<=i<T_r，注意这里是把Q分成了Tr=[N/B_r]块，每一块的大小是[B_r, d]这么大
    # 所以在python实现的时候就直接通过一个步长为B_r的循环来处理
    for i in range(0, N, B_r):
        Qi = Q[i:i+B_r, :]
        # 对应伪代码的第 6 行，for 1<=j<=T_c，注意这里是把K, V分成了T_c=[N/B_c]块，每一块的大小是[B_c, d]这么大
        # 所以在python实现的时候就直接通过一个步长为B_c的循环来处理 
        for j in range(0, N, B_c):  # 内循环遍历Q的块
            Kj = K[j:j+B_c, :]
            Vj = V[j:j+B_c, :]

            # 对应伪代码的第8行：on chip, compute Sij，Sij的形状是[B_r, B_c]
            Sij = Qi @ Kj.T
            # 对应伪代码的第9行求m_i^(j)的操作，mi_new的形状是B_r
            mi_new = torch.max(torch.column_stack([m[i:i+B_r], torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
            # 对应伪代码的第9行求Pij_hat的操作，Pij_hat的形状是(B_r x B_c)，和Sij一致
            Pij_hat = torch.exp(Sij - mi_new)
            # 对应伪代码的第9行求lij的操作
            l[i:i+B_r] = torch.exp(m[i:i+B_r] - mi_new) * l[i:i+B_r] + torch.sum(Pij_hat, dim=1)[:, None]
            # 对应伪代码的第10行求O_ij的操作
            O[i:i+B_r] = O[i:i+B_r] * torch.exp(m[i:i+B_r] - mi_new) + Pij_hat @ Vj
            m[i:i+B_r] = mi_new

    O = O / l  # 对应伪代码第12行，根据softmax的分母校正输出
    return O

# 执行flash attention v2计算
flash_attention_v2_output = flash_attention_v2(Q_mat, K_mat, V_mat)

# 执行标准的PyTorch softmax和attention计算
_, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)

# 断言flash attention计算的结果与标准计算结果是否接近
assert torch.allclose(flash_attention_v2_output, expected_attention), "Error in flash attention calculation"
