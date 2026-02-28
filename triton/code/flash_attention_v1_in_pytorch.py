import torch

N, d = 1024, 64  # 更新N和d的值

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

def standard_softmax_attention(Q, K, V):
    """
    执行标准的pytorch softmax和attention计算。
    """
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_softmax, expected_attention

def flash_attention(Q, K, V, B_r=64, B_c=768):
    """
    使用分块计算和在线softmax校正执行flash attention算法。
    """
    O = torch.zeros((N, d))  # 初始化输出矩阵，对应伪代码的第2行
    l = torch.zeros((N, 1))  # 存储softmax分母，对应伪代码的第2行
    m = torch.full((N, 1), -torch.inf)  # 存储每个block的最大值，对应伪代码的第2行

    # 对应伪代码的第5行，for 1<=j<=T_c，注意这里是把K, V分成了T_c=[N/B_c]块，每一块的大小是[B_c, d]这么大
    # 所以在python实现的时候就直接通过一个步长为B_c的循环来处理
    for j in range(0, N, B_c):
        # 下面三行就对应了伪代码的第6行，Load Kj, Vj from HBM to on-chip SRAM
        # 但是这里是单纯的 python 实现，我们不可能真的把这一块内存从HBM上放到SRAM上
        # 这里只是一个伪代码的逻辑说明，可以假装它做到了，因为在Triton里面真的可以在Python层做到。
        j_end = j + B_c
        Kj = K[j:j_end, :]
        Vj = V[j:j_end, :]

        # 对应伪代码的第7行，for 1<=i<T_r，注意这里是把Q分成了Tr=[N/B_r]块，每一块的大小是[B_r, d]这么大
        # 所以在python实现的时候就直接通过一个步长为B_r的循环来处理
        for i in range(0, N, B_r):
            i_end = i + B_r
            mi = m[i:i_end, :]
            li = l[i:i_end, :]
            Oi = O[i:i_end, :]
            Qi = Q[i:i_end, :]

            # 对应伪代码的第9行：on chip, compute Sij，Sij的形状是[B_r, B_c]
            Sij = Qi @ Kj.T
            # 对应伪代码的第10行
            mij_hat = torch.max(Sij, dim=1).values[:, None]
            pij_hat = torch.exp(Sij - mij_hat)
            lij_hat = torch.sum(pij_hat, dim=1)[:, None]

            # 对应伪代码的第11行求mi_new的操作，注意这里要对两个张量求整体的max，所以才有这个stack操作
            mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
            # 对应伪代码的第11行求li_new的操作
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat
            # 对应伪代码的第12行，更新O_i。这里容易有一个疑问，伪代码上有一个diag操作，为什么下面的实现忽略了
            # 这是因为这个diag是作用在vector上的，实际上是为了在伪代码上能对应上维度，而PyTorch的实现是自动
            # 支持张量广播机制的，所以这里可以直接计算。
            O_i = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

            # 对应伪代码的第13行，更新m_i，l_i，O_i。
            m[i:i_end, :] = mi_new
            l[i:i_end, :] = li_new
            O[i:i_end, :] = O_i

    return O

# 执行flash attention计算
flash_attention_output = flash_attention(Q_mat, K_mat, V_mat)

# 执行标准的pytorch softmax和attention计算
expected_softmax, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)

# 断言flash attention计算的结果与标准计算结果是否接近
assert torch.allclose(flash_attention_output, expected_attention), "error in flash attention calculation"
