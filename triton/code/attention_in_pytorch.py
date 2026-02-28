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

def safe_softmax_attention(Q, K, V):
    """
    执行安全的softmax和attention计算。
    """
    S_mat = Q @ K.T
    row_max = torch.max(S_mat, dim=1).values[:, None]
    input_safe = S_mat - row_max
    softmax_numerator = torch.exp(input_safe)
    softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
    safe_softmax = softmax_numerator / softmax_denominator
    matmul_result = safe_softmax @ V
    return safe_softmax, matmul_result

# 使用标准softmax和attention计算
expected_softmax, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)
# 使用安全softmax和attention计算
safe_softmax, safe_attention = safe_softmax_attention(Q_mat, K_mat, V_mat)

# 断言两种方法计算的softmax和attention结果是否接近
assert torch.allclose(safe_softmax, expected_softmax), "error in safe softmax"
assert torch.allclose(safe_attention, expected_attention), "error in safe attention"
