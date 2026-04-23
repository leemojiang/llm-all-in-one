import torch
import pytest
from minimind_learning.model.model_minimind import LayerNorm

def test_layernorm_output_shape():
    layer = LayerNorm(dim=4)
    x = torch.randn(2, 3, 4)  # [batch_size, seq_len, dim]
    out = layer(x)
    assert out.shape == x.shape

def test_layernorm_mean_std():
    layer = LayerNorm(dim=4)
    x = torch.randn(10, 5, 4)
    out = layer(x)
    # 计算归一化后的均值和标准差（近似）
    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1)

def test_layernorm_dtype_consistency():
    layer = LayerNorm(dim=4)
    x = torch.randn(2, 3, 4).half()  # 测试 float16 输入
    out = layer(x)
    assert out.dtype == x.dtype
