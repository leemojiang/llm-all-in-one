# PyTorch Cookbook

这篇笔记主要整理我在学习 LLM 和阅读训练代码时经常会碰到的 PyTorch 张量操作。目标不是写成一份完整文档，而是把那些最常用、最容易混淆、最需要建立直觉的内容整理出来，方便后面反复查。

## 1. PyTorch 张量基础

### 1.1 张量的维度和常见 shape

在 PyTorch 里，最先要建立的直觉就是：很多操作本质上都和 shape 有关。  
同一个算子，一旦输入张量的维度理解错了，后面的广播、拼接、索引、矩阵乘法都会跟着错。

一些常见模型里的张量 shape：

| 模型 | 常见 shape | 含义 |
| --- | --- | --- |
| MLP | `(batch_size, feature_dim)` | 最后一个维度通常是特征维度 |
| RNN / LSTM | `(batch_size, seq_len, dim)` | 每个时间步对应一个 embedding |
| Transformer | `(batch_size, seq_len, dim)` | 每个 token 对应一个 embedding |
| CNN | `(batch_size, channels, H, W)` | 最后两个维度通常是空间维度 |

在 LLM 代码里，经常会看到下面这种张量：

```python
x.shape == (batch_size, seq_len, hidden_dim)
```

这里：

- `batch_size` 表示一批样本的数量。
- `seq_len` 表示序列长度，也就是 token 个数。
- `hidden_dim` 表示每个 token 对应的隐藏向量维度。

很多层默认都是“最后一个维度是特征维度”这个思路在工作，比如 `LayerNorm`、线性层前后的很多广播操作。

### 1.2 索引机制

PyTorch 的索引方式和 NumPy 很接近。最常见的是切片、整数索引和维度插入。

```python
import torch

x = torch.randn(2, 3, 4)

x[0].shape        # (3, 4)
x[:, 1].shape     # (2, 4)
x[:, :, 0].shape  # (2, 3)
```

可以把它理解成：每写一个索引，就是在对应维度上做一次选择。

### 1.3 `None` 和 `...`

`None` 是 Python 的语法糖，用于在当前位置插入一个长度为 1 的新维度，效果等价于 `unsqueeze()`。

```python
x = torch.randn(2, 3, 4, 5)

y = x[:, :, :, None, :]
z = x.unsqueeze(3)

print(y.shape)  # (2, 3, 4, 1, 5)
print(z.shape)  # (2, 3, 4, 1, 5)
```

所以：

```python
x[:, :, :, None, :] == x.unsqueeze(3)
```

`...` 表示“中间剩下的维度全部保留”，在高维张量里很好用：

```python
x[..., 0]     # 取最后一个维度上的第 0 个元素
x[..., None]  # 在最后面插入一个新维度
```

### 1.4 广播机制

广播机制是 PyTorch 里最重要的基础概念之一。几乎所有逐元素运算，比如加法、减法、乘法、除法，都支持广播。

例如：

```python
self.weight * normalized_x
```

如果：

```python
self.weight.shape == (d,)
normalized_x.shape == (batch_size, ..., d)
```

那么结果 shape 为：

```python
(batch_size, ..., d)
```

原因是广播会从最后一个维度开始对齐。

广播规则可以简化为：

1. 从最后一个维度开始对齐，也就是右对齐。
2. 如果两个维度相等，或者其中一个维度是 `1`，就可以广播。
3. 如果两个维度既不相等，也都不是 `1`，就会报错。

例如：

```python
[m, n] * [n]
[l, m, n] * [n]
[l, m, n] * [m, n]
[l, m, n] * [m, 1]
[l, m, n] * [1, m, 1]
```

一个比较有用的理解方式是：把没有出现的维度当成自动补了一个 `1`，然后从右往左匹配。  
和 `1` 对应的那个维度，可以理解成“这一维要被复制展开”。

需要注意的是：

- 广播不会真的复制数据。
- 它只是逻辑上把 shape 扩展到可以逐元素运算的形式。
- 所以广播通常很省内存。

### 1.5 连续内存、`stride` 和 contiguous

PyTorch 里的张量不只是 shape，还有内存布局问题。

连续张量可以粗略理解为：它在内存中的排布是按当前 shape 顺序紧密存储的，没有跳着访问，也没有重复引用。

判断一个张量是否连续：

```python
x.is_contiguous()
```

有些操作会返回非连续张量，比如：

- `transpose`
- `permute`
- `expand`

这些操作很多时候并不会真的复制数据，而只是修改张量的“视图解释方式”。这时就会涉及 `stride`。

`stride` 可以理解成：在某个维度上移动一步时，底层内存地址要跳多少。

```python
x.stride()
```

如果你只是想先建立直觉，可以先记住一句话：

- `shape` 决定“怎么看这个张量”。
- `stride` 决定“按这个看法去访问内存时怎么跳”。

这也是为什么有些张量虽然 shape 看起来没问题，但不能直接 `view()`。

如果需要查看底层存储大小，可以用：

```python
x.untyped_storage().size()
```

## 2. 形状变换与维度操作

### 2.1 `unsqueeze()` 和 `squeeze()`

`torch.unsqueeze(dim)` 用于在指定位置插入一个长度为 1 的维度。

```python
x = torch.tensor([1, 2, 3])  # shape: [3]

print(x.unsqueeze(0).shape)  # [1, 3]
print(x.unsqueeze(1).shape)  # [3, 1]
```

`squeeze()` 用于去掉长度为 1 的维度。

```python
x = torch.randn(1, 3, 1, 4)

print(x.squeeze().shape)    # [3, 4]
print(x.squeeze(0).shape)   # [3, 1, 4]
print(x.squeeze(2).shape)   # [1, 3, 4]
```

这两个函数在对齐 shape、准备广播、给 `matmul` 或 `gather` 喂输入时非常常见。

<details style="color:rgb(128,128,128)">
<summary>numpy: expand_dims() / squeeze()</summary>

```python
import numpy as np

x = np.array([1, 2, 3])

print(np.expand_dims(x, axis=0).shape)  # (1, 3)
print(np.expand_dims(x, axis=1).shape)  # (3, 1)

y = np.random.randn(1, 3, 1, 4)
print(np.squeeze(y).shape)              # (3, 4)
print(np.squeeze(y, axis=0).shape)      # (3, 1, 4)
```

</details>

### 2.2 `view()` vs `reshape()`

这两个函数都可以改 shape，但区别非常重要。

| 特性 | `view()` | `reshape()` |
| --- | --- | --- |
| 是否要求连续内存 | 是，必须是 contiguous tensor | 否，不连续时会自动处理 |
| 是否可能复制数据 | 不会，只返回 view | 可能复制数据 |
| 失败行为 | 张量不连续时直接报错 | 通常会自动返回可用结果 |

例子：

```python
x = torch.arange(12)
y = x.view(3, 4)
z = x.reshape(3, 4)
```

如果张量经过了 `transpose()` 或 `permute()`，这时往往不能直接 `view()`，需要先：

```python
x = x.contiguous().view(...)
```

或者直接：

```python
x = x.reshape(...)
```

经验上可以这么记：

- 你明确知道张量是连续的，并且想要“只改视图不复制数据”，可以用 `view()`。
- 如果只是想安全地改 shape，通常 `reshape()` 更省心。

<details style="color:rgb(128,128,128)">
<summary>numpy: reshape()</summary>

```python
import numpy as np

x = np.arange(12)
y = x.reshape(3, 4)

print(y.shape)  # (3, 4)
```

</details>

### 2.3 `transpose()`、`permute()` 和 `contiguous()`

`transpose(dim0, dim1)` 用于交换两个维度。

```python
x = torch.randn(2, 3, 4)
print(x.transpose(1, 2).shape)  # (2, 4, 3)
```

`permute()` 更一般，可以一次性重排多个维度。

```python
x = torch.randn(2, 3, 4)
print(x.permute(2, 0, 1).shape)  # (4, 2, 3)
```

在 Transformer 代码里，`permute()` 和 `transpose()` 很常见，比如把张量从：

```python
(batch_size, seq_len, num_heads, head_dim)
```

改成：

```python
(batch_size, num_heads, seq_len, head_dim)
```

很多时候这些操作只会修改视图，不会复制数据，因此结果张量常常不是连续的。

这时如果后面要 `view()`，通常需要先调用：

```python
x = x.contiguous()
```

<details style="color:rgb(128,128,128)">
<summary>numpy: transpose()</summary>

```python
import numpy as np

x = np.random.randn(2, 3, 4)

print(np.swapaxes(x, 1, 2).shape)   # (2, 4, 3)
print(np.transpose(x, (2, 0, 1)).shape)  # (4, 2, 3)
```

</details>

### 2.4 `expand()` vs `repeat()` vs `repeat_interleave()`

这几个函数经常一起混。

#### `expand()`：广播视图，不复制数据

`expand()` 返回的是一个广播视图。它不会真的复制数据，所以很省内存，但只能扩展原来 size 为 `1` 的维度。

```python
x = torch.tensor([[1, 2]])
x_expand = x.expand(3, 2)

print(x_expand.shape)  # (3, 2)
```

这里看起来像是把第一维复制成了 3 份，但实际上底层数据并没有复制。

#### `repeat()`：真实复制数据

`repeat()` 会真的复制内容，生成一个新的张量。

```python
x = torch.tensor([[1, 2]])
x_repeat = x.repeat(2, 3)

print(x_repeat.shape)  # (2, 6)
```

#### `repeat_interleave()`：逐元素重复

如果你想重复的是“元素”而不是整个维度块，可以用 `repeat_interleave()`。

```python
x = torch.tensor([1, 2, 3])
print(torch.repeat_interleave(x, repeats=2))
# tensor([1, 1, 2, 2, 3, 3])
```

总结如下：

| 函数 | 是否复制数据 | 是否节省内存 | 是否支持任意维度扩展 | 典型用途 |
| --- | --- | --- | --- | --- |
| `repeat` | 是 | 否 | 是 | 需要真实复制数据 |
| `expand` | 否 | 是 | 否，只能扩展 size 为 1 的维度 | 做广播视图 |
| `repeat_interleave` | 是 | 否 | 元素级重复 | 扩展标签、索引、位置等 |

还要注意一点：  
`expand()` 得到的张量常常不是标准连续内存布局，所以后面如果继续 `view()` 或某些依赖连续内存的操作，需要格外小心。

<details style="color:rgb(128,128,128)">
<summary>numpy: broadcast_to() / tile() / repeat()</summary>

```python
import numpy as np

x = np.array([[1, 2]])

print(np.broadcast_to(x, (3, 2)).shape)  # (3, 2)
print(np.tile(x, (2, 3)).shape)          # (2, 6)
print(np.repeat(np.array([1, 2, 3]), 2))
# [1 1 2 2 3 3]
```

</details>

### 2.5 `flatten()`

`flatten()` 用于把若干连续维度压平成一个维度。

```python
x = torch.randn(2, 3, 4)
print(x.flatten().shape)        # (24,)
print(x.flatten(1).shape)       # (2, 12)
print(x.flatten(0, 1).shape)    # (6, 4)
```

这在把多维特征送进线性层、或者把批量维和时间维合并时很常见。

<details style="color:rgb(128,128,128)">
<summary>numpy: reshape() / ravel()</summary>

```python
import numpy as np

x = np.random.randn(2, 3, 4)

print(x.reshape(-1).shape)     # (24,)
print(x.reshape(2, -1).shape)  # (2, 12)
print(x.reshape(6, 4).shape)   # (6, 4)
```

</details>

## 3. 张量拼接与组合

### 3.1 `torch.cat()`

`torch.cat([tensor1, tensor2, ...], dim=0)` 用于在已有维度上拼接张量。

特点：

- 按指定维度进行拼接。
- 除了拼接的那个维度之外，其他维度必须完全一致。
- `torch.cat()` 本身不支持广播。

例子：

```python
x = torch.randn([2, 3, 4])
y = torch.randn([2, 3, 3])

print(torch.cat([x, x]).shape)
print(torch.cat([x, x], dim=1).shape)
print(torch.cat([x, y], dim=-1).shape)
```

输出：

```python
torch.Size([4, 3, 4])
torch.Size([2, 6, 4])
torch.Size([2, 3, 7])
```

<details style="color:rgb(128,128,128)">
<summary>numpy: concatenate()</summary>

```python
import numpy as np

x = np.random.randn(2, 3, 4)
y = np.random.randn(2, 3, 3)

print(np.concatenate([x, x], axis=0).shape)   # (4, 3, 4)
print(np.concatenate([x, x], axis=1).shape)   # (2, 6, 4)
print(np.concatenate([x, y], axis=-1).shape)  # (2, 3, 7)
```

</details>

### 3.2 `torch.stack()`

`torch.stack([tensor1, tensor2, ...], dim=0)` 会先插入一个新维度，再把多个张量沿这个新维度叠起来。

特点：

- 所有输入张量的 shape 必须完全一致。
- 结果比原张量多一个维度。
- `dim` 表示新维度插入的位置。

例如：

```python
x = torch.randn(2, 3)
y = torch.randn(2, 3)

print(torch.stack([x, y], dim=0).shape)  # (2, 2, 3)
print(torch.stack([x, y], dim=1).shape)  # (2, 2, 3)
```

虽然 shape 都是 `(2, 2, 3)`，但维度语义不同。

可以粗略理解成：

- `cat` 是在原有轴上接起来。
- `stack` 是新增一个轴，把多个张量摞起来。

<details style="color:rgb(128,128,128)">
<summary>numpy: stack()</summary>

```python
import numpy as np

x = np.random.randn(2, 3)
y = np.random.randn(2, 3)

print(np.stack([x, y], axis=0).shape)  # (2, 2, 3)
print(np.stack([x, y], axis=1).shape)  # (2, 2, 3)
```

</details>

### 3.3 `torch.hstack()` 和 `torch.vstack()`

这两个函数本质上可以理解成 `cat()` 的语法糖。

| 函数 | 本质操作 | 默认拼接维度 | 适用场景 |
| --- | --- | --- | --- |
| `torch.cat()` | 通用拼接函数 | 手动指定 `dim` | 最灵活 |
| `torch.hstack()` | 水平拼接 | 最后一维 | 类似 NumPy 的 `hstack` |
| `torch.vstack()` | 垂直拼接 | 第 0 维 | 类似 NumPy 的 `vstack` |

需要注意：

- 它们只适用于维度大于等于 1 的张量。
- 对一维张量来说，`hstack` 和 `vstack` 的行为不完全一样。
- `vstack` 往往会先把一维张量视为行向量再拼接。

<details style="color:rgb(128,128,128)">
<summary>numpy: hstack() / vstack()</summary>

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(np.hstack([x, y]))      # [1 2 3 4 5 6]
print(np.vstack([x, y]).shape)  # (2, 3)
```

</details>

### 3.4 `torch.outer()`

`torch.outer(a, b)` 要求输入都是一维张量。

如果：

```python
a.shape == (m,)
b.shape == (n,)
```

那么：

```python
torch.outer(a, b).shape == (m, n)
```

本质上就是把两个向量做外积，每一对元素都相乘一次。

<details style="color:rgb(128,128,128)">
<summary>numpy: outer()</summary>

```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4, 5])

print(np.outer(a, b))
# [[ 3  4  5]
#  [ 6  8 10]]
```

</details>

## 4. 索引、选择与收集

### 4.1 基础切片和布尔索引

基础切片前面已经讲过。另一个很常用的是布尔索引。

```python
x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2

print(mask)      # tensor([False, False,  True,  True,  True])
print(x[mask])   # tensor([3, 4, 5])
```

这在筛选 loss、过滤 padding、提取满足条件的位置时很常见。

<details style="color:rgb(128,128,128)">
<summary>numpy: 布尔索引</summary>

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
mask = x > 2

print(mask)     # [False False  True  True  True]
print(x[mask])  # [3 4 5]
```

</details>

### 4.2 `torch.gather()`

`torch.gather(input, dim, index)` 是 LLM 代码里非常常见的函数，尤其是在：

- 按 token 位置取值
- 从 logits 中取出目标 token 对应分数
- 根据索引收集某一维上的元素

它的参数含义：

- `input`：原始张量。
- `dim`：沿着哪一个维度取值。
- `index`：要取哪些位置。

形状规则：

- `index.shape` 必须和 `input.shape` 在除了 `dim` 之外的维度上保持一致。
- 在 `dim` 这个维度上，`index.size(dim)` 可以和 `input.size(dim)` 不同。
- 输出的 shape 就等于 `index.shape`。

例子：

```python
input = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])  # shape: [2, 3]

index = torch.tensor([
    [2, 1, 0],
    [0, 1, 2]
])  # shape: [2, 3]

out = torch.gather(input, dim=1, index=index)

print(out)
```

输出：

```python
tensor([
    [30, 20, 10],
    [40, 50, 60]
])
```

如果你把它放到 LLM 场景里，可以把 `input` 理解成 logits，把 `index` 理解成目标 token id，就比较容易理解为什么它这么常用。

<details style="color:rgb(128,128,128)">
<summary>numpy: take_along_axis()</summary>

```python
import numpy as np

input = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

index = np.array([
    [2, 1, 0],
    [0, 1, 2]
])

out = np.take_along_axis(input, index, axis=1)
print(out)
# [[30 20 10]
#  [40 50 60]]
```

</details>

### 4.3 `index_select()`

`index_select()` 也是按索引取值，但它比 `gather()` 更简单，适合“在某个维度上统一选几列/几行”的场景。

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

index = torch.tensor([0, 2])
out = torch.index_select(x, dim=1, index=index)

print(out)
```

输出：

```python
tensor([
    [10, 30],
    [40, 60]
])
```

可以简单记成：

- `gather()` 更灵活，适合“每个位置取的索引都可能不同”。
- `index_select()` 更适合“整行整列统一挑选”。

<details style="color:rgb(128,128,128)">
<summary>numpy: take()</summary>

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

index = np.array([0, 2])
out = np.take(x, index, axis=1)

print(out)
# [[10 30]
#  [40 60]]
```

</details>

## 5. 常用 API 速查

这一节不打算写成完整手册，只整理一些在模型代码里最常见的 API。

### 5.1 查看张量属性

```python
x.shape
x.size()
x.dim()
x.dtype
x.device
x.stride()
x.is_contiguous()
```

### 5.2 常见创建函数

```python
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(10)
torch.randn(2, 3)
torch.tensor([1, 2, 3])
torch.zeros_like(x)
torch.ones_like(x)
torch.randn_like(x)
```

### 5.3 常见形状操作

```python
x.unsqueeze(dim)
x.squeeze(dim)
x.view(...)
x.reshape(...)
x.flatten(...)
x.transpose(dim0, dim1)
x.permute(...)
x.contiguous()
```

### 5.4 常见拼接操作

```python
torch.cat([...], dim=...)
torch.stack([...], dim=...)
torch.hstack([...])
torch.vstack([...])
torch.outer(a, b)
```

### 5.5 常见归约操作

```python
x.sum(dim=...)
x.mean(dim=...)
x.max(dim=...)
x.argmax(dim=...)
```

这里需要注意：

- `sum`、`mean`、`max` 这类操作往往会让某个维度消失。
- 如果后面还要保留这个维度参与广播，可以用 `keepdim=True`。

例如：

```python
x = torch.randn(2, 3, 4)
y = x.mean(dim=-1, keepdim=True)

print(y.shape)  # (2, 3, 1)
```

### 5.6 常见逐元素运算

```python
x + y
x - y
x * y
x / y
torch.exp(x)
torch.log(x)
torch.sqrt(x)
torch.clamp(x, min=0.0)
```

这类操作通常都支持广播。

### 5.7 `matmul` / `bmm` / `einsum`

这几个函数在 Transformer 和注意力代码里非常常见。它们本质上都和“乘法 + 某些维度上的求和”有关，但抽象层级不一样。

#### `torch.matmul()`

`matmul()` 是最通用的矩阵乘法接口。它会根据输入维度自动选择行为：

- 两个一维张量：做点积
- 两个二维张量：做标准矩阵乘法
- 高维张量：把前面的维度当作 batch 维，最后两维做矩阵乘法

例如：

```python
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.matmul(x, y)

print(z.shape)  # (2, 4)
```

在高维情况下：

```python
q = torch.randn(8, 16, 128, 64)
k = torch.randn(8, 16, 64, 128)
scores = torch.matmul(q, k)

print(scores.shape)  # (8, 16, 128, 128)
```

这里前两维 `(8, 16)` 可以看作 batch 维，最后两维做矩阵乘法。这正是多头注意力里非常典型的写法。

#### `torch.bmm()`

`bmm()` 是 batched matrix multiplication，只支持三维张量。

如果：

```python
x.shape == (b, m, n)
y.shape == (b, n, p)
```

那么：

```python
torch.bmm(x, y).shape == (b, m, p)
```

例子：

```python
x = torch.randn(32, 128, 64)
y = torch.randn(32, 64, 128)
z = torch.bmm(x, y)

print(z.shape)  # (32, 128, 128)
```

可以把它理解成：对 batch 中的每一对矩阵分别做一次普通矩阵乘法。

经验上：

- 如果你想写通用代码，通常 `matmul()` 更常用。
- 如果你明确知道自己处理的是三维 batch 矩阵，`bmm()` 语义更直接。

#### `torch.einsum()`

`einsum()` 更像是一种“张量计算记号”，它允许你直接描述每个维度之间如何对应、如何求和。

例如矩阵乘法：

```python
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = torch.einsum("ik,kj->ij", x, y)
```

这里：

- `i` 表示 `x` 的第 0 维
- `k` 表示被求和掉的中间维
- `j` 表示 `y` 的第 1 维

在注意力里，一个常见写法是：

```python
scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
```

符号说明：

- `b`：batch size
- `h`：num_heads
- `q`：query length
- `k`：key length
- `d`：head_dim

这里的意思是：在 `d` 这个维度上做乘法并求和，输出 `(b, h, q, k)`，也就是 attention score。

`einsum()` 的优点是表达力很强，读维度关系很直接；缺点是刚开始不熟的时候容易写错。

可以先记住它的使用场景：

- 维度很多，`permute()` + `matmul()` 写起来不直观
- 想直接把“哪个维度保留，哪个维度求和”写清楚

### 5.8 Transformer 里常见的几个操作

#### `softmax()`

`softmax()` 通常用于把一组分数归一化成概率分布。

如果输入为向量 $x = (x_1, x_2, \dots, x_n)$，其中 $x_i$ 表示第 $i$ 个位置的原始分数，那么 softmax 定义为：

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

在 PyTorch 里经常写成：

```python
probs = torch.softmax(logits, dim=-1)
```

`dim=-1` 表示在最后一个维度上做归一化。在 LLM 里，这通常意味着：

- 对词表维度做 softmax，得到下一个 token 的概率
- 对 attention score 的 key 维度做 softmax，得到注意力权重

#### `masked_fill()`

`masked_fill()` 很适合和 mask 一起使用。它的含义是：把 mask 为真的那些位置，用某个值填掉。

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
mask = torch.tensor([[False, True], [False, False]])

print(x.masked_fill(mask, float("-inf")))
```

输出：

```python
tensor([[1., -inf],
        [3., 4.]])
```

在 attention 里，一个非常常见的写法是：

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = torch.softmax(scores, dim=-1)
```

原因是 softmax 之后：

- `-inf` 对应的位置概率会变成 0
- 这样就能把 padding token 或未来位置屏蔽掉

#### `torch.where()`

`torch.where(condition, a, b)` 可以理解成逐元素版的 if-else。

```python
x = torch.tensor([1, 2, 3, 4])
y = torch.where(x > 2, x, torch.zeros_like(x))

print(y)  # tensor([0, 0, 3, 4])
```

它在这些场景里经常出现：

- 根据条件选择不同值
- 构造 mask 后做条件替换
- 避免直接写 Python 循环

和 `masked_fill()` 相比：

- `masked_fill()` 更适合“把某些位置统一替换成同一个值”
- `where()` 更适合“满足条件时选 a，否则选 b”

## 6. PyTorch 与深度学习模型实现

前面更多是在整理“张量怎么操作”，这一节开始整理“模型代码到底是怎么组织起来的”。

读训练代码的时候，经常会同时看到下面这些概念：

- `nn.Parameter`
- `buffer`
- `requires_grad`
- `grad`
- `nn.Module`
- `state_dict()`

它们之间其实是有关联的。可以先用下面这个角度整体理解：

| 概念 | 是什么 | 会不会被优化器更新 | 会不会跟着模型一起搬到 GPU | 会不会进入 `state_dict()` |
| --- | --- | --- | --- | --- |
| 普通 `Tensor` | 只是一个普通张量属性 | 默认不会 | 默认不会自动跟随 | 默认不会 |
| `nn.Parameter` | 被注册为模型参数的张量 | 会，如果 `requires_grad=True` | 会 | 会 |
| buffer | 被注册为模型状态的张量，但不是参数 | 不会 | 会 | 默认会，除非 `persistent=False` |
| `grad` | 参数在反向传播后得到的梯度 | 不是参数本身 | 跟着对应参数走 | 不单独保存 |

很多看起来“都是挂在 module 上的 tensor”，但语义完全不同：

- 如果它是需要训练的量，就应该注册成 `nn.Parameter`。
- 如果它是模型运行时需要保存的状态，但不参与训练，就适合注册成 buffer。
- 如果它只是临时变量，那就只是普通 tensor，不需要注册。

### 6.1 `nn.Module` 是什么

几乎所有模型都继承自 `nn.Module`，因为它不只是一个“写 forward 的类”，更重要的是它提供了一整套模型管理机制。

一个 `nn.Module` 主要负责：

- 注册参数
- 注册子模块
- 管理训练 / 推理模式
- 管理 device 迁移
- 导出和加载 `state_dict`

一个最简单的例子：

```python
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, x):
        return self.linear(x)
```

这里 `self.linear` 会自动被注册成子模块，而 `linear` 里面的权重和偏置又会自动被注册成参数。

所以当你调用：

```python
model.parameters()
model.to("cuda")
model.state_dict()
```

这些操作都会递归地作用到整个模型树上。这也是为什么 `nn.Module` 是 PyTorch 模型组织的核心。

### 6.2 `nn.Parameter` 与参数注册

`nn.Parameter` 的作用可以简单概括成一句话：把一个张量明确标记为“这是模型参数”。

例如：

```python
self.weight = nn.Parameter(torch.ones(d))
```

这行代码的含义不是“创建了一个 tensor”这么简单，而是：

- 这个张量会被 `nn.Module` 识别为参数。
- 它会出现在 `model.parameters()` 里。
- 优化器会默认看到它。
- 它会随着 `model.to(device)` 一起迁移。
- 它会进入 `state_dict()`。

如果你只是这样写：

```python
self.weight = torch.ones(d)
```

那它只是一个普通 tensor 属性，不会自动被注册成参数。

一个最小例子：

```python
class MyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * self.weight
```

这里的 `weight` 就是一个可学习参数。前面讲广播机制时提到的：

```python
self.weight * normalized_x
```

经常就是这种写法。

### 6.3 `requires_grad` 和 `grad`

`requires_grad` 表示：这个张量是否需要被 autograd 跟踪，并在反向传播时计算梯度。

例如：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
```

如果一个张量 `requires_grad=True`，并且它参与了计算图，那么在调用：

```python
loss.backward()
```

之后，它的梯度会出现在：

```python
x.grad
```

对于模型训练来说，最常见的流程是：

```python
optimizer.zero_grad()
loss = model(x)
loss.backward()
optimizer.step()
```

这里的逻辑是：

1. `forward()` 算出 loss。
2. `backward()` 沿着计算图反向传播，给各个参数算出梯度。
3. 梯度会存到参数的 `.grad` 属性里。
4. `optimizer.step()` 读取这些梯度，更新参数。

要注意：

- `.grad` 是梯度，不是参数本身。
- 梯度默认会累积，所以每轮训练前通常都要 `zero_grad()`。

对于大多数 `nn.Parameter` 来说，默认就是：

```python
requires_grad = True
```

但也可以手动冻结参数：

```python
param.requires_grad = False
```

这在冻结 embedding、冻结 backbone、只训练 LoRA 层时很常见。

### 6.4 `register_buffer()` 与 buffer

有些张量是模型运行时的一部分，但并不是要训练的参数。这类东西就很适合注册成 buffer。

你的这段例子就非常典型：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.hidden_size // config.num_attention_heads,
    end=config.max_position_embeddings,
    rope_base=config.rope_theta,
    rope_scaling=config.rope_scaling
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

这里的 `freqs_cos` 和 `freqs_sin` 是预先计算好的 RoPE 频率表。它们：

- 是模型运行时需要用到的状态。
- 应该跟着模型一起搬到 GPU。
- 但它们不是要优化的参数，不应该交给优化器更新。

这正是 buffer 的典型使用场景。

可以这样理解 `register_buffer()`：

- 它把一个 tensor 注册为“模型状态的一部分”。
- 它会跟着 `model.to(device)` 一起迁移。
- 它默认会出现在 `state_dict()` 里。
- 但它不是参数，不会出现在 `model.parameters()` 里。

所以你原来的理解可以整理成：

- `nn.Parameter`：把向量保存成参数。
- `register_buffer`：模型的一部分，一起加载进 GPU，不用手动 `to(device)`，不过不是参数，不进行优化。

这里还有一个细节：

```python
persistent=False
```

这表示这个 buffer 不进入 `state_dict()`。

也就是说：

- 它仍然是 buffer。
- 它仍然会随着模型一起迁移 device。
- 但在保存模型权重时不会被保存下来。

这很适合那种“可以根据配置重新计算出来”的量，比如某些预计算表、缓存或辅助常量。

### 6.5 `state_dict()`：模型到底保存了什么

`state_dict()` 可以理解成：模型当前状态的一个字典表示。

通常它包含：

- 所有参数
- 所有 persistent buffer

例如：

```python
model.state_dict().keys()
```

通常会看到：

- `embedding.weight`
- `layers.0.attn.q_proj.weight`
- `norm.weight`
- 某些 buffer 名字

需要注意：

- 普通 tensor 属性默认不会进 `state_dict()`。
- `nn.Parameter` 会进。
- `register_buffer(..., persistent=True)` 注册的 buffer 会进。
- `persistent=False` 的 buffer 不会进。

所以从“是否保存模型状态”这个角度，也可以反过来理解 parameter 和 buffer 的语义。

### 6.6 `nn.ModuleList` vs `nn.Sequential`

这两个东西本质上都在做一件事：注册并管理子模块。

但它们的使用场景不一样。

| 特性 | `nn.ModuleList` | `nn.Sequential` |
| --- | --- | --- |
| 是否自动 forward | 否，需要手动执行 | 是，自动顺序执行 |
| 是否注册参数 | 是 | 是 |
| 是否支持灵活逻辑 | 是，适合复杂结构 | 否，更适合纯顺序结构 |
| 典型应用 | Transformer、ResNet 等 | 简单 MLP、CNN 堆叠结构 |

可以把它们理解成：

- `nn.Sequential`：不仅帮你注册模块，还默认把这些模块按顺序连起来。
- `nn.ModuleList`：只负责把模块收进来并注册好，真正怎么执行要你自己写。

例如：

```python
self.layers = nn.ModuleList([
    Block(config) for _ in range(config.num_hidden_layers)
])

for layer in self.layers:
    x = layer(x)
```

这是 Transformer 里非常常见的写法。因为每一层之间往往不只是“无脑串起来”，中间可能还要插入：

- attention mask
- residual
- cache
- 条件分支

这时候 `ModuleList` 就比 `Sequential` 灵活得多。

### 6.7 `train()` 和 `eval()`

`nn.Module` 还有一个很重要但很容易被忽略的机制，就是训练模式和推理模式。

```python
model.train()
model.eval()
```

它们不会直接关闭梯度，也不会直接更新参数，而是告诉某些模块当前应该采用哪种行为。

最典型的例子有：

- `Dropout`
- `BatchNorm`

训练时和推理时它们的行为不同，所以模型在验证和推理前通常都要显式切到：

```python
model.eval()
```

如果只是想关闭梯度计算，通常用的是：

```python
with torch.no_grad():
    ...
```

这两个概念不要混在一起。

### 6.8 Autograd 图、叶子节点和 `detach()`

前面已经讲了 `requires_grad` 和 `.grad`，这里再补一层更接近底层的理解。

PyTorch 的 autograd 本质上是在记录一张计算图。只要一个张量：

- `requires_grad=True`
- 并且参与了后续计算

那么 PyTorch 就会把这些操作串成一张图，等你调用：

```python
loss.backward()
```

时再沿图反向传播。

这里一个常见概念是叶子节点。

粗略理解：

- 用户直接创建、并且 `requires_grad=True` 的参数，通常是叶子节点
- 中间计算结果通常不是叶子节点

例如：

```python
w = torch.tensor([2.0], requires_grad=True)
y = w * 3
z = y.sum()
```

这里：

- `w` 是叶子节点
- `y`、`z` 是中间结果

通常只有叶子节点会默认保留 `.grad`。

#### `detach()`

`detach()` 的作用是：返回一个和原张量共享数据、但不再参与当前计算图的新张量。

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()
```

这里 `z` 的数据和 `y` 一样，但 autograd 不会继续追踪 `z` 后面的操作。

常见用途：

- 不希望某段路径继续反向传播
- 记录中间结果但不保留梯度链路
- 构造 target、cache 或某些分析输出

#### `torch.no_grad()`

如果你想在一整段代码里都关闭梯度计算，通常用：

```python
with torch.no_grad():
    y = model(x)
```

这和 `detach()` 的区别是：

- `detach()` 是针对某个张量切断图
- `torch.no_grad()` 是在一个上下文里整体不记录计算图

推理和验证阶段非常常用 `no_grad()`，因为这样可以节省显存和计算开销。

### 6.9 CUDA、`device` 和 `dtype` 管理

训练代码里另一个非常常见的问题不是“公式错了”，而是“张量不在同一个 device / dtype 上”。

#### `device`

每个 tensor 都有自己的 device，比如：

```python
x.device
```

常见值有：

- `cpu`
- `cuda:0`

把张量或模型移动到某个 device：

```python
x = x.to("cuda")
model = model.to("cuda")
```

一个非常重要的原则是：

- 参与同一次运算的张量，通常必须在同一个 device 上。

例如下面这种情况就会报错：

```python
x = torch.randn(2, 3, device="cuda")
y = torch.randn(2, 3, device="cpu")
z = x + y
```

#### 为什么 parameter 和 buffer 很重要

这也能反过来解释，为什么 `nn.Parameter` 和 buffer 要注册到 `nn.Module` 里。

因为一旦它们被注册了，下面这种操作：

```python
model.to("cuda")
```

就会自动把：

- 参数
- buffer

一起迁移过去。否则很多时候你就得手动管理每个 tensor 的 device。

#### `dtype`

`dtype` 表示张量的数据类型，例如：

- `torch.float32`
- `torch.float16`
- `torch.bfloat16`
- `torch.int64`

查看类型：

```python
x.dtype
```

转换类型：

```python
x = x.to(torch.float16)
x = x.float()
x = x.long()
```

在 LLM 里经常会同时碰到：

- token id：通常是整数类型，比如 `torch.long`
- 激活值和参数：通常是浮点类型，比如 `float32`、`float16`、`bfloat16`

这也是为什么 embedding 的输入通常要是整数索引，而不能直接拿 float 去喂。

#### 一个常见写法

训练代码里经常会看到：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

如果还要统一 dtype，也可能写成：

```python
model = model.to(device=device, dtype=torch.bfloat16)
```

#### 常见报错来源

如果你在读 MiniMind 或自己写代码时看到下面这种错误，通常就该先检查 `device` 和 `dtype`：

- expected all tensors to be on the same device
- expected scalar type Float but found Half
- expected Long but got Float

先查这几个属性往往比盯着公式更有效：

```python
x.shape
x.device
x.dtype
```

## 7. 一些容易混淆的点

### 7.1 `cat` 和 `stack` 的区别

- `cat`：沿已有维度拼接，不增加新维度。
- `stack`：先增加一个新维度，再拼接。

### 7.2 `view` 和 `reshape` 的区别

- `view`：要求连续内存，不复制数据。
- `reshape`：更宽松，必要时可能复制数据。

### 7.3 `expand` 和 `repeat` 的区别

- `expand`：不复制数据，只做广播视图。
- `repeat`：真实复制数据。

### 7.4 为什么有时候 `view()` 会报错

通常是因为前面做了 `transpose()`、`permute()`、`expand()` 之类的操作，得到的张量不是 contiguous tensor。

这时一般有两种处理方式：

```python
x = x.contiguous().view(...)
```

或者：

```python
x = x.reshape(...)
```
<!-- 
## 8. 后续可补充的方向

- `nn.ModuleDict`、hook、参数初始化
- `Embedding`、`Linear`、`LayerNorm` 这些常见模块本身的输入输出 shape
- mixed precision、`autocast` 和 gradient scaler
- KV cache、causal mask、attention mask 的实现细节 -->
