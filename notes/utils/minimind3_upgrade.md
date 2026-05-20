# MiniMind 3 升级内容

这份笔记前半部分是参考25年9月份Minimind的实现,现在26年5月MiniMind又做了一些升级,在此我们对项目中的代码也进行了相应的更新,以适配Minimind项目最新的实现.

> Commit hash @9da8e1ab18ca0bbb3df9d8b5dbfbc8748810a5bf

如果用一句话概括，MiniMind 3 的升级方向大概是：

> 让模型结构更接近当前主流小型 LLM 的实现方式，让数据和训练接口更统一，同时让训练脚本在续训、分布式、混合精度和编译优化下更稳。

整篇文章将按照这3个角度进行梳理：
1. 模型/数据的改进
2. 训练流程的改进
3. 工程优化,细节上的升级.

## 1. 模型和数据做了哪些改进

### 1.1 Attention 中加入 Q/K RMSNorm

新版模型在 attention 的 query 和 key 上增加了 RMSNorm。这个变化不改变 attention 的基本形式，仍然是：

其中，\\(Q\\) 表示 query 矩阵，\\(K\\) 表示 key 矩阵，\\(V\\) 表示 value 矩阵，\\(d\\) 表示每个 attention head 的维度。标准 attention 可以写成：

\\[
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\\]

但是在真正计算 RoPE 和 attention score 之前，MiniMind 3 会先对 \\(Q\\) 和 \\(K\\) 做 RMSNorm。这样做的直观意义是：让参与点积的 query/key 向量尺度更稳定，从而让 attention score 不容易因为向量范数波动过大而变得过尖或者过散。

对应代码在 [src/minimind_learning/model/model_minimind.py](../../src/minimind_learning/model/model_minimind.py)：

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        # Q/K/V/O 投影矩阵。GQA 下 K/V head 数量可以少于 Q head 数量。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 新版 MiniMind 在 Q/K 上增加 RMSNorm，有助于稳定注意力分数。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

在 forward 里，Q/K Norm 发生在 RoPE 之前：

```python
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

xq, xk = self.q_norm(xq), self.k_norm(xk)
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

这说明新版结构更重视 attention 内部的数值稳定性。对于小模型来说，这类改动通常不会显著增加参数量，但会让训练过程更稳。

### 1.2 RoPE 外推改成新版 YaRN 形式

RoPE 的作用是把位置信息注入到 attention 的 query/key 中。MiniMind 3 中仍然保留 RoPE，但对长上下文外推的处理更接近新版 YaRN 写法。

如果不考虑外推，RoPE 中每个维度对应的频率可以理解为：

其中，\\(d\\) 表示 head 维度，\\(i\\) 表示频率维度索引，\\(\theta\\) 表示 `rope_theta`。频率为：

\\[
\omega_i = \frac{1}{\theta^{2i/d}}
\\]

对于位置 \\(p\\)，旋转角度可以写成：

\\[
\alpha_{p,i} = p \cdot \omega_i
\\]

MiniMind 3 的升级重点不在这个基础公式，而在于当目标上下文长度超过原始训练长度时，如何平滑地缩放不同频率的部分。代码里使用了一个 ramp，让低维和高维频率不是突然整体缩放，而是按区间逐步过渡。

对应代码在 [src/minimind_learning/model/model_minimind.py](../../src/minimind_learning/model/model_minimind.py)：

```python
if rope_scaling is not None:
    orig_max, factor, beta_fast, beta_slow, attn_factor = (
        rope_scaling.get("original_max_position_embeddings", 2048),
        rope_scaling.get("factor", 16),
        rope_scaling.get("beta_fast", 32.0),
        rope_scaling.get("beta_slow", 1.0),
        rope_scaling.get("attention_factor", 1.0),
    )
    if end / orig_max > 1.0:
        # YaRN: f'(i) = f(i) * ((1 - ramp) + ramp / factor)
        def inv_dim(beta):
            return (dim * math.log(orig_max / (beta * 2 * math.pi))) / (2 * math.log(rope_base))

        low = max(math.floor(inv_dim(beta_fast)), 0)
        high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
            0,
            1,
        )
        freqs = freqs * (1 - ramp + ramp / factor)
```

这里可以把 `ramp` 理解成一个从 0 到 1 的平滑权重。它不是简单地把所有频率都除以同一个 `factor`，而是让不同频段有不同程度的缩放。这样做的目的，是在扩展上下文长度时，尽量减少位置编码分布和原训练分布之间的突变。

### 1.3 引入 MoE FeedForward

旧版学习实现里虽然有 `use_moe` 配置，但模型主体中实际上不支持 MoE。MiniMind 3 中，FFN 可以从普通 dense FFN 切换成 MoE FFN。

普通 FFN 是所有 token 都走同一套参数：

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        middle = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(middle)
```

MoE FFN 则多了一层 router。router 会给每个 token 分配专家权重，然后只激活 top-k 个 expert：

```python
scores = F.softmax(self.gate(x_flat), dim=-1)
topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
if self.config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = topk_idx == i
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

从结构上看，MoE 的核心思想是：总参数量可以变大，但每个 token 实际激活的参数量不一定同步变大。比如有 4 个 expert，每个 token 只选择 1 个 expert，那么模型拥有更多“容量”，但单个 token 的计算路径仍然比较稀疏。

MoE 还有一个重要问题：如果 router 总是偏向某几个 expert，其他 expert 学不到东西。所以训练时会加入 router auxiliary loss：

```python
if self.training and self.config.router_aux_loss_coef > 0:
    load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
    self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
else:
    self.aux_loss = scores.new_zeros(1).squeeze()
```

这部分 loss 的作用不是直接预测下一个 token，而是约束 router 不要过度集中到少数 expert 上。

### 1.4 模型 forward 直接支持 labels

旧版训练代码中，Dataset 返回 `(X, Y, loss_mask)`，训练脚本手动计算交叉熵：

```python
loss = loss_fct(
    # B batch size, L sequence length, V vocab size
    # [B,L,V] -> [B*L,V]
    res.logits.view(-1, res.logits.size(-1)),
    Y.view(-1)
).view(Y.size())

loss = (loss * loss_mask).sum() / loss_mask.sum()
```

MiniMind 3 更接近 HuggingFace 模型的接口习惯：Dataset 返回 `input_ids` 和 `labels`，模型内部完成 shift 和 `ignore_index=-100` 的 loss 计算。

对应代码在 [src/minimind_learning/model/model_minimind.py](../../src/minimind_learning/model/model_minimind.py)：

```python
loss = None
if labels is not None:
    # shifted LM loss
    # B batch size, L sequence length, V vocab size
    # logits : [B,L,V] -> [B,L-1,V] 
    # labels: [B,L] -> [B,L-1]

    x = logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

这个变化看起来只是接口变化，但实际影响很大：pretrain、SFT、DPO 都可以围绕同一种模型输出结构来组织，训练脚本不再需要各自手写一套语言模型 loss。

### 1.5 数据格式从 loss_mask 转向 labels

对应地，数据集也从显式返回 `loss_mask`，转向直接构造 `labels`。

以 SFT 为例，用户问题、system prompt、工具描述等位置都不应该参与 loss；只有 assistant 回复部分参与训练。新版实现里，这些“不训练的位置”直接写成 `-100`：

```python
def generate_labels(self, input_ids: list):
    """
    仅 assistant 回复部分参与 loss 计算；其他位置设置为 -100。
    labels 和 input_ids 同长度，shift 由模型内部完成。
    """
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

这样做的好处是，mask 逻辑被压缩到数据构造阶段。训练阶段只需要把 `input_ids` 和 `labels` 交给模型：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
```

从学习角度看，这也更容易建立统一心智模型：`labels == -100` 的位置不参与语言模型 loss，其他位置正常预测下一个 token。

## 2. 训练部分引入了哪些改进

### 2.1 Pretrain 和 SFT 训练接口统一

新版 pretrain 和 full SFT 的训练循环非常接近，核心都是：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)

    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps
```

这说明 pretrain 和 SFT 在训练代码层面的差异被压缩了。它们真正的区别主要转移到了数据构造上：

- pretrain 数据：几乎所有非 pad token 都参与预测。
- SFT 数据：只有 assistant 回复部分参与预测。

这是一种更清晰的分层：模型只关心 `input_ids/labels`，训练循环只关心反向传播和优化器更新，任务差异由 Dataset 负责。

### 2.2 DPO loss 口径调整

DPO 的目标是让策略模型相对于参考模型，更偏好 chosen 而不是 rejected。

其中，\\(\pi_\theta\\) 表示当前训练的策略模型，\\(\pi_\mathrm{ref}\\) 表示冻结的参考模型，\\(y_w\\) 表示 chosen response，\\(y_l\\) 表示 rejected response，\\(x\\) 表示 prompt，\\(\beta\\) 是控制偏好强度的超参数。DPO loss 可以写成：

<script type="math/tex; mode=display">
\mathcal{L}_\mathrm{DPO}
=
- \log \sigma \left(
\beta
\left[
\log \frac{\pi_\theta(y_w \mid x)}{\pi_\theta(y_l \mid x)}
-
\log \frac{\pi_\mathrm{ref}(y_w \mid x)}{\pi_\mathrm{ref}(y_l \mid x)}
\right]
\right)
</script>

代码实现里，先对每个 token 的 log probability 按 mask 聚合：

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)。
    batch 前半部分为 chosen，后半部分为 rejected。
    """
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    # log-ratio 比较：策略模型 vs 参考模型。
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

这个版本更贴近 DPO 公式本身：比较的是 response 整体的 log probability 差异，而不是每个样本平均 token log probability 之后再比较。

### 2.3 MoE auxiliary loss 进入训练目标

因为模型可能启用 MoE，所以训练 loss 不再只有语言模型 loss 或 DPO loss，还要加上 `aux_loss`：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

DPO 中也是一样：

```python
dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
loss = dpo_loss_val + outputs.aux_loss
loss = loss / args.accumulation_steps
```

如果不开 MoE，`aux_loss` 基本就是 0，不影响普通 dense 模型。这样训练脚本可以同时兼容 dense 和 MoE，不需要为 MoE 单独写一套训练流程。

### 2.4 梯度累积处理更完整

梯度累积的目的是用多个小 batch 模拟一个更大的 batch。核心逻辑是：每个小 batch 都 backward，但不是每次都 `optimizer.step()`。

新版训练脚本除了正常的 accumulation step，还补上了 epoch 末尾“不足一个累积窗口”的情况：

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这个细节的意义是：如果一个 epoch 的 batch 数不能被 `accumulation_steps` 整除，最后剩下的梯度也不会被浪费掉。

### 2.5 增加 torch.compile 入口

训练脚本新增了 `--use_compile`，可以选择性启用 PyTorch 2.x 的 `torch.compile`：

```python
if args.use_compile == 1:
    model = torch.compile(model)
    Logger("torch.compile enabled")
```

这属于训练性能优化，不改变模型数学含义。它的作用是让 PyTorch 尝试对模型计算图进行编译优化，在某些环境下可以提升吞吐。不过它也可能增加首次编译开销，所以做实验时最好把它作为一个显式开关，而不是默认强制开启。

## 3. 工程上做了哪些优化

### 3.1 checkpoint 保存更稳

MiniMind 3 对 checkpoint 保存做了更稳的处理，尤其是兼容 DDP 和 `torch.compile`。

在 DDP 下，真正的模型包在 `model.module` 里；在 `torch.compile` 之后，原始模型可能包在 `_orig_mod` 里。如果保存时不 unwrap，可能会保存到包装器状态，或者出现权重 key 不符合预期的问题。

现在统一通过 `unwrap_model` 取原始模型：

```python
def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)
```

保存权重时也会转成 half 并放到 CPU：

```python
raw_model = unwrap_model(model)
state_dict = raw_model.state_dict()
state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
ckp_tmp = ckp_path + ".tmp"
torch.save(state_dict, ckp_tmp)
os.replace(ckp_tmp, ckp_path)
```

这里有两个工程意义：

1. `half().cpu()` 可以减少保存文件大小，也减少保存 checkpoint 时的显存压力。
2. 先保存到 `.tmp`，再用 `os.replace` 替换正式文件，可以降低写 checkpoint 中途失败导致文件损坏的风险。

### 3.2 resume 对 world size 变化更友好

分布式训练中，续训时 GPU 数量可能和上次不一样。比如上次 2 张卡，这次 1 张卡，如果直接用旧 step，跳过 batch 的数量可能不一致。

因此 checkpoint 加载时会记录并比较 world size：

```python
saved_ws = ckp_data.get("world_size", 1)
current_ws = dist.get_world_size() if dist.is_initialized() else 1
if saved_ws != current_ws:
    ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
    Logger(f'GPU数量变化({saved_ws}->{current_ws})，step已自动转换为{ckp_data["step"]}')
```

这不是严格意义上的训练算法改进，但对实际训练很重要。因为真实实验里，经常会遇到中途换机器、换卡数、恢复训练的情况。

### 3.3 模型命名统一为 MiniMindModel

这次 `src` 中的模型主干命名也做了整理。旧实现里主干模型叫 `MiniMind_Dense`，容易让人误以为它只能表示 dense 版本。但新版模型已经可以根据配置切换 dense FFN 或 MoE FFN，因此更合适的名字是 `MiniMindModel`。

现在结构是：

```python
class MiniMindModel(torch.nn.Module):
    """
    MiniMind 主干模型：Embedding -> Transformer Blocks -> RMSNorm。
    """
    ...


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
```

这个命名更符合常见 HuggingFace 风格：

- `MiniMindModel` 表示 backbone。
- `MiniMindForCausalLM` 表示在 backbone 上加了 `lm_head` 的语言模型。

为了避免旧笔记或旧代码引用断掉，代码中暂时保留了一个兼容别名：

```python
# 兼容旧笔记/旧代码中的命名。新代码请使用 MiniMindModel。
MiniMind_Dense = MiniMindModel
```

### 3.4 参数统计区分总参数和激活参数

MoE 模型有一个很容易误解的地方：总参数量和每个 token 实际激活的参数量不是一回事。

新版工具函数中增加了参数统计逻辑：

```python
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "num_experts", 0)
    n_active = getattr(config, "num_experts_per_tok", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    base = total - expert * n_routed
    active = base + expert * n_active
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")
```

这里的 `total` 是模型拥有的总参数量，`active` 是一个 token 前向传播时大致会激活的参数量。对于 MoE 模型来说，这两个数字都值得看：

- 总参数量反映模型容量。
- 激活参数量更接近单 token 的计算成本。

## 4. 小结

MiniMind 3 的升级可以理解为三条线同时推进。

第一条线是模型结构升级：Q/K RMSNorm、YaRN RoPE、MoE FFN，让模型更接近当前小型 LLM 的常见实现。

第二条线是训练接口升级：把 `labels` 交给模型内部计算 loss，pretrain/SFT/DPO 的训练循环因此更统一；MoE 的 auxiliary loss 也自然进入总训练目标。

第三条线是工程稳定性升级：checkpoint unwrap、CPU 半精度保存、resume world size 处理、`torch.compile` 开关和参数统计，都让这个项目更像一个可以反复实验的训练框架，而不只是一个最小 demo。

从学习角度看，这次升级最值得关注的不是某一个参数默认值变了，而是代码组织方式的变化：模型、数据、训练、保存这几层之间的边界更清晰了。
