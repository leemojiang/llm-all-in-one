import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .config_minimind import MiniMindConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        dim: embedding dim
        weights: [dim,]
        """
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # [batch_size, seq_len, dim] * [batchsize, seq_len , 1]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        x [batch_size, seq_len, dim]
        return [barch_size,seq_len,dim]
        """
        return self.weights.type_as(x) * self._norm(x.float()).type_as(
            x
        )  # 处理不同的类型转化很重要


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        mean = torch.mean(x, -1, keepdim=True)  # [batchsize, seq_len , 1]
        var = (x - mean).pow(2).mean(-1, keepdim=True)  # [batchsize, seq_len , dim]
        inv_std = torch.rsqrt(var + self.eps)  # [batchsize, seq_len , dim]
        return (x - mean) * inv_std

    def forward(self, x):
        """
        x [batch_size, seq_len, dim]
        return [barch_size,seq_len,dim]
        """
        norm_x = self._norm(x.float()).type_as(x)
        return self.beta.type_as(x) + self.gamma.type_as(x) * norm_x


class FeedForward(nn.Module):
    """
    GLU Gate Linear Unit的变体
    From LLaMA 系列
    LLaMA2 首次引入这种结构作为默认 FFN
    Meta 的论文中称之为 Gated Linear Units with SiLU activation

    更强的非线性建模能力:门控乘法能动态调节信息流
    更好的训练稳定性:SiLU 激活 + 无 bias + 64 对齐
    更高的参数利用率:相比单路径 FFN，双路径乘法更充分利用中间维度
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 64 padding!
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size,seq_length,hidden_dim]
            hidden_states already applied  post_layernorm
        Retuen: [batch_size,seq_length,hidden_dim]
        """
        middle = self.up_proj(x) * self.act_fn(
            self.gate_proj(x)
        )  # [...,hidden_dim] -> [..., intermediate_dim]

        return self.dropout(
            self.down_proj(middle)
        )  # [...,hidden_dim] -> [..., intermediate_dim]


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.num_q_heads = config.num_attention_heads
        assert (
            self.num_q_heads % self.num_key_value_heads == 0
        )  # # attention_head is # q_head

        # 这个命名是为了适配可能有些算法里head数目的动态调整 主要用在forward里面
        self.n_local_heads = config.num_attention_heads  # Q heads num
        self.n_local_kv_heads = self.num_key_value_heads  # KV heads num
        # 参数结果
        self.n_rep = (
            self.n_local_heads // self.n_local_kv_heads
        )  # repeat 每个q_head 需要几个kv_head
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 模型参数
        # 4个W投影矩阵 实际合并在一起
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_q_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 两个配置参数
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        """
        x: Embedding_tensor
            [batch_size, seq_length, hidden_dim]

        position_embeddings: (freqs_cos,freq_sin)
            [seq_length,head_dim]

        attention_mask:
            [batch_size, seq_len]  1 for NO AFFECT 0 for PADDING

        past_key_value: (Past_K_tensor,Past_V_tensor)
            Shape: [bsc, seq_length, #kv_heads , head_dim]

        use_cache: Bool

        Return: (Embedding_tensor,(Past_K_tensor,Past_V_tensor))
             [batch_size, seq_length, hidden_dim] , ([bsc, seq_length+1, #kv_heads , head_dim],[bsc, seq_length+1, #kv_heads , head_dim])

        """
        bsz, seq_len, _ = x.shape
        # [...,# q_heads * head_dim] [...,# kv_heads * head_dim] [...,# kv_heads * head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # [seq_length,head_dim],[seq_length,head_dim]
        cos, sin = position_embeddings
        # No RoPE on V
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # xq xk xv
        # [bsc, seq_length, #q_heads , head_dim]

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # xq xk xv 标准的Attention输入
        # [bsc, #q_heads, seq_length , head_dim]

        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )

            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            # [batch_size, num_heads, seq_len_q, head_dim]
        else:
            # 手写Attention计算实现:
            # Q @ K^T / sqrt(d)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )  # [bsc, #q_heads, seq_length_q , seq_length_kv]

            # Add Causal Mask:
            # Mask shape [1,1,seq_length_q,seq_length_kv]
            # [0., -inf, -inf],
            # [0., 0., -inf],
            # [0., 0., 0.]
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            # Add padding Mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    2
                )  # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len] 0 Padding
                extended_attention_mask = (
                    1.0 - extended_attention_mask
                ) * -1e9  # -inf for Padding 0 no effect
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(
                xq
            )  # [batch_size, num_heads, seq_len_q, seq_len_k] 数值变成softmax 权重
            scores = self.attn_dropout(scores)
            #  scores: [bsc, #num_heads, seq_len_q , seq_len_k]
            #  xv    : [bsc, #num_heads, seq_len_k, head_dim]
            output = scores @ xv  # -> [bsc, #num_heads, seq_len_q, head_dim]

        # Reshape for output
        output = output.transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q ,num_heads,head_dim]
        output = output.reshape(
            bsz, seq_len, -1
        )  # ->  [batch_size, seq_len_q ,num_heads * head_dim]
        output = self.resid_dropout(
            self.o_proj(output)
        )  # -> [batch_size, seq_len_q ,hidden_dim]
        return output, past_kv


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [bs,slen,num_key_value_heads,head_dim]
    return : [bs,slen,num_key_value_heads * n_rep ,head_dim]

    Equal to:
        torch.repeat_interleave(x, dim=2, repeats=n_rep)
        torch.repeat([1,1,n_rep,1])

    底下这复杂的一大部分操作 是为了使用Expand来降低内存复制

    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    q k : [batch_size, seq_len, head_dim]
    cos sin : [seq_length, head_dim]
    """

    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    dim: hidden_size
    end: max_seq_length
    rope_base: w = 1 / (rope_base)^(2*i/dim)  i as index in embedding
    rope_scaling: Yarn Config dict

    Return: Tuple of (freqs_cos,freqs_sin) of cos(theta),sin(theta) index with p of position and i of dim
        freqs_cos: [max_seq_length, dim]
        freqs_sin: [max_seq_length, dim]
    """
    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [dim//2,]

    # YaRN 长度外推算法 (推理和计算的时候都会用到)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )
        if end / orig_max > 1.0:
            corr_dim = next(
                (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max),
                dim // 2,
            )
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
                dim // 2 - 1, 1
            )
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )
            freqs = freqs * scale

    t = torch.arange(
        end, device=freqs.device
    )  # 表示位置索引 [0, 1, 2, ..., max_seq_len-1]
    # freqs: 表示每个维度的频率 shape [dim//2]
    freqs = torch.outer(t, freqs).float()  # shape: [max_seq_len, dim//2]
    # 最后一个维度 拼在一起
    freqs_cos = torch.cat(
        [torch.cos(freqs), torch.cos(freqs)], dim=-1
    )  # [max_seq_len, dim]
    freqs_sin = torch.cat(
        [torch.sin(freqs), torch.sin(freqs)], dim=-1
    )  # [max_seq_len, dim]
    return freqs_cos, freqs_sin


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        assert not config.use_moe, "Moe not implemented "
        # self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        """
        hidden_states: torch.Tensor [batch_size, seq_length, hidden_dim]
            input embeddings.
        position_embeddings: (freqs_cos,freq_sin) [seq_length,hidden_dim]

        attention_mask:

        past_key_value:
            kv cache
        use_cache:
            use kv cache

        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class MiniMind_Dense(torch.nn.Module):
    """
    Dense模型的定义
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        # Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )  # [vocab_size , embedding_size]
        # Dropout and norm
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE vector
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size
            // config.num_attention_heads,  # dim for each attention heads
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # Attention Layers
        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        input_ids : [batch_size , seq_len]
            Tensor of token indices.
        attention_mask:
            Tensor with shape [batch_size, seq_len] or [batch_size, 1, 1, seq_len].
        past_key_values:  List[Tuple[key, value]] len of [self.layers]
            Cached key and value tensors from previous forward passes.
            Used for efficient autoregressive decoding.
            Each tuple corresponds to one Transformer layer: (past_key, past_value),
                - key: [batch_size, num_heads, past_seq_len, head_dim]
                - value: [batch_size, num_heads, past_seq_len, head_dim]
        use_cache :
            If True, the model will return updated `past_key_values` for caching.
            Useful during generation to avoid recomputing attention for previous tokens.

        Returns:
        --------
        hidden_states : torch.Tensor [batch_size, seq_len, hidden_dim]
            Final hidden representations of shape [batch_size, seq_len, hidden_dim].
            Used for downstream decoding or output projection.

        presents : List[Tuple[torch.Tensor, torch.Tensor]]
            Cached key/value tensors from each Transformer layer.
            Used for efficient autoregressive decoding in subsequent steps.

        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding
        # [batch_size , seq_len] -> [batch_size , seq_len , hidden_size] -> [batch_size , seq_len , hidden_size]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # [seq_len,]
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        # Transforms
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            layer: MiniMindBlock
            past_key_value: Tuple[torch.Tensor, torch.Tensor]

            # [batch_size , seq_len , hidden_size] -> [batch_size , seq_len , hidden_size]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # Norm
        # [batch_size , seq_len , hidden_size] -> [batch_size , seq_len , hidden_size]
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents , None

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMind_Dense(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        '''
            input_ids: [batch_size,seq_length]
                输入的 token ID 序列，通常来自 tokenizer.encode()
            attention_mask: [batch_size, seq_len]
                用于屏蔽 padding 或控制注意力范围，1 表示有效位置，0 表示忽略
            past_key_values:
                List of tuples，每层一个 (key, value)
                每个 key/value 的 shape: (batch_size, num_heads, past_seq_len, head_dim)
                描述: 用于增量推理（缓存历史注意力），加速 autoregressive 生成
            logits_to_keep:
                int 或 
                1D-tensor 
                    LongTensor: [token_to_keep]
                    BoolTensor: [seq_len]
                描述: 用于 top-k 或 mask 策略，控制哪些 logits 被保留（可选）0表示保留整个sequence

            Return:
                CausalLMOutputWithPast
        '''

        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        ) # h: [batch_size,seq_length,hidden_dim]

        # 从序列的倒数第 logits_to_keep 个位置开始
        # 一直到末尾（None 表示默认到结尾）, step 默认是 1
        # 对seq_length维度进行切片 表示保存最后几个token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        logits = self.lm_head(h[:, slice_indices, :]) # [batch_size, token_to_keep, hidden_dim] -> [batch_size, token_to_keep, vocab_size]
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
