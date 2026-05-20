import math

from transformers import PretrainedConfig


# Model Config
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind_reimplement"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,  # begin of sentence
        eos_token_id: int = 2,  # end of sentence
        hidden_act: str = "silu",
        hidden_size: int = 768,  # !
        intermediate_size: int = None,  # FFN intermediate size
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,  # !
        num_hidden_layers: int = 8,  # !
        num_key_value_heads: int = 4,  # !
        head_dim: int = None,  # 每个 attention head 的维度
        vocab_size: int = 6400,  # !
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        tie_word_embeddings: bool = True,
        ####################################################
        # Here are the specific configurations of MoE.
        # When use_moe is False, the following is invalid.
        ####################################################
        use_moe: bool = False,
        num_experts: int = 4,  # 总专家数量
        num_experts_per_tok: int = 1,  # 每个 token 选择的专家数量
        moe_intermediate_size: int = None,  # 每个专家内部 FFN 的中间层维度
        norm_topk_prob: bool = True,  # 是否标准化 top-k 概率
        router_aux_loss_coef: float = 5e-4,  # router 辅助损失系数
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        # 上游新版默认使用 hidden_size * pi，并对齐到 64 的倍数。
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn
        self.tie_word_embeddings = tie_word_embeddings

        ####################################################
        # Here are the specific configurations of MoE.
        # When use_moe is False, the following is invalid.
        ####################################################
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
