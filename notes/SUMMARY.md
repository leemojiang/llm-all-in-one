# Summary

- [首页](index.md)

# Quick Start
- [环境配置 & 项目结构](quickstart/0.GettingStarted.md)

# Pretrain - Model
- [Pretrain 导言](pretrain/index.md)
- [背景: 从 Language Model 到 Transformer](pretrain/1.background.md)
- [历史脉络: 从 NLP 到 LLM, 再到 Agent](pretrain/1b.history.md)
- [Tokenizer: 文本如何变成 token id?](pretrain/1a.tokenizer.md)
- [理解 Attention 机制](pretrain/3.attention.md)
- [模型总览: 一个预训练模型由哪些部分组成?](pretrain/2.model-overview.md)
    - [RoPE: 位置编码如何进入 Attention?](pretrain/2a.rope.md)
    - [FlashAttention: attention 为什么还能更快?](pretrain/2b.flash-attention.md)
    - [GQA: 为什么 Query head 和 KV head 可以不一样?](pretrain/2c.gqa.md)
    - [KV Cache: 自回归推理为什么能避免重复计算?](pretrain/2d.kv-cache.md)
    - [Long Context: 长上下文能力通常在改什么?](pretrain/2e.long-context.md)


# Pretrain - Training
- [MiniMind Pretrain 实践入口](pretrain/4.practice.md)
<!-- # 训练篇

- [Pretrain](training/pretrain.md)
- [SFT](training/sft.md)
- [DPO](training/dpo.md)

# 专题篇

- [Tokenizer](topics/tokenizer.md)
- [Transformer](topics/transformer.md)
- [训练技巧](topics/training-tricks.md) -->

# 附录
- [Pytorch Cookbook](appendix/pytorch.md)
