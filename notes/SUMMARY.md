# Summary

- [首页](index.md)

# Quick Start
- [环境配置 & 项目结构](quickstart/0.GettingStarted.md)

# Pretrain - Model
- [Pretrain 导言](model-basic/index.md)
- [背景: 从 Language Model 到 Transformer](model-basic/1.background.md)
- [历史脉络: 从 NLP 到 LLM, 再到 Agent](model-basic/1b.history.md)
- [Tokenizer: 文本如何变成 token id?](model-basic/1a.tokenizer.md)
- [理解 Attention 机制](model-basic/3.attention.md)
- [模型总览: 一个预训练模型由哪些部分组成?](model-basic/2.model-overview.md)

# Model - Extra
- [进阶部分导言](model-extra/index.md)
- [RoPE: 位置编码如何进入 Attention?](model-extra/2a.rope.md)
- [FlashAttention: attention 为什么还能更快?](model-extra/2b.flash-attention.md)
- [GQA: 为什么 Query head 和 KV head 可以不一样?](model-extra/2c.gqa.md)
- [KV Cache: 自回归推理为什么能避免重复计算?](model-extra/2d.kv-cache.md)
- [Long Context: 长上下文能力通常在改什么?](model-extra/2e.long-context.md)

# Pretrain - Training
- [Pretrain 训练导言](pretrain/training-index.md)
- [Pretrain 的训练目标和 Loss](pretrain/1.loss.md)
- [优化器、学习率和数据设置](pretrain/2.optimizer-lr-data.md)
- [Pretrain 的实施细节和常见坑](pretrain/3.training-details.md)
- [Pretrain 的 Eval 指标](pretrain/4.eval.md)
- [MiniMind Pretrain 实践入口](pretrain/1.Pretrain.md)
- [MiniMind Pretrain 实践总结](pretrain/1.Pretrain2.md)

<!-- # 训练篇

- [Pretrain](training/pretrain.md)
- [SFT](training/sft.md)
- [DPO](training/dpo.md)

# 专题篇

- [Tokenizer](topics/tokenizer.md)
- [Transformer](topics/transformer.md)
- [训练技巧](topics/training-tricks.md) -->

# Utils
- [MiniMind 3 升级内容](utils/minimind3_upgrade.md)

# 附录
- [Pytorch Cookbook](appendix/pytorch.md)

# 参考
- [参考资料与引用](ref/index.md)
