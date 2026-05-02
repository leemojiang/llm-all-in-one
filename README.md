# LLM ALL in One: 从零开始构建大型语言模型

## MiniMind 学习笔记

这是一个围绕大语言模型学习过程持续整理的笔记项目。它最初源于我对 [MiniMind](https://github.com/jingyaogong/minimind) 的代码学习，后来逐渐扩展为一套以问题为线索、以实践为主线的 LLM 学习笔记。

仓库中保留了上游参考代码，也加入了我自己的代码阅读、实验记录与笔记整理。相比单纯摘录概念，我更希望把“模型原理”“训练流程”“工程实现”“实践经验”放在同一个语境中串联起来，形成一份可以持续扩展的个人知识库，也尽量让它对其他正在入门 LLM 的学习者有帮助。

[在线预览](https://leemojiang.github.io/llm-all-in-one/)

## 这个项目想解决什么问题

在学习 LLM 的过程中，我越来越强烈地感受到两个问题：

第一，LLM 相关知识点很多，而且彼此关联紧密。单独记录零散笔记虽然方便，但随着内容变多，很容易失去整体结构。

第二，很多内容只有在“真正写过、跑过、对照过代码”之后，才会从“看懂了”变成“理解了”。因此，这个项目并不是只想整理理论知识，而是想把学习过程中的问题、思考、代码实现和实验过程一起沉淀下来。

所以，这个仓库最终变成了一套用 `mdBook` 组织的学习笔记。你可以把它当作：

- 一份从实践出发整理的 LLM 学习地图
- 一套围绕问题展开的阅读提纲
- 一个可以继续补充、修正和扩展的笔记工程

## 如何阅读这份笔记

这套笔记整体上是按照**问题导向**的方式组织的。相比从头到尾线性阅读，我更推荐先浏览目录，再挑选自己当前最关心的问题进入相应章节，例如：

- 一个 LLM 系统整体由哪些部分组成？
- Tokenizer、Embedding、Attention 在代码里分别是什么？
- 预训练阶段到底在优化什么？
- 训练流程中的数据、学习率、Loss、Eval 是怎样衔接起来的？

这种组织方式并不追求把所有知识一次性讲完，而是希望把关键问题串起来，帮助读者先建立整体框架，再逐步深入细节。**在AI时代,好的问题可能比答案更重要**.

事实上我在创作过程中大量使用了Agent的辅助,对我来说创作过程就是一种用输出来驱动的学习过程.很多章节开始可能都是几个简单的问题开始,随着提问的深入,很多新的细节被发现,问题从主干延伸到细节,整个构建过程如同进行一次树搜索.因此,这也意味着这份笔记更接近“我的学习路径”而不是标准教材。它很适合作为**问题提纲**、**知识索引**和**实践参考**，但未必是每位读者都需要逐字阅读的内容。 就是以你完全可以把它作为出发点，再结合自己的理解重组出一份更适合自己的学习笔记。

## 内容特点与说明

- 笔记尽量保留较完整的推导、公式定义和代码细节，方便后续查阅与复用.
- 整个项目仍在持续更新中，目前发布的是第一阶段的整理成果

如果其中的内容对你有帮助，欢迎提出问题、反馈错误或分享建议。我会非常感谢，也会尽量继续完善这套笔记。

## 项目结构

- `minimind_upstream`：上游 MiniMind 代码，用作对照和参考
- `src`：当前仓库中的实验或扩展代码
- `tests`：测试代码
- `notes`：学习笔记，按 `mdBook` 方式组织
- `datasets`：数据集文件
- `notebooks`：实验用 Jupyter Notebook
- `book`：`mdBook` 构建后的静态页面

## 环境准备

```bash
# 同步项目依赖（CPU 环境）
uv sync --extra cpu

# 如果使用 CUDA 环境，也可以切换到 cuda 依赖
uv sync --extra cuda

# 同步 submodule
git submodule update --init --recursive

# 以开发模式安装当前项目
uv pip install -e .

# 登录 swanlab（如果需要使用实验记录）
swanlab login
```

## 本地阅读笔记

如果本地已经安装 `mdbook`，可以直接在仓库根目录运行：

```bash
mdbook serve
```

启动后即可在本地浏览完整笔记。

如果你希望通过 GitHub Pages 在线阅读，也可以直接访问部署后的文档站点：

- GitHub Pages: `https://leemojiang.github.io/llm-all-in-one/`

## 笔记入口

- [目录总览](notes/SUMMARY.md)
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

# 附录
- [Pytorch Cookbook](appendix/pytorch.md)

# 参考
- [参考资料与引用](ref/index.md)

## 引用

> **引用**：转载、引用或参考本项目内容时，请注明原作者和项目来源。

**Cited as:**

> LEE. (May 2026). LLM ALL in One: 从零开始构建大型语言模型.  
> https://github.com/leemojiang/llm-from-scratch

Or

```bibtex
@misc{lee2026llm_all_in_one,
  title        = {LLM ALL in One: 从零开始构建大型语言模型},
  author       = {LEE},
  year         = {2026},
  month        = may,
  howpublished = {\url{https://github.com/leemojiang/llm-from-scratch}},
  note         = {GitHub repository}
}
```

## 致谢

本项目的重要参考起点是 [MiniMind](https://github.com/jingyaogong/minimind) 项目。很多笔记内容都建立在对其代码、训练流程和实现细节的学习之上。

此外，项目中的一些学习路径、资料整理与问题设计，也参考了公开课程、技术文章与社区资料。相关内容已在笔记中的“参考资料与引用”章节中持续补充。
