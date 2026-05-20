import argparse
import json
from dataclasses import dataclass

import torch


def json_2_args(json_path: str) -> argparse.Namespace:
    with open(json_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    try:
        config_dict = config_dict["argparse_args"]
    except KeyError:
        pass
    return argparse.Namespace(**config_dict)


@dataclass(kw_only=True)
class TrainConfig:
    save_dir: str = "./out"  # 模型保存目录
    ckpt_dir: str = "./checkpoints"  # checkpoint保存目录
    save_weight: str = "pretrain"  # 保存权重的前缀名
    epochs: int = 2  # 训练轮数
    batch_size: int = 32  # batch size
    learning_rate: float = 5e-4  # 初始学习率
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # 训练设备
    dtype: str = "bfloat16"  # 混合精度类型
    num_workers: int = 8  # 数据加载线程数
    accumulation_steps: int = 8  # 梯度累积步数
    grad_clip: float = 1.0  # 梯度裁剪阈值
    log_interval: int = 100  # 日志打印间隔
    save_interval: int = 1000  # 模型保存间隔
    hidden_size: int = 768  # 隐藏层维度
    num_hidden_layers: int = 8  # 隐藏层数量
    max_seq_len: int = 340  # 最大序列长度
    use_moe: int = 0  # 是否使用MoE，0否1是
    data_path: str = "./dataset/pretrain_t2t_mini.jsonl"  # 数据路径
    from_weight: str = "none"  # 基于哪个权重训练
    from_resume: int = 0  # 是否自动检测续训，0否1是
    use_wandb: bool = False  # 是否使用wandb/swanlab
    wandb_project: str = "MiniMind-Pretrain"  # wandb项目名
    use_compile: int = 0  # 是否使用torch.compile，0否1是


def json_2_cfg(json_path: str) -> TrainConfig:
    with open(json_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    try:
        config_dict = config_dict["argparse_args"]
    except KeyError:
        pass
    return TrainConfig(**config_dict)
