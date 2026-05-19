"""
训练工具函数集合
"""

import json
import math
import os
import random
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler

from minimind_learning.model.config_minimind import MiniMindConfig


def is_main_process():
    # 是否为分布式节点的主进程
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


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


def setup_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed_mode():
    # 检查是否设置了 RANK 环境变量（用于判断是否启用分布式训练）
    # 如果未设置，则说明当前不是分布式模式，直接返回 0（默认 GPU 编号）
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式，使用单卡训练
    # 初始化分布式通信组，使用 NCCL 后端（适用于 GPU）
    dist.init_process_group(backend="nccl")
    # 获取当前进程的本地 GPU 编号（LOCAL_RANK 是由启动器如 torchrun 设置的）
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置当前进程使用的 GPU（确保每个进程绑定到正确的设备）
    torch.cuda.set_device(local_rank)
    # 返回当前进程的 GPU 编号，供后续模型或数据加载器使用
    return local_rank


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def lm_checkpoint(
    lm_config: MiniMindConfig,
    weight: str = "full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    tokens_seen=0,
    wandb=None,
    save_dir="../checkpoints",
    **kwargs,
):
    """
    weight: 保存权重前缀名
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        raw_model = unwrap_model(model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "step": step,
            "tokens_seen": tokens_seen,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    resume_data[key] = unwrap_model(value).state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}->{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def save_config_to_json(save_dir: str, args: Namespace, config: MiniMindConfig):
    """
    将 argparse 参数和 PretrainedConfig 配置一起保存为 JSON 文件。

    参数:
        save_dir (str): 保存 JSON 的路径（例如 './output'）
        args (Namespace): argparse 解析得到的参数对象
        config (PretrainedConfig): 模型配置对象（如 MiniMindConfig）
    """
    args_dict = vars(args)
    config_dict = config.to_dict()
    combined = {
        "argparse_args": args_dict,
        "pretrained_config": config_dict,
    }

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"config_{args.save_weight}_{args.hidden_size}_{datetime.now().strftime('%Y-%m-%d %H-%M')}.json",
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    Logger(f"参数和配置已保存到 {save_path}")


def init_model(lm_config, from_weight="pretrain", tokenizer_path="../tokenizer", save_dir="../out", device="cuda"):
    from transformers import AutoTokenizer

    from minimind_learning.model.model_minimind import MiniMindForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        Logger(f"从 {weight_path} 加载模型权重完成。")

    get_model_params(model, lm_config)
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device), tokenizer


def get_lr(current_step, total_steps, lr):
    # 余弦退火学习率调度器
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
