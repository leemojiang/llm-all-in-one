"""
训练工具函数集合
"""

import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from minimind_learning.model.config_minimind import MiniMindConfig
from argparse import Namespace
import json
from datetime import datetime

def is_main_process():
    #是分布式节点的主进程!
    return not dist.is_initialized() or dist.get_rank() == 0 

def Logger(content):
    if is_main_process():
        print(content)

def setup_seed(seed: int):
    """Set all seeds for reproducibility

    Args:
        seed (int): The seed value to use
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # CUDNN
    torch.backends.cudnn.benchmark = False  # CUDNN

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

def lm_checkpoint(lm_config:MiniMindConfig, weight:str='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    '''
       weight: 保存权重前缀名 
    '''
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

def save_config_to_json(save_dir:str,args:Namespace,config:MiniMindConfig):
    """
    将 argparse 参数和 PretrainedConfig 配置一起保存为 JSON 文件。

    参数:
        save_dir (str): 保存 JSON 的路径（例如 './output'）
        args (Namespace): argparse 解析得到的参数对象
        config (PretrainedConfig): 模型配置对象（如 MiniMindConfig）
    """
    # 将 argparse.Namespace 转为字典
    args_dict = vars(args)

    # 将 config 转为字典（PretrainedConfig 提供 .to_dict() 方法）
    config_dict = config.to_dict()

    # 合并两个字典
    combined = {
        "argparse_args": args_dict,
        "pretraine d_config": config_dict
    }

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,f"config_{args.save_weight}_{args.hidden_size}_{datetime.now().strftime("%Y-%m-%d %H-%M")}.json")
    # 保存为 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    Logger(f"参数和配置已保存到: {save_path}")

def init_model(lm_config, from_weight='pretrain', tokenizer_path='../tokenizer', save_dir='./out', device='cuda'):
    from transformers import AutoTokenizer
    from minimind_learning.model.model_minimind import MiniMindForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        Logger(f'从 {weight_path} 加载模型权重完成。')
    
    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer

def get_lr(current_step, total_steps, lr):
    #余弦退火学习率调度器
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

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
