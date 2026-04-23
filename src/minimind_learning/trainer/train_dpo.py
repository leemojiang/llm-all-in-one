import os
import argparse
import warnings
import time
from contextlib import nullcontext

import torch
from torch import optim, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minimind_learning.dataset.lm_dataset import DPODataset
from minimind_learning.model.config_minimind import MiniMindConfig


from minimind_learning.trainer.trainer_utils import init_distributed_mode, setup_seed, Logger,lm_checkpoint,save_config_to_json,init_model,SkipBatchSampler,is_main_process,get_lr
# from minimind_learning.model.model_lora import apply_lora,save_lora

warnings.filterwarnings('ignore')


class DPOTrainer():
    def __init__(self, args:argparse.Namespace):
        self.args = args

        # ========== 1. 初始化环境和随机种子 ==========
        local_rank = init_distributed_mode()
        if dist.is_initialized(): args.device = f"cuda:{local_rank}"
        setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

        # ========== 2. 配置目录、模型参数、检查ckp ==========
        os.makedirs(args.save_dir, exist_ok=True)
        lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.ckpt_dir) if args.from_resume==1 else None

        # 保存所有Config
        save_config_to_json(args.save_dir + "/configs",args,lm_config)
        if ckp_data : Logger("检测到续训检查点，训练将从检查点状态恢复。")

        # ========== 3. 设置混合精度 ==========
        device_type = "cuda" if "cuda" in args.device else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

        # ========== 4. 配wandb ==========
        wandb = None
        if args.use_wandb and is_main_process():
            import swanlab as wandb
            wandb_id = ckp_data.get('wandb_id') if ckp_data else None
            resume = 'must' if wandb_id else None
            wandb_run_name = f"MiniMind-DPO-{args.save_weight}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
            Logger(f"Wandb已连接，项目名: {args.wandb_project}, 运行名: {wandb_run_name}")

        
        # ========== 5. 定义模型和参考模型 ==========
        model, tokenizer = init_model(lm_config, args.from_weight, device=args.device , save_dir=args.save_dir)
        Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
        # 初始化参考模型（ref_model冻结）
        ref_model, _ = init_model(lm_config, args.from_weight, device=args.device , save_dir=args.save_dir)
        ref_model.eval()
        ref_model.requires_grad_(False)
        Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

        # ========== 6. 定义数据和优化器 ==========
        train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        # 这里最大的不同在于优化器只更新LoRA参数
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        # ========== 7. 从ckp恢复状态 ==========
        start_epoch, start_step = 0, 0
        if ckp_data:
            model.load_state_dict(ckp_data['model'])
            optimizer.load_state_dict(ckp_data['optimizer'])
            scaler.load_state_dict(ckp_data['scaler'])
            start_epoch = ckp_data['epoch']
            start_step = ckp_data.get('step', 0)
            Logger(f"从检查点恢复训练状态... start_epoch:{start_epoch}, start_step:{start_step}")

        # ========== 8. DDP包模型 ==========
        if dist.is_initialized():
            model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
            model = DistributedDataParallel(model, device_ids=[local_rank])

        # 参数表
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.scaler = scaler
        self.optimizer = optimizer
        self.autocast_ctx = autocast_ctx

        self.start_epoch = start_epoch
        self.start_step = start_step
        self.train_sampler = train_sampler
        self.wandb = wandb
        self.lm_config = lm_config


    def train(self):
        args = self.args
        train_ds = self.train_ds
        model = self.model
        ref_model = self.ref_model
        scaler = self.scaler
        optimizer = self.optimizer
        start_epoch = self.start_epoch
        start_step = self.start_step
        train_sampler = self.train_sampler
        wandb = self.wandb

        # ========== 9. 开始训练 ==========
        model.train() # 设置模型为训练模式(模型一般是默认train)
        for epoch in range(start_epoch, args.epochs):
            train_sampler and train_sampler.set_epoch(epoch)
            if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
                batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
                loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
                Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
                self.train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, start_step, wandb, args.beta)
            else: # 默认从头开始
                loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
                self.train_epoch(epoch, loader, len(loader), ref_model, 0, wandb,args.beta)
    
    
    def logits_to_log_probs(self,logits, labels):
        # logits shape: (batch_size, seq_len, vocab_size)
        # labels shape: (batch_size, seq_len)
        # log_probs shape: (batch_size, seq_len)
        log_probs = F.log_softmax(logits, dim=2)
        log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
        return log_probs_per_token

    def dpo_loss(self,ref_probs, probs, mask, beta):
        # ref_probs: (batch_size, seq_len) 来自参考模型（Reference Model）的 log-probs
        # probs: (batch_size, seq_len)     来自当前策略模型（Policy Model）的 log-probs
        # mask: (batch_size, seq_len)      用于标记哪些 token 被计入损失（如生成部分）
        # beta: float                      DPO 的超参数控制分布偏移强度

        # Step 1: 每个样本的有效长度（非 padding 部分 token 的数量）
        seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        # Step 2: 对每个样本计算平均 log-probs，仅在 mask == 1 的位置有效
        ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze(1)  # (batch_size,)
        probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze(1)          # (batch_size,)

        # Step 3: 将 batch 划分为前一半为 chosen，后一半为 rejected
        batch_size = ref_probs.shape[0]  # 假设 batch_size 是偶数，前半是 chosen，后半是 rejected

        chosen_ref_probs = ref_probs[:batch_size // 2]     # (batch_size // 2,)
        reject_ref_probs = ref_probs[batch_size // 2:]     # (batch_size // 2,)
        chosen_probs = probs[:batch_size // 2]             # (batch_size // 2,)
        reject_probs = probs[batch_size // 2:]             # (batch_size // 2,)

        # Step 4: log-ratio 比较（策略模型 vs 参考模型）
        pi_logratios = chosen_probs - reject_probs         # (batch_size // 2,)
        ref_logratios = chosen_ref_probs - reject_ref_probs  # (batch_size // 2,)

        # Step 5: DPO 损失计算，鼓励 chosen 比 rejected 的分数更高
        logits = pi_logratios - ref_logratios              # (batch_size // 2,)
        loss = -F.logsigmoid(beta * logits)                # (batch_size // 2,)

        return loss.mean()  # 标量，.mean()等价于DPO loss数学公式中的期望符号E

    # def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    #     # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    #     # https://github.com/jingyaogong/minimind/issues/298
    #     seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask导致除零NaN #[bach_size,1]
    #     ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    #     policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    #     # 将 chosen 和 rejected 数据分开
    #     batch_size = ref_log_probs.shape[0]
    #     chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    #     reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    #     chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    #     reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    #     pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    #     ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    #     logits = pi_logratios - ref_logratios
    #     loss = -F.logsigmoid(beta * logits)
    #     return loss.mean()

    def train_epoch(self, epoch, loader, iters, ref_model, start_step=0, wandb=None,beta=0.1):
        args = self.args
        optimizer = self.optimizer
        scaler = self.scaler
        autocast_ctx = self.autocast_ctx
        
        lm_config = self.lm_config
        model = self.model

        # loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        for step, batch in enumerate(loader, start=start_step + 1):
            x_chosen = batch['x_chosen'].to(args.device)
            x_rejected = batch['x_rejected'].to(args.device)
            y_chosen = batch['y_chosen'].to(args.device)
            y_rejected = batch['y_rejected'].to(args.device)
            mask_chosen = batch['mask_chosen'].to(args.device)
            mask_rejected = batch['mask_rejected'].to(args.device)
            x = torch.cat([x_chosen, x_rejected], dim=0) #[2*bsc,seq_len]
            y = torch.cat([y_chosen, y_rejected], dim=0) #[2*bsc,seq_len]
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)            

            lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with autocast_ctx:
                with torch.no_grad():
                    ref_outputs = ref_model(x)
                    ref_logits = ref_outputs.logits #[2*bsc,seq_len,vocab_size]
                    ref_log_probs = self.logits_to_log_probs(ref_logits, y) #[2*bsc,seq_len]

                outputs = model(x)
                logits = outputs.logits
                policy_log_probs = self.logits_to_log_probs(logits, y)

                loss = self.dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 or step == iters - 1:
                spend_time = time.time() - start_time
                current_loss = loss.item() * args.accumulation_steps
                current_lr = optimizer.param_groups[-1]['lr']
                eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
                
                Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
                
                if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

            if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
                model.eval()
                moe_suffix = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
                torch.save(state_dict, ckp)
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.ckpt_dir)
                model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="checkpoint保存目录")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=False, type=bool, help="是否使用MoE")
    parser.add_argument("--data_path", type=str, default="../datasets/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, help="是否自动检测&续训，0否1是")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    args = parser.parse_args()
    trainer = DPOTrainer(args)
    trainer.train()
