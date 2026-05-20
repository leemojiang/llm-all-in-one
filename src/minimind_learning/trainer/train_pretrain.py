import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minimind_learning.dataset.lm_dataset import PretrainDataset
from minimind_learning.model.config_minimind import MiniMindConfig
from minimind_learning.trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    save_config_to_json,
    setup_seed,
    unwrap_model,
)


def train_epoch(epoch, loader, iters, start_step=0, start_tokens_seen=0, wandb=None):
    """
    epoch: 当前 epoch (数据集走的次数)
    iters: 每个 epoch 最大迭代步数/总的steps数,始终等于 num_samples // batch_size
    step: 走了多少个mini-batch (注意Effective batch size = batch_size * accumulation_steps)
    """
    start_time = time.time()
    last_step = start_step
    tokens_seen = int(start_tokens_seen)
    last_grad_norm = None
    profile_start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        step_start_time = time.time()
        # B batch size, L seq_len
        input_ids = input_ids.to(args.device) # input_ids shape: [B, L]
        labels = labels.to(args.device) # labels shape: [B, L]
        last_step = step
        # 统计tokens_seen，分布式训练时需要全局同步
        batch_tokens = (labels != -100).sum()
        if dist.is_initialized():
            dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(batch_tokens.item())

        # 这里的LR 是按照 mini-batch/micro-batch/step 来调整的,而不是effective batch size.
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # accumulation_steps 原本大batch拆成小batch 但是为了保证梯度的稳定性(降低方差) 还是需要用大batch估计梯度 所以把小batch的梯度保留然后拼回去
        # 相当于一个正常batch结束了 要更新一下
        did_optimizer_step = step % args.accumulation_steps == 0
        if did_optimizer_step:
            scaler.unscale_(optimizer) #作用：把梯度从 GradScaler 的缩放状态恢复到正常大小。在混合精度训练中，scaler.scale(loss).backward() 会把梯度放大，以避免 fp16 下的数值下溢。在做梯度裁剪 或其他需要真实梯度值的操作前，必须先调用 unscale_。
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) #对梯度进行 裁剪，防止梯度爆炸。
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) #对梯度进行 裁剪，防止梯度爆炸。
            last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            scaler.step(optimizer) # ：执行一次参数更新。和普通的 optimizer.step() 不同，scaler.step() 会检查梯度是否为 NaN 或 Inf（数值不稳定）。
            scaler.update() #动态调整缩放因子。如果梯度稳定，GradScaler 会逐步增大缩放因子，提高精度利用率。如果出现溢出（NaN/Inf），它会减小缩放因子，保证安全。

            optimizer.zero_grad(set_to_none=True)#清空梯度，为下一次迭代做准备。set_to_none=True 会把 .grad 设为 None 而不是 0，这样更节省显存和计算开销。下次反向传播时，PyTorch 会重新分配梯度张量。

        # 最后一个step或者 log_interval个 step保存模型
        if args.profile_train and ((step - start_step) % args.profile_interval == 0 or step == iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated_mb = torch.cuda.memory_allocated() / 1024**2
                reserved_mb = torch.cuda.memory_reserved() / 1024**2
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                allocated_mb = reserved_mb = peak_mb = 0.0
            step_time = time.time() - step_start_time
            avg_step_time = (time.time() - profile_start_time) / max(step - start_step, 1)
            tokens_per_second = int(batch_tokens.item()) / max(step_time, 1e-9)
            Logger(
                f"Profile step {step}: step_time={step_time:.3f}s, avg_step_time={avg_step_time:.3f}s, "
                f"tokens/s={tokens_per_second:.1f}, optimizer_step={did_optimizer_step}, "
                f"cuda_alloc={allocated_mb:.1f}MB, cuda_reserved={reserved_mb:.1f}MB, cuda_peak={peak_mb:.1f}MB"
            )

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            current_tokens_per_second = int(batch_tokens.item()) / max(time.time() - step_start_time, 1e-9)
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, "
                f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, "
                f"tokens_seen: {tokens_seen}, tokens/s: {current_tokens_per_second:.1f}, "
                f"grad_norm: {last_grad_norm if last_grad_norm is not None else 0.0:.4f}, "
                f"epoch_time: {eta_min:.1f}min"
            )

            if wandb:
                log_data = {
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": current_tokens_per_second,
                    "optimizer_step": did_optimizer_step,
                    "epoch_time": eta_min,
                    "step": epoch * iters + step,
                }
                if last_grad_norm is not None:
                    log_data["grad_norm"] = last_grad_norm
                wandb.log(
                    log_data
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = unwrap_model(model).state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss
        if args.max_train_steps > 0 and (step - start_step) >= args.max_train_steps:
            Logger(f"Reached max_train_steps={args.max_train_steps}, stop training early for this run.")
            break

    # epoch 结束时，如果梯度累积窗口没有刚好对齐，也补一次参数更新。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return tokens_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--max_train_steps", type=int, default=0, help="最多训练多少个micro step，0表示不限制")
    parser.add_argument("--profile_train", action="store_true", help="打印训练耗时和显存诊断信息")
    parser.add_argument("--profile_interval", type=int, default=1, help="训练诊断信息打印间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=340, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练，none表示从头开始")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode() #当前GPU编号
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0)) # 确保不同节点之间初始化不一样

    # ========== 2. 配置目录、模型参数、检查 ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None
    save_config_to_json(args.save_dir, args, lm_config)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从 ckp 恢复状态 ==========
    start_epoch, start_step, start_tokens_seen = 0, 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        start_tokens_seen = ckp_data.get("tokens_seen", 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    Logger(f"Starting training from epoch {start_epoch}, step {start_step}, tokens_seen {start_tokens_seen}")
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")

            # 这个Iters参数是因为skip-sampler导致的
            # 如果某个 epoch 原本有 1000 个 batch，checkpoint 存在 start_step=400，那么 SkipBatchSampler 会让 loader 只剩 600 个 batch。此时传：
            # iters = len(loader) + skip = 600 + 400 = 1000 保持总迭代数目不变

            start_tokens_seen = train_epoch(epoch, loader, len(loader) + skip, start_step, start_tokens_seen, wandb)
        else:
            start_tokens_seen = train_epoch(epoch, loader, len(loader), 0, start_tokens_seen, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
