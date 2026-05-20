# Run:
#   RUN_GPU_DIAGNOSTICS=1 uv run python -m pytest tests/test_gpu_diagnostics.py -s
#
# Optional variants:
#   RUN_GPU_DIAGNOSTICS=1 GPU_DIAG_BATCH_SIZE=32 GPU_DIAG_SEQ_LEN=512 GPU_DIAG_DTYPE=float16 uv run python -m pytest tests/test_gpu_diagnostics.py -s
#   RUN_GPU_DIAGNOSTICS=1 GPU_DIAG_NUM_WORKERS=2 uv run python -m pytest tests/test_gpu_diagnostics.py -s
#   RUN_GPU_DIAGNOSTICS=1 GPU_DIAG_HIDDEN_SIZE=768 GPU_DIAG_SEQ_LEN=768 GPU_DIAG_BATCH_SIZE=16 GPU_DIAG_ACCUMULATION_STEPS=8 uv run python -m pytest tests/test_gpu_diagnostics.py -s
#   RUN_GPU_DIAGNOSTICS=1 GPU_DIAG_DATA_PATH=../dataset/pretrain_t2t_mini.jsonl GPU_DIAG_HIDDEN_SIZE=768 GPU_DIAG_SEQ_LEN=768 GPU_DIAG_BATCH_SIZE=16 uv run python -m pytest tests/test_gpu_diagnostics.py -s
#
# This is an environment diagnostic, not a normal unit test. It is skipped by
# default because GPU speed and memory vary by machine and current background
# processes.

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from minimind_learning.dataset.lm_dataset import PretrainDataset
from minimind_learning.model.config_minimind import MiniMindConfig
from minimind_learning.trainer.trainer_utils import get_lr, init_model


class RepeatedTextTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        token_ids = [3 + (i % 100) for i, _ in enumerate(text)]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return SimpleNamespace(input_ids=token_ids)


def _memory_mb():
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def _device_memory_mb():
    if not torch.cuda.is_available():
        return {"total_mb": 0.0, "free_mb": 0.0, "used_mb": 0.0}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "total_mb": total_bytes / 1024**2,
        "free_mb": free_bytes / 1024**2,
        "used_mb": (total_bytes - free_bytes) / 1024**2,
    }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seconds_since(start):
    _sync()
    return time.perf_counter() - start


def _write_synthetic_pretrain_jsonl(path, rows, seq_len):
    text = "x" * max(seq_len * 2, 32)
    path.write_text("\n".join(json.dumps({"text": text}) for _ in range(rows)), encoding="utf-8")


def _sum_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": total / 1e6, "trainable_m": trainable / 1e6}


def _summarize_steps(step_reports, tokens_seen, total_memory_mb):
    total_step_times = [step["total_step_s"] for step in step_reports]
    forward_times = [step["forward_s"] for step in step_reports]
    backward_times = [step["backward_s"] for step in step_reports]
    optimizer_times = [step["optimizer_step_s"] for step in step_reports if step["did_optimizer_step"]]
    peak_allocated = max(step["memory_after_step"]["peak_allocated_mb"] for step in step_reports)
    peak_reserved = max(step["memory_after_step"]["reserved_mb"] for step in step_reports)
    measured_s = sum(total_step_times)
    headroom_mb = total_memory_mb - peak_reserved
    headroom_ratio = headroom_mb / total_memory_mb if total_memory_mb else 0.0

    if headroom_ratio < 0.10:
        recommendation = "too_close_to_limit"
    elif headroom_ratio < 0.25:
        recommendation = "usable_but_tight"
    else:
        recommendation = "comfortable"

    return {
        "steps": len(step_reports),
        "total_measured_step_s": measured_s,
        "avg_step_s": statistics.fmean(total_step_times),
        "median_step_s": statistics.median(total_step_times),
        "avg_forward_s": statistics.fmean(forward_times),
        "avg_backward_s": statistics.fmean(backward_times),
        "avg_optimizer_step_s": statistics.fmean(optimizer_times) if optimizer_times else 0.0,
        "tokens_per_second": tokens_seen / measured_s if measured_s > 0 else 0.0,
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "estimated_reserved_headroom_mb": headroom_mb,
        "estimated_reserved_headroom_ratio": headroom_ratio,
        "recommendation": recommendation,
    }


@pytest.mark.skipif(os.getenv("RUN_GPU_DIAGNOSTICS") != "1", reason="set RUN_GPU_DIAGNOSTICS=1 to run GPU diagnostics")
def test_cuda_environment_and_pretrain_step_diagnostics(tmp_path):
    assert torch.cuda.is_available(), "CUDA is not available in this Python environment"

    batch_size = int(os.getenv("GPU_DIAG_BATCH_SIZE", "2"))
    seq_len = int(os.getenv("GPU_DIAG_SEQ_LEN", "128"))
    hidden_size = int(os.getenv("GPU_DIAG_HIDDEN_SIZE", "512"))
    num_layers = int(os.getenv("GPU_DIAG_NUM_LAYERS", "8"))
    num_workers = int(os.getenv("GPU_DIAG_NUM_WORKERS", "0"))
    accumulation_steps = int(os.getenv("GPU_DIAG_ACCUMULATION_STEPS", "1"))
    requested_steps = int(os.getenv("GPU_DIAG_STEPS", str(max(2, accumulation_steps))))
    learning_rate = float(os.getenv("GPU_DIAG_LEARNING_RATE", "5e-4"))
    grad_clip = float(os.getenv("GPU_DIAG_GRAD_CLIP", "1.0"))
    dtype_name = os.getenv("GPU_DIAG_DTYPE", "float16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    data_path_env = os.getenv("GPU_DIAG_DATA_PATH")
    micro_steps = max(requested_steps, accumulation_steps)

    timings = {}
    device_memory_before = _device_memory_mb()
    report = {
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_workers": num_workers,
        "micro_steps": micro_steps,
        "accumulation_steps": accumulation_steps,
        "learning_rate": learning_rate,
        "grad_clip": grad_clip,
        "dtype": dtype_name,
        "data_source": data_path_env or "synthetic_jsonl",
        "device_memory_before": device_memory_before,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    data_path = Path(data_path_env) if data_path_env else tmp_path / "pretrain_gpu_diag.jsonl"
    if data_path_env is None:
        _write_synthetic_pretrain_jsonl(data_path, max(batch_size * micro_steps, 4), seq_len)
    timings["write_dataset_s"] = time.perf_counter() - start

    t = time.perf_counter()
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=False)
    model, tokenizer = init_model(config, "none", device="cuda:0")
    _sync()
    timings["init_model_cuda_s"] = time.perf_counter() - t
    report["model_params"] = _sum_model_params(model)
    report["memory_after_model"] = _memory_mb()

    t = time.perf_counter()
    dataset = PretrainDataset(str(data_path), tokenizer or RepeatedTextTokenizer(), max_length=seq_len)
    timings["init_dataset_s"] = time.perf_counter() - t
    report["dataset_len"] = len(dataset)

    t = time.perf_counter()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_iter = iter(loader)
    timings["loader_iter_s"] = time.perf_counter() - t

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    optimizer.zero_grad(set_to_none=True)

    step_reports = []
    tokens_seen = 0
    last_loss = None
    torch.cuda.reset_peak_memory_stats()
    for micro_step in range(1, micro_steps + 1):
        step_report = {"micro_step": micro_step}
        step_start = time.perf_counter()

        t = time.perf_counter()
        input_ids, labels = next(data_iter)
        step_report["load_batch_cpu_s"] = time.perf_counter() - t

        t = time.perf_counter()
        input_ids = input_ids.to("cuda:0", non_blocking=True)
        labels = labels.to("cuda:0", non_blocking=True)
        step_report["batch_to_cuda_s"] = _seconds_since(t)

        batch_tokens = int((labels != -100).sum().item())
        tokens_seen += batch_tokens
        step_report["batch_tokens"] = batch_tokens

        lr = get_lr(micro_step, micro_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        step_report["learning_rate"] = lr

        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=dtype):
            result = model(input_ids, labels=labels)
            loss = (result.loss + result.aux_loss) / accumulation_steps
        step_report["forward_s"] = _seconds_since(t)

        t = time.perf_counter()
        scaler.scale(loss).backward()
        step_report["backward_s"] = _seconds_since(t)

        did_optimizer_step = micro_step % accumulation_steps == 0
        if did_optimizer_step:
            t = time.perf_counter()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_report["optimizer_step_s"] = _seconds_since(t)
            step_report["grad_norm"] = float(grad_norm.detach().cpu())
        else:
            step_report["optimizer_step_s"] = 0.0
            step_report["grad_norm"] = None
        step_report["did_optimizer_step"] = did_optimizer_step
        step_report["loss"] = float((loss.detach() * accumulation_steps).cpu())
        step_report["memory_after_step"] = _memory_mb()
        step_report["total_step_s"] = _seconds_since(step_start)
        step_report["tokens_per_second"] = batch_tokens / step_report["total_step_s"]
        step_reports.append(step_report)
        last_loss = loss

    report["tokens_seen"] = tokens_seen
    report["loss"] = step_reports[-1]["loss"]
    report["timings"] = timings
    report["steps"] = step_reports
    report["memory_final"] = _memory_mb()
    report["device_memory_after"] = _device_memory_mb()
    report["summary"] = _summarize_steps(step_reports, tokens_seen, report["device_memory_before"]["total_mb"])

    print("\nGPU diagnostics report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    assert last_loss is not None
    assert torch.isfinite(last_loss)
    assert tokens_seen > 0
