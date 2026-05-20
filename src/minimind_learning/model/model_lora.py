import os

import torch
from torch import nn


class LoRA(nn.Module):
    """A small low-rank adapter attached to selected linear layers."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def _model_device(model):
    return next(model.parameters()).device


def apply_lora(model, rank: int = 16):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            if hasattr(module, "lora"):
                continue
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(_model_device(model))
            module.lora = lora
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA 权重不存在: {path}")

    state_dict = torch.load(path, map_location=_model_device(model))
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    missing_layers = []
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            lora_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
            if lora_state:
                module.lora.load_state_dict(lora_state)
            else:
                missing_layers.append(name)

    if missing_layers:
        raise ValueError(f"LoRA 权重中缺少 {len(missing_layers)} 个 adapter 层，示例: {missing_layers[:3]}")


def save_lora(model, path: str):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f"{clean_name}.lora.{k}": v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path: str, save_path: str):
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if ".lora." not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and ".lora." not in name:
            state_dict[f"{name}.weight"] = module.weight.data.clone().cpu().half()
            if hasattr(module, "lora"):
                state_dict[f"{name}.weight"] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
