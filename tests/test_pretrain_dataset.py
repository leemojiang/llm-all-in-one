# Run:
#   uv run python -m pytest tests/test_pretrain_dataset.py -q
#
# This file keeps the default tests small and CPU-friendly. It verifies the
# dataset path, token accounting, and checkpoint metadata without launching a
# full training job.

from types import SimpleNamespace

import torch

from minimind_learning.dataset.lm_dataset import PretrainDataset
from minimind_learning.model.config_minimind import MiniMindConfig
from minimind_learning.model.model_minimind import MiniMindForCausalLM
from minimind_learning.trainer.trainer_utils import lm_checkpoint


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        token_ids = [3 + (ord(ch) % 32) for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return SimpleNamespace(input_ids=token_ids)


def test_pretrain_dataset_loads_jsonl_and_masks_padding(tmp_path):
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text('{"text":"hello"}\n{"text":"world!"}\n', encoding="utf-8")

    dataset = PretrainDataset(str(data_path), TinyTokenizer(), max_length=8)

    input_ids, labels = dataset[0]
    assert len(dataset) == 2
    assert input_ids.shape == torch.Size([8])
    assert labels.shape == torch.Size([8])
    assert input_ids[0].item() == TinyTokenizer.bos_token_id
    assert labels[-1].item() == -100
    assert int((labels != -100).sum().item()) == 7


def test_pretrain_forward_backward_and_token_count(tmp_path):
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text('{"text":"abc"}\n{"text":"defgh"}\n', encoding="utf-8")
    dataset = PretrainDataset(str(data_path), TinyTokenizer(), max_length=8)

    batch = [dataset[0], dataset[1]]
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    tokens_seen = int((labels != -100).sum().item())

    config = MiniMindConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=64,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = model(input_ids, labels=labels)
    loss = result.loss + result.aux_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert torch.isfinite(loss)
    assert tokens_seen == 12


def test_lm_checkpoint_saves_and_loads_tokens_seen(tmp_path):
    config = MiniMindConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=64,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    lm_checkpoint(
        config,
        weight="unit_pretrain",
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=3,
        tokens_seen=1234,
        save_dir=str(tmp_path),
    )
    checkpoint = lm_checkpoint(config, weight="unit_pretrain", save_dir=str(tmp_path))

    assert checkpoint["epoch"] == 0
    assert checkpoint["step"] == 3
    assert checkpoint["tokens_seen"] == 1234
    assert checkpoint["optimizer"] is not None
