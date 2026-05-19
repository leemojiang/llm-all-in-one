import json
import os
import random

import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留，不额外插入 system prompt。
    if any(conv.get("tools") or conv.get("functions") for conv in conversations):
        return conversations

    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(system_prompts)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以一定概率移除空 thinking 标签。
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建普通语言模型预训练样本：BOS + text + EOS + PAD。
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # labels 和 input_ids 同长度，模型内部负责 shift；padding 位置用 -100 忽略。
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "reasoning_content": Value("string"),
                        "tools": Value("string"),
                        "functions": Value("string"),
                        "tool_calls": Value("string"),
                    }
                ]
            }
        )
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations: list):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and (message.get("tools") or message.get("functions")):
                raw_tools = message.get("tools") or message.get("functions")
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids: list):
        """
        仅 assistant 回复部分参与 loss 计算；其他位置设置为 -100。
        labels 和 input_ids 同长度，shift 由模型内部完成。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    # https://github.com/hans0809/MiniMind-in-Depth/blob/main/src/10-DPO-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E7%9A%84%E6%96%B0%E8%8C%83%E5%BC%8F.md
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 特殊标记：assistant 段落开始和结束。
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)

        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length")
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length")

        chosen_input_ids = chosen_encoding["input_ids"]
        rejected_input_ids = rejected_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # 构造左移一位预测：x 是输入，y 是下一个 token。
        return {
            "x_chosen": torch.tensor(chosen_input_ids[:-1], dtype=torch.long),
            "y_chosen": torch.tensor(chosen_input_ids[1:], dtype=torch.long),
            "mask_chosen": torch.tensor(chosen_loss_mask[1:], dtype=torch.long),
            "x_rejected": torch.tensor(rejected_input_ids[:-1], dtype=torch.long),
            "y_rejected": torch.tensor(rejected_input_ids[1:], dtype=torch.long),
            "mask_rejected": torch.tensor(rejected_loss_mask[1:], dtype=torch.long),
        }

    def generate_loss_mask(self, input_ids):
        """
        根据 assistant 段落位置标记哪些 token 参与 DPO log-prob 聚合。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
