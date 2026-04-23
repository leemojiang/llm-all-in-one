from torch.utils.data import Dataset, DataLoader
import json
import torch
from typing import List

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length, # Token最大数目
            padding='max_length', # Padding到max_length batch对齐
            truncation=True, # 超过是否截断
            return_tensors='pt'
        )
        # 返回结构
        # encoding = {
        #     'input_ids': Tensor(shape=[1, max_length]),
        #     'attention_mask': Tensor(shape=[1, max_length]),
        #     # 可选: 'token_type_ids', 'position_ids', 'special_tokens_mask' 等
        # }

        input_ids = encoding.input_ids.squeeze() #shape [max_length]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long) # All in max_len -1
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        #  Sample 的示例
        # {
        #     "conversations": [
        #         {"role": "user", "content": "你好"},
        #         {"role": "assistant", "content": "你好！"},
        #         {"role": "user", "content": "再见"},
        #         {"role": "assistant", "content": "再见！"}
        #     ]
        # }

        sample = self.samples[index]

        # 构建 ChatML 格式 prompt（字符串）
        prompt = self._create_chat_prompt(sample['conversations'])

        # 分词并截断，确保长度 <= max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # 右侧填充 pad_token 直到 max_length 长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态 loss mask，仅对 assistant 响应位置计算 loss
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练样本：
        # 模型输入为前 n-1 个 token，预测目标为第 2 到第 n 个 token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标标签（shifted）
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐 Y 的位置（从第一个预测 token 开始）

        return X, Y, loss_mask
    
    def _create_chat_prompt(self, cs:list):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None

        # 返回字符串形式的 prompt，而非直接 tokenize
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    def _generate_loss_mask(self, input_ids:list):
        '''
            1 保留 参与损失计算
            0 不保留 不参与损失计算
            对应的是 X 表示这个token是否计算损失
        '''
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def load_data(self, path) ->List[dict]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
class DPODataset(Dataset):
    # https://github.com/hans0809/MiniMind-in-Depth/blob/main/src/10-DPO-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E7%9A%84%E6%96%B0%E8%8C%83%E5%BC%8F.md
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 特殊标记 <|im_start|>assistant 和 <|im_end|> 的 token ids（一般是开头和结尾的边界符）
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids  # list[int]
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids              # list[int]

        # 加载 JSONL 格式数据：每行为一个 dict，有 chosen 和 rejected
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        chosen = item['chosen']
        rejected = item['rejected']

        # 拼接成字符串（不 tokenize，只生成 prompt 文本）
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 编码为 input_ids（截断 + 填充）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        # 转换为 token ID 列表，长度为 max_length
        chosen_input_ids = chosen_encoding['input_ids']           # shape: (max_length,)
        rejected_input_ids = rejected_encoding['input_ids']       # shape: (max_length,)

        # 构造 loss mask：仅在 assistant 段落（<|im_start|>assistant ... <|im_end|>）中的 token 参与损失
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)     # shape: (max_length,)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids) # shape: (max_length,)

        # （MiniMind没有将padding的token掩掉）

        # 构造训练数据：左移一位预测（即 y 是 x 的下一位）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)      # shape: (max_length - 1,)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)       # shape: (max_length - 1,)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)    # shape: (max_length - 1,)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)  # shape: (max_length - 1,)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)   # shape: (max_length - 1,)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)# shape: (max_length - 1,)

        return {
            'x_chosen': x_chosen,           # shape: (max_length - 1,)
            'y_chosen': y_chosen,           # shape: (max_length - 1,)
            'mask_chosen': mask_chosen,     # shape: (max_length - 1,)

            'x_rejected': x_rejected,       # shape: (max_length - 1,)
            'y_rejected': y_rejected,       # shape: (max_length - 1,)
            'mask_rejected': mask_rejected  # shape: (max_length - 1,)
        }

    def _generate_loss_mask(self, input_ids):
        """
        根据 <|im_start|>assistant 和 <|im_end|> 的位置标记哪些 token 应该参与损失计算。
        返回一个和 input_ids 等长的 0/1 mask。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 匹配一个 assistant 段落开头
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 查找 assistant 的回答终止符 <|im_end|>
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 在 <|im_start|>assistant 和 <|im_end|> 之间部分启用 loss
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask