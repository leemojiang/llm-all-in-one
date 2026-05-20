import argparse
import os
import json
import random
import sys
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from minimind_learning.model.config_minimind import MiniMindConfig
from minimind_learning.model.model_lora import apply_lora, load_lora
from minimind_learning.model.model_minimind import MiniMindForCausalLM
from minimind_learning.trainer.trainer_utils import get_model_params, setup_seed


warnings.filterwarnings("ignore")


DEFAULT_PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的",
    "请用Python写一个计算斐波那契数列的函数",
    '解释一下"光合作用"的基本过程',
    "如果明天下雨，我应该如何出门",
    "比较一下猫和狗作为宠物的优缺点",
    "解释什么是机器学习",
    "推荐一些中国的美食",
]


def is_none(value: str | None) -> bool:
    return value is None or value.lower() == "none"


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if "model" in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
        ))
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        print(f"Loading model from {os.path.abspath(ckp)} ...")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if not is_none(args.lora_weight):
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        print(f"Load General Transformer Model Type:{type(model)}")
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def build_inputs(tokenizer, prompt, conversation, args):
    if "pretrain" in args.weight:
        text = tokenizer.bos_token + prompt
    else:
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(args.open_thinking),
        )
    return tokenizer(text, return_tensors="pt", truncation=True).to(args.device)


def generate_once(model, tokenizer, prompt, conversation, args, streamer=None):
    conversation = conversation[-args.historys:] if args.historys else []
    conversation.append({"role": "user", "content": prompt})
    inputs = build_inputs(tokenizer, prompt, conversation, args)

    start = time.time()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    input_len = len(inputs["input_ids"][0])
    response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": response})

    completion_tokens = len(generated_ids[0]) - input_len
    latency = time.time() - start
    return response, conversation, {
        "prompt_tokens": input_len,
        "completion_tokens": completion_tokens,
        "latency_sec": round(latency, 4),
        "tokens_per_sec": round(completion_tokens / latency, 2) if latency > 0 else None,
    }


def load_eval_prompts(eval_file):
    prompts = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict) or "prompt" not in data:
                raise ValueError(f"{eval_file}:{line_no} 必须是包含 prompt 字段的 JSON 对象")
            prompts.append(data["prompt"])
    return prompts


def run_eval_file(model, tokenizer, args):
    prompts = load_eval_prompts(args.eval_file)
    output_file = None if is_none(args.eval_output_file) else Path(args.eval_output_file)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(prompts)} eval prompts from {args.eval_file}")
    writer = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        for index, prompt in enumerate(prompts):
            setup_seed(args.seed if args.seed >= 0 else random.randint(0, 31415926))
            print(f"\n[{index + 1}/{len(prompts)}] 💬: {prompt}")
            response, _, metrics = generate_once(model, tokenizer, prompt, [], args, streamer=None)
            print(f"🤖️: {response}")
            if args.show_speed:
                print(f"[Speed]: {metrics['tokens_per_sec']} tokens/s")

            record = {"index": index, "prompt": prompt, "answer": response, **metrics}
            if writer:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
    finally:
        if writer:
            writer.close()


def run_chat(model, tokenizer, args):
    conversation = []
    input_mode = str(input("[y] 自动测试\n[n] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt_iter = DEFAULT_PROMPTS if input_mode.lower() == "y" else iter(lambda: input("💬: "), "")

    for prompt in prompt_iter:
        setup_seed(args.seed if args.seed >= 0 else random.randint(0, 31415926))
        if input_mode.lower() == "y":
            print(f"💬: {prompt}")
        print("🤖️: ", end="")
        response, conversation, metrics = generate_once(model, tokenizer, prompt, conversation, args, streamer=streamer)
        if args.show_speed:
            print(f"\n[Speed]: {metrics['tokens_per_sec']} tokens/s\n")
        else:
            print("\n")


def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--tokenizer_path', default='../tokenizer', type=str, help="tokenizer加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA低秩矩阵的rank")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help="重复惩罚系数")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="是否显示decode速度（0=否，1=是）")
    parser.add_argument('--seed', default=-1, type=int, help="随机种子，-1表示每轮随机")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--eval_file', default="None", type=str, help='评测JSONL路径，每行格式为 {"prompt": "..."}；None表示进入问答模式')
    parser.add_argument('--eval-output-file', default="../eval_result.jsonl", type=str, help="评测结果输出路径，默认为None表示不保存")
    args = parser.parse_args()

    model, tokenizer = init_model(args)
    if is_none(args.eval_file):
        run_chat(model, tokenizer, args)
    else:
        run_eval_file(model, tokenizer, args)

if __name__ == "__main__":
    main()
