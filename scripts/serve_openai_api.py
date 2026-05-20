import argparse
import json
import re
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import uvicorn
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from minimind_learning.model.config_minimind import MiniMindConfig
from minimind_learning.model.model_lora import apply_lora, load_lora
from minimind_learning.model.model_minimind import MiniMindForCausalLM

warnings.filterwarnings('ignore')

app = FastAPI()
server_args = None
model = None
tokenizer = None
device = None


def is_none(value: str | None) -> bool:
    return value is None or value.lower() == "none"


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = PROJECT_ROOT / args.save_dir / f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_position_embeddings=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if not is_none(args.lora_weight):
            apply_lora(model, rank=args.lora_rank)
            lora_path = PROJECT_ROOT / args.save_dir / 'lora' / f'{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth'
            load_lora(model, str(lora_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.half().eval().to(args.device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list[dict[str, Any]] = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict[str, Any] | None = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def build_prompt(messages, max_tokens, tools=None, open_thinking=False):
    if 'pretrain' in server_args.weight:
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        return tokenizer.bos_token + last_user
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools or None,
        open_thinking=open_thinking,
    )[-max_tokens:]


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        new_prompt = build_prompt(messages, max_tokens, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            try:
                model.generate(
                    inputs=inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )
            except Exception as exc:
                queue.put(json.dumps({"error": str(exc)}, ensure_ascii=False))
                queue.put(None)

        Thread(target=_generate, daemon=True).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            if text.startswith('{"error":'):
                yield text
                continue
            full_text += text

            if not thinking_ended:
                pos = full_text.find('</think>')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


def sse_response_chunks(request: ChatRequest):
    for chunk in generate_stream_response(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        tools=request.tools,
        open_thinking=request.get_open_thinking()
    ):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                sse_response_chunks(request),
                media_type="text/event-stream"
            )
        else:
            new_prompt = build_prompt(
                request.messages,
                request.max_tokens,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs=inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--tokenizer_path', default='../tokenizer', type=str, help="tokenizer加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA低秩矩阵的rank")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--host', default='0.0.0.0', type=str, help="服务监听地址")
    parser.add_argument('--port', default=8998, type=int, help="服务监听端口")
    args = parser.parse_args()
    server_args = args
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host=args.host, port=args.port)
