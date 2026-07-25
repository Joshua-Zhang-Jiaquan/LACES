#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_utils import load_relay_model
from sample_prefix_suffix_cfg import apply_repetition_penalty, apply_top_p, encode_prefix, sample_ddim_cfg


TASKS_DEFAULT = ["lambada", "obqa", "hellaswag", "mmlu", "race", "siqa", "squad", "story_cloze"]


def parse_args():
    p = argparse.ArgumentParser(description="Run Cola-DLM 8-task generative benchmark with prefix/suffix CFG RELAY.")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_data_dir", default="baseline/Cola-DLM/generate_task_data")
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for task JSONL files. For Cola-DLM acc_calc.py, put this under eval_output/ with a tasks_ prefix.",
    )
    p.add_argument("--tasks", default=",".join(TASKS_DEFAULT))
    p.add_argument("--method", choices=["prefix_cfg", "raw"], default="prefix_cfg")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_prompt_template(task: str, context: str, question: str, choices: Optional[list[str]]) -> str:
    if task == "lambada":
        return question
    if task == "squad":
        return (
            "Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway.\n"
            "Question: In what country is Normandy located?\n"
            "Answer: France\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if task == "obqa":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Question: Which tool is best for tightening a screw?\n"
            "(A) spoon\n"
            "(B) hammer\n"
            "(C) screwdriver\n"
            "(D) paintbrush\n"
            "Answer: screwdriver\n\n"
            "Question: What do plants absorb from the air during photosynthesis?\n"
            "(A) carbon dioxide\n"
            "(B) oxygen\n"
            "(C) helium\n"
            "(D) salt\n"
            "Answer: carbon dioxide\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    if task == "hellaswag":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Context: The girl puts the bread into the toaster and pushes the lever down. The bread\n"
            "(A) becomes a slice of pizza.\n"
            "(B) starts to toast and turn brown.\n"
            "(C) disappears immediately.\n"
            "(D) turns into a glass of water.\n"
            "Answer: starts to toast and turn brown.\n\n"
            "Context: The goalkeeper sees the ball coming towards the net. He dives and\n"
            "(A) catches the ball with his hands.\n"
            "(B) starts dancing in the field.\n"
            "(C) opens a laptop to check email.\n"
            "(D) runs away from the stadium.\n"
            "Answer: catches the ball with his hands.\n\n"
            f"Context: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    if task == "mmlu":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Question: Which gas do plants absorb from the air during photosynthesis?\n"
            "(A) Oxygen\n"
            "(B) Carbon dioxide\n"
            "(C) Nitrogen\n"
            "(D) Hydrogen\n"
            "Answer: Carbon dioxide\n\n"
            "Question: A triangle has angles 50 degrees and 60 degrees. What is the third angle?\n"
            "(A) 60 degrees\n"
            "(B) 70 degrees\n"
            "(C) 80 degrees\n"
            "(D) 90 degrees\n"
            "Answer: 70 degrees\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    if task == "race":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Read the following article and answer the question.\n\n"
            "Article: Mary went to the store to buy some fruits. She bought five apples and two oranges. She paid 5 dollars in total. What did Mary buy?\n"
            "Options:\n"
            "(A) Bananas\n"
            "(B) Apples and oranges\n"
            "(C) Grapes\n"
            "(D) Watermelon\n"
            "Answer: Apples and oranges\n\n"
            f"Article: {question}\n"
            f"Options:\n{current_choices_text}\n"
            "Answer:"
        )
    if task == "siqa":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Question: Jordan wanted to tell a joke to his friends. What does Jordan need to do before this?\n"
            "(A) ignore his friends\n"
            "(B) think of a funny story\n"
            "(C) leave the room\n"
            "Answer: think of a funny story\n\n"
            "Question: Kai helped his neighbor carry heavy groceries inside. How would the neighbor feel?\n"
            "(A) angry\n"
            "(B) grateful\n"
            "(C) scared\n"
            "Answer: grateful\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    if task == "story_cloze":
        choices = choices or ["", ""]
        current_choices_text = f"(A) {choices[0]}\n(B) {choices[1]}"
        return (
            "Story: I wanted to make an omelet. I cracked two eggs into a bowl and whisked them. Then I poured them into a hot pan.\n"
            "(A) I ate a delicious omelet for breakfast.\n"
            "(B) I decided to order a pizza instead.\n"
            "End: I ate a delicious omelet for breakfast.\n\n"
            "Story: The runner tied his shoes tight. He sprinted as fast as he could during the race. He crossed the finish line first.\n"
            "(A) He was sad that he lost the race.\n"
            "(B) He won the gold medal.\n"
            "End: He won the gold medal.\n\n"
            f"Story: {question}\n"
            f"{current_choices_text}\n"
            "End:"
        )
    return question


def get_ground_truth(item: dict[str, Any]) -> str:
    return str(item.get("ground_truth", item.get("answer", item.get("target", ""))))


def get_question(item: dict[str, Any]) -> str:
    return str(item.get("question", item.get("prompt", item.get("context", ""))))


def build_prompt(task: str, item: dict[str, Any]) -> str:
    if "prompt" in item and isinstance(item["prompt"], str) and item["prompt"]:
        prefix = item.get("few_shot_prefix")
        if isinstance(prefix, str) and prefix and not item["prompt"].startswith(prefix):
            return prefix + item["prompt"]
        return item["prompt"]
    return apply_prompt_template(
        task,
        str(item.get("context", "")),
        get_question(item),
        item.get("choices", None),
    )


@torch.no_grad()
def sample_next_token(logits, generated_ids, args):
    logits = apply_repetition_penalty(logits.float(), generated_ids, args.repetition_penalty)
    if args.temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / args.temperature
    probs = torch.softmax(logits, dim=-1)
    if args.top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, args.top_k)
        probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
        probs = probs / probs.sum().clamp(min=1e-12)
    probs = apply_top_p(probs, args.top_p)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, z, args) -> tuple[str, int, int]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    states = model.predict_states(z)
    out = model.rwkv_model(input_ids=input_ids, attention_mask=attention_mask.bool(), use_cache=True, return_dict=True)
    cache = model.inject_into_cache(out.past_key_values, states)
    out = model.rwkv_model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    cache = out.past_key_values
    generated_ids: list[int] = []
    all_ids = list(input_ids[0].tolist())
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=args.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), input_ids.shape[1], len(generated_ids)


@torch.no_grad()
def generate_raw_answer(model, tokenizer, input_ids, args) -> tuple[str, int, int]:
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    generated_ids: list[int] = []
    all_ids = list(input_ids[0].tolist())
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), input_ids.shape[1], len(generated_ids)


def iter_jsonl(path: Path, max_samples: int):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary: dict[str, Any] = {
        "ckpt_dir": args.ckpt_dir,
        "checkpoint_step": ckpt.get("step", -1),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "tasks": {},
    }
    for task_i, task in enumerate(tasks):
        input_path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not input_path.exists():
            print(f"[SKIP] missing {input_path}", flush=True)
            continue
        output_path = output_dir / f"{task}.jsonl"
        n = 0
        with output_path.open("w", encoding="utf-8") as out_f:
            for sample_i, item in iter_jsonl(input_path, args.max_samples):
                choices = item.get("choices", None)
                prompt = build_prompt(task, item)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                attention_mask = torch.ones_like(input_ids)
                if args.method == "raw":
                    generated, prompt_tokens, generated_tokens = generate_raw_answer(model, tokenizer, input_ids, args)
                else:
                    torch.manual_seed(args.seed + 100000 * task_i + sample_i)
                    cond = encode_prefix(model, input_ids, attention_mask)
                    z = sample_ddim_cfg(model, cond, args.steps, args.cfg_scale, args.device, dtype)
                    generated, prompt_tokens, generated_tokens = generate_answer(model, tokenizer, prompt, z, args)
                rec = dict(item)
                rec.update(
                    {
                        "id": item.get("id", sample_i),
                        "prompt": prompt,
                        "generate": generated,
                        "ground_truth": get_ground_truth(item),
                        "choices": choices or item.get("choices", []),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "method": args.method,
                        "cfg_scale": args.cfg_scale,
                        "steps": args.steps,
                    }
                )
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples", flush=True)
        summary["tasks"][task] = {"samples": n, "output": str(output_path)}
        print(f"[DONE] {task}: {n} -> {output_path}", flush=True)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
