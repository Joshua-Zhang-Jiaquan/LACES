#!/usr/bin/env python3
"""Run the Cola-DLM 8-task generate-then-match benchmark with a prefix/suffix
TRAJECTORY CFG RELAY model.

This mirrors scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py (single-z), but
uses the trajectory prefix/suffix pipeline from
scripts/eval/sample_prefix_suffix_trajectory_cfg.py:

  prompt -> encode_prefix -> z_prefix
         -> sample_trajectory_cfg (ddim / rf / rf_heun / flow, dispatched on the
            checkpoint's gen_type) -> z_traj [B, horizon, latent_dim]
         -> chunk-wise decode: per chunk inject/blend the predicted RWKV state,
            then autoregress chunk_size tokens with the frozen RWKV.

Differences vs the raw trajectory sampler that make it usable for the 8-task
generate-then-match protocol:
  1. Greedy decoding when temperature <= 0 (the sampler divides by temperature and
     would crash at the Cola-DLM default temperature=0.0).
  2. Returns ONLY the newly generated continuation (not prompt+continuation), so the
     output JSONL matches what baseline/Cola-DLM/scripts/acc_calc.py expects.
  3. Stops at EOS and caps at --max_new_tokens.

Output JSONL schema and run_summary.json are identical to the single-z runner, so
the same acc_calc.py scoring applies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relay_utils import load_relay_model

# Proven trajectory sampling pipeline (sampler dispatch + helpers).
from sample_prefix_suffix_trajectory_cfg import (
    apply_repetition_penalty,
    apply_top_p,
    encode_prefix,
    sample_trajectory_cfg,
)
from sample_trajectory_ladire import sample_trajectory_ddim_ladire


def _safe_apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    token_ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    token_ids = token_ids.clamp(0, logits.size(-1) - 1)
    token_scores = logits[token_ids]
    logits = logits.clone()
    logits[token_ids] = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
    return logits

# Reuse the single-z task harness (prompt templates / ground truth / jsonl iter).
from run_cola_dlm_tasks_prefix_suffix_cfg import (
    TASKS_DEFAULT,
    build_prompt,
    get_ground_truth,
    iter_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Cola-DLM 8-task generative benchmark with prefix/suffix TRAJECTORY CFG RELAY."
    )
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_data_dir", default="baseline/Cola-DLM/generate_task_data")
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for task JSONL files. For Cola-DLM acc_calc.py, put this under eval_output/ with a tasks_ prefix.",
    )
    p.add_argument("--tasks", default=",".join(TASKS_DEFAULT))
    p.add_argument("--method", choices=["traj_cfg", "raw"], default="traj_cfg")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument(
        "--trajectory_s1_mode",
        choices=("independent", "transformer", "rwkv", "birwkv"),
        default=None,
        help="Override the checkpoint's trajectory_s1_mode if set.",
    )
    p.add_argument("--trajectory_state_blend", type=float, default=None, help="Override trajectory_state_blend if set.")
    p.add_argument("--trajectory_sampler", choices=("rf_heun",), default=None, help="Use rf_heun for rf checkpoints.")
    p.add_argument("--cond_boundary_scale", type=float, default=1.0,
                   help="Scale boundary-token conditioning (condboundary checkpoints only; 1.0=trained behavior)")
    p.add_argument("--ladire_gamma", type=float, default=0.0,
                   help="LaDiR repulsion gamma (0=off, 0.1=recommended). Overrides sampler to LaDiR DDiM.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def sample_next_token(logits, generated_ids, args) -> int:
    """Greedy when temperature<=0, else temperature/top_k/top_p sampling.

    Uses the trajectory module's repetition-penalty / top-p helpers so behavior
    matches sample_prefix_suffix_trajectory_cfg.py, while adding the greedy path
    the Cola-DLM protocol needs (temperature=0.0).
    """
    logits = _safe_apply_repetition_penalty(logits.float(), generated_ids, args.repetition_penalty)
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
def sample_next_tokens(logits, generated_by_row, args) -> torch.Tensor:
    next_ids = []
    for row in range(logits.shape[0]):
        next_ids.append(sample_next_token(logits[row], generated_by_row[row], args))
    return torch.tensor(next_ids, device=logits.device, dtype=torch.long)


@torch.no_grad()
def recompute_logits_from_injected_cache(model, context_ids, attention_mask, past_kv):
    out = model.rwkv_model(
        input_ids=context_ids,
        attention_mask=attention_mask.bool() if attention_mask is not None else None,
        past_key_values=past_kv,
        use_cache=True,
        return_dict=True,
    )
    return out.past_key_values, out.logits[:, -1]


def build_padded_contexts(contexts, pad_id, device):
    max_len = max(len(ids) for ids in contexts)
    input_ids = torch.full((len(contexts), max_len), int(pad_id), device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(contexts), max_len), device=device, dtype=torch.long)
    for row, ids in enumerate(contexts):
        ids_tensor = torch.tensor(ids, device=device, dtype=torch.long)
        input_ids[row, : ids_tensor.numel()] = ids_tensor
        attention_mask[row, : ids_tensor.numel()] = 1
    return input_ids, attention_mask


@torch.no_grad()
def generate_answer_trajectory(model, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, args):
    """Chunk-wise trajectory decode; returns ONLY the new continuation tokens."""
    chunk_size = int(model.trajectory_chunk_size)
    s1_mode = str(model.config.get("trajectory_s1_mode", "independent"))
    prompt_len = input_ids.shape[1]
    all_ids = list(input_ids[0].tolist())  # full context for repetition penalty
    new_ids: list[int] = []
    past_kv = prefix_cache
    logits = prefix_logits
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    else:
        layer_states = None
        blend = 1.0

    stop = False
    for h in range(z_traj.shape[1]):
        if stop or len(new_ids) >= args.max_new_tokens:
            break
        if layer_states is not None:
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
        context_ids = torch.tensor([all_ids], device=input_ids.device, dtype=torch.long)
        context_mask = torch.ones_like(context_ids)
        past_kv, logits_batch = recompute_logits_from_injected_cache(model, context_ids, context_mask, past_kv)
        logits = logits_batch[0]
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                stop = True
                break
            next_id = sample_next_token(logits, all_ids, args)
            if eos_id is not None and next_id == eos_id:
                stop = True
                break
            new_ids.append(next_id)
            all_ids.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=input_ids.device),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]

    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, prompt_len, len(new_ids)


@torch.no_grad()
def generate_answer_trajectory_batch(model, tokenizer, input_ids, attention_mask, prefix_cache, prefix_logits, z_traj, args):
    batch_size = input_ids.shape[0]
    chunk_size = int(model.trajectory_chunk_size)
    s1_mode = str(model.config.get("trajectory_s1_mode", "independent"))
    prompt_tokens = attention_mask.long().sum(dim=1).tolist()
    all_ids = []
    for row in range(batch_size):
        valid = input_ids[row, : int(prompt_tokens[row])]
        all_ids.append(list(valid.tolist()))
    new_ids: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)
    past_kv = prefix_cache
    logits = prefix_logits
    eos_id = getattr(tokenizer, "eos_token_id", None)
    filler_id = int(eos_id if eos_id is not None else 0)

    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    else:
        layer_states = None
        blend = 1.0

    for h in range(z_traj.shape[1]):
        if bool(finished.all()) or all(len(ids) >= args.max_new_tokens for ids in new_ids):
            break
        if layer_states is not None:
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
        contexts = [all_ids[row] for row in range(batch_size)]
        context_ids, context_mask = build_padded_contexts(contexts, filler_id, input_ids.device)
        past_kv, logits = recompute_logits_from_injected_cache(model, context_ids, context_mask, past_kv)
        for _ in range(chunk_size):
            active = (~finished) & torch.tensor(
                [len(ids) < args.max_new_tokens for ids in new_ids],
                device=input_ids.device,
                dtype=torch.bool,
            )
            if not bool(active.any()):
                break
            next_ids = sample_next_tokens(logits, all_ids, args)
            for row in range(batch_size):
                if not bool(active[row]):
                    next_ids[row] = filler_id
                    continue
                token_id = int(next_ids[row].item())
                if eos_id is not None and token_id == eos_id:
                    finished[row] = True
                    next_ids[row] = filler_id
                    continue
                new_ids[row].append(token_id)
                all_ids[row].append(token_id)
            out = model.rwkv_model(
                input_ids=next_ids.view(batch_size, 1),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1]

    texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in new_ids]
    generated_tokens = [len(ids) for ids in new_ids]
    return texts, prompt_tokens, generated_tokens


@torch.no_grad()
def generate_raw_answer(model, tokenizer, input_ids, args):
    """Pure frozen-RWKV autoregression baseline (no trajectory state injection)."""
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        new_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), input_ids.shape[1], len(new_ids)


def main():
    args = parse_args()
    args.batch_size = max(1, int(args.batch_size))
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    tokenizer_any = cast(Any, tokenizer)
    if getattr(tokenizer_any, "pad_token_id", None) is None:
        if getattr(tokenizer_any, "eos_token", None) is not None:
            tokenizer_any.pad_token = tokenizer_any.eos_token
        elif getattr(tokenizer_any, "unk_token", None) is not None:
            tokenizer_any.pad_token = tokenizer_any.unk_token
    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._trajectory_sampler = args.trajectory_sampler
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    if args.trajectory_s1_mode is not None:
        model_any.config.trajectory_s1_mode = args.trajectory_s1_mode
        model_any.trajectory_s1_mode = args.trajectory_s1_mode
    if args.trajectory_state_blend is not None:
        model_any.config.trajectory_state_blend = float(args.trajectory_state_blend)
        model_any.trajectory_state_blend = float(args.trajectory_state_blend)
    if args.cond_boundary_scale != 1.0 and hasattr(model_any.trajectory_denoiser, "cond_boundary_scale"):
        model_any.trajectory_denoiser.cond_boundary_scale = float(args.cond_boundary_scale)

    s1_mode = str(model_any.config.get("trajectory_s1_mode", "independent"))
    state_blend = float(model_any.config.get("trajectory_state_blend", 1.0))
    gen_type = str(getattr(model_any, "_gen_type", "ddpm"))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary: dict[str, Any] = {
        "ckpt_dir": args.ckpt_dir,
        "checkpoint_step": ckpt.get("step", -1),
        "method": args.method,
        "gen_type": gen_type,
        "trajectory_sampler": args.trajectory_sampler,
        "trajectory_s1_mode": s1_mode,
        "trajectory_state_blend": state_blend,
        "cond_boundary_scale": float(args.cond_boundary_scale),
        "trajectory_horizon": int(model_any.trajectory_horizon),
        "trajectory_chunk_size": int(model_any.trajectory_chunk_size),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
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
        batch = []
        with output_path.open("w", encoding="utf-8") as out_f:
            def flush_batch():
                nonlocal batch, n
                if not batch:
                    return
                sample_ids = [sample_i for sample_i, _ in batch]
                items = [item for _, item in batch]
                prompts = [build_prompt(task, item) for item in items]
                if args.method == "raw" or len(batch) == 1:
                    for sample_i, item, prompt in zip(sample_ids, items, prompts):
                        input_ids = tokenizer_any(prompt, return_tensors="pt").input_ids.to(args.device)
                        attention_mask = torch.ones_like(input_ids)
                        if args.method == "raw":
                            generated, prompt_tokens, generated_tokens = generate_raw_answer(model_any, tokenizer, input_ids, args)
                        else:
                            torch.manual_seed(args.seed + 100000 * task_i + sample_i)
                            z_prefix, prefix_cache, prefix_logits = encode_prefix(model_any, input_ids, attention_mask)
                            if args.ladire_gamma > 0:
                                z_traj = sample_trajectory_ddim_ladire(
                                    model_any, z_prefix, args.steps, args.cfg_scale,
                                    args.device, dtype, gamma_max=args.ladire_gamma,
                                )
                            else:
                                z_traj = sample_trajectory_cfg(model_any, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
                            generated, prompt_tokens, generated_tokens = generate_answer_trajectory(
                                model_any, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, args
                            )
                        rec = dict(item)
                        rec.update(
                            {
                                "id": item.get("id", sample_i),
                                "prompt": prompt,
                                "generate": generated,
                                "ground_truth": get_ground_truth(item),
                                "choices": item.get("choices", []),
                                "prompt_tokens": prompt_tokens,
                                "generated_tokens": generated_tokens,
                                "method": args.method,
                                "cfg_scale": args.cfg_scale,
                                "steps": args.steps,
                            }
                        )
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                else:
                    encoded = tokenizer_any(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_ids = encoded.input_ids.to(args.device)
                    attention_mask = encoded.attention_mask.to(args.device)
                    torch.manual_seed(args.seed + 100000 * task_i + sample_ids[0])
                    z_prefix, prefix_cache, prefix_logits = encode_prefix(model_any, input_ids, attention_mask)
                    if args.ladire_gamma > 0:
                        z_traj = sample_trajectory_ddim_ladire(
                            model_any, z_prefix, args.steps, args.cfg_scale,
                            args.device, dtype, gamma_max=args.ladire_gamma,
                        )
                    else:
                        z_traj = sample_trajectory_cfg(model_any, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
                    generated, prompt_tokens, generated_tokens = generate_answer_trajectory_batch(
                        model_any, tokenizer, input_ids, attention_mask, prefix_cache, prefix_logits, z_traj, args
                    )
                    for sample_i, item, prompt, gen, ptok, gtok in zip(
                        sample_ids, items, prompts, generated, prompt_tokens, generated_tokens
                    ):
                        rec = dict(item)
                        rec.update(
                            {
                                "id": item.get("id", sample_i),
                                "prompt": prompt,
                                "generate": gen,
                                "ground_truth": get_ground_truth(item),
                                "choices": item.get("choices", []),
                                "prompt_tokens": ptok,
                                "generated_tokens": gtok,
                                "method": args.method,
                                "cfg_scale": args.cfg_scale,
                                "steps": args.steps,
                            }
                        )
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples", flush=True)
                batch = []

            for sample_i, item in iter_jsonl(input_path, args.max_samples):
                if args.shard_count > 1 and (sample_i % args.shard_count) != args.shard_index:
                    continue
                batch.append((sample_i, item))
                if len(batch) >= args.batch_size:
                    flush_batch()
            flush_batch()
        summary["tasks"][task] = {"samples": n, "output": str(output_path)}
        print(f"[DONE] {task}: {n} -> {output_path}", flush=True)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY] {output_dir / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
