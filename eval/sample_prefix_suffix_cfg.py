#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import cosine_alpha_bar
from relay_utils import load_relay_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    token_ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    token_ids = token_ids.clamp(0, logits.size(-1) - 1)
    token_scores = logits[token_ids]
    logits = logits.clone()
    logits[token_ids] = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
    return logits


def apply_top_p(probs, p):
    if p <= 0.0 or p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum().clamp(min=1e-12)


@torch.no_grad()
def encode_prefix(model, input_ids, attention_mask):
    out = model.rwkv_model(
        input_ids=input_ids,
        attention_mask=attention_mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], attention_mask)
    if model.encoder_type == "variational":
        pooled = pooled.to(next(model.encoder_trunk.parameters()).dtype)
        h = model.encoder_trunk(pooled)
        z = model.mu_head(h)
    elif model.encoder_type == "mlp":
        pooled = pooled.to(next(model.encoder.parameters()).dtype)
        z = model.encoder(pooled)
    elif model.encoder_type == "identity":
        pooled = pooled.to(model.latent_mu.dtype)
        z = (pooled - model.latent_mu) / model.latent_sigma
        z = z.to(next(model.alpha_heads.parameters()).dtype)
    else:
        raise ValueError(model.encoder_type)
    return z


@torch.no_grad()
def sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype):
    z = torch.randn(cond.shape[0], model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(cond.shape[0])
        eps_cond = model.latent_denoiser(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model.latent_denoiser(z, t_batch, cond=uncond)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
    return z


@torch.no_grad()
def generate(model, tokenizer, input_ids, attention_mask, z, args):
    states = model.predict_states(z)
    out = model.rwkv_model(input_ids=input_ids, attention_mask=attention_mask.bool(), use_cache=True, return_dict=True)
    past_kv = model.inject_into_cache(out.past_key_values, states)
    out = model.rwkv_model(input_ids=input_ids, past_key_values=past_kv, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    generated = list(input_ids[0].tolist())
    logits = out.logits[0, -1]
    for _ in range(args.max_new_tokens):
        logits = apply_repetition_penalty(logits.float(), generated, args.repetition_penalty)
        logits = logits / args.temperature
        probs = torch.softmax(logits, dim=-1)
        if args.top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, args.top_k)
            probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
            probs = probs / probs.sum().clamp(min=1e-12)
        probs = apply_top_p(probs, args.top_p)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(generated), states


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    tokenizer_any = cast(Any, tokenizer)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    input_ids = tokenizer_any(args.prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    cond = encode_prefix(model, input_ids, attention_mask)
    z = sample_ddim_cfg(model, cond, args.steps, args.cfg_scale, args.device, dtype)
    text, states = generate(model, tokenizer, input_ids, attention_mask, z, args)
    payload = {
        "ckpt_dir": args.ckpt_dir,
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_gen_type": cfg_any.training.get("gen_type"),
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "z_prefix_norm": float(cond.float().norm(dim=-1).mean().item()),
        "z_suffix_norm": float(z.float().norm(dim=-1).mean().item()),
        "state0_norm": float(states[0].float().norm().item()),
        "text": text,
    }
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
