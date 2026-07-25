#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--trajectory_s1_mode", choices=("independent", "transformer", "rwkv", "birwkv"), default=None)
    p.add_argument("--trajectory_state_blend", type=float, default=None)
    p.add_argument("--trajectory_sampler", choices=("rf_heun",), default=None)
    p.add_argument("--cond_boundary_scale", type=float, default=1.0,
                   help="Scale boundary-token conditioning (condboundary checkpoints only; 1.0=trained behavior)")
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
    z_prefix, _ = model._encode_pooled(pooled)
    return z_prefix, out.past_key_values, out.logits[0, -1]


@torch.no_grad()
def sample_trajectory_ddim_cfg(model, cond, steps, cfg_scale, device, dtype):
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(cond.shape[0])
        eps_cond = model.trajectory_denoiser(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model.trajectory_denoiser(z, t_batch, cond=uncond)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
    return z


@torch.no_grad()
def sample_trajectory_flow_cfg(model, cond, steps, cfg_scale, device, dtype):
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device, dtype=dtype)
        z0_cond = model.trajectory_denoiser(z, t, cond=cond)
        if cfg_scale == 1.0:
            z0_pred = z0_cond
        else:
            z0_uncond = model.trajectory_denoiser(z, t, cond=uncond)
            z0_pred = z0_uncond + cfg_scale * (z0_cond - z0_uncond)
        one_minus_t = (1.0 - t.view(cond.shape[0], 1, 1)).clamp(min=0.1)
        z = z + dt * ((z0_pred - z) / one_minus_t)
    return z


@torch.no_grad()
def sample_trajectory_rf_cfg(model, cond, steps, cfg_scale, device, dtype):
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device, dtype=dtype)
        v_cond = model.trajectory_denoiser(z, t, cond=cond)
        if cfg_scale == 1.0:
            v_pred = v_cond
        else:
            v_uncond = model.trajectory_denoiser(z, t, cond=uncond)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        z = z + dt * v_pred
    return z


@torch.no_grad()
def sample_trajectory_rf_heun_cfg(model, cond, steps, cfg_scale, device, dtype):
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device, dtype=dtype)
        t_next = torch.full((cond.shape[0],), min((i + 1) * dt, 1.0), device=device, dtype=dtype)
        v_cond_1 = model.trajectory_denoiser(z, t, cond=cond)
        if cfg_scale == 1.0:
            v_1 = v_cond_1
        else:
            v_uncond_1 = model.trajectory_denoiser(z, t, cond=uncond)
            v_1 = v_uncond_1 + cfg_scale * (v_cond_1 - v_uncond_1)
        z_euler = z + dt * v_1
        v_cond_2 = model.trajectory_denoiser(z_euler, t_next, cond=cond)
        if cfg_scale == 1.0:
            v_2 = v_cond_2
        else:
            v_uncond_2 = model.trajectory_denoiser(z_euler, t_next, cond=uncond)
            v_2 = v_uncond_2 + cfg_scale * (v_cond_2 - v_uncond_2)
        z = z + 0.5 * dt * (v_1 + v_2)
    return z


@torch.no_grad()
def sample_trajectory_cfg(model, cond, steps, cfg_scale, device, dtype):
    gen_type = str(getattr(model, "_gen_type", "ddpm"))
    if gen_type == "flow":
        return sample_trajectory_flow_cfg(model, cond, steps, cfg_scale, device, dtype)
    if gen_type == "rf":
        if getattr(model, "_trajectory_sampler", None) == "rf_heun":
            return sample_trajectory_rf_heun_cfg(model, cond, steps, cfg_scale, device, dtype)
        return sample_trajectory_rf_cfg(model, cond, steps, cfg_scale, device, dtype)
    return sample_trajectory_ddim_cfg(model, cond, steps, cfg_scale, device, dtype)


@torch.no_grad()
def generate(model, tokenizer, input_ids, attention_mask, prefix_cache, prefix_logits, z_traj, args):
    chunk_size = int(model.trajectory_chunk_size)
    s1_mode = str(model.config.get("trajectory_s1_mode", "independent"))
    generated = list(input_ids[0].tolist())
    past_kv = prefix_cache
    logits = prefix_logits
    state_norm = 0.0

    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        state_norm = float(torch.stack([s.float().norm() for s in layer_states]).mean().item())
        blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    else:
        layer_states = None
        blend = 1.0

    for h in range(z_traj.shape[1]):
        if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
            break
        if s1_mode in ("transformer", "rwkv", "birwkv"):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            state_norm += float(torch.stack([s.float().norm() for s in states_h]).mean().item())
            past_kv = model.inject_into_cache(past_kv, states_h)
        for _ in range(chunk_size):
            if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
                break
            logits = apply_repetition_penalty(logits.float(), generated, args.repetition_penalty)
            if args.temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
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

    if s1_mode not in ("transformer", "rwkv", "birwkv") and z_traj.shape[1] > 0:
        state_norm /= float(z_traj.shape[1])
    return tokenizer.decode(generated), state_norm


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    tokenizer_any = cast(Any, tokenizer)
    cfg_any = cast(Any, cfg)
    if args.trajectory_s1_mode is not None:
        model_any.config.trajectory_s1_mode = args.trajectory_s1_mode
        model_any.trajectory_s1_mode = args.trajectory_s1_mode
    if args.trajectory_state_blend is not None:
        model_any.config.trajectory_state_blend = float(args.trajectory_state_blend)
        model_any.trajectory_state_blend = float(args.trajectory_state_blend)
    model_any._trajectory_sampler = args.trajectory_sampler
    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    if args.cond_boundary_scale != 1.0 and hasattr(model_any.trajectory_denoiser, "cond_boundary_scale"):
        model_any.trajectory_denoiser.cond_boundary_scale = float(args.cond_boundary_scale)

    input_ids = tokenizer_any(args.prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    z_prefix, prefix_cache, prefix_logits = encode_prefix(model_any, input_ids, attention_mask)
    z_traj = sample_trajectory_cfg(model_any, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
    text, state_norm = generate(
        model_any, tokenizer_any, input_ids, attention_mask, prefix_cache, prefix_logits, z_traj, args
    )
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
        "trajectory_sampler": args.trajectory_sampler,
        "trajectory_s1_mode": str(model_any.config.get("trajectory_s1_mode", "independent")),
        "trajectory_state_blend": float(model_any.config.get("trajectory_state_blend", 1.0)),
        "trajectory_horizon": int(model_any.trajectory_horizon),
        "trajectory_chunk_size": int(model_any.trajectory_chunk_size),
        "z_prefix_norm": float(z_prefix.float().norm(dim=-1).mean().item()),
        "z_suffix_traj_norm": float(z_traj.float().norm(dim=-1).mean().item()),
        "state_norm": state_norm,
        "text": text,
    }
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
