#!/usr/bin/env python3
"""LaDiR-style repulsion guidance for trajectory DDiM sampling (zero-training).

At each denoising step, applies an RBF repulsion force between chunk latents
to prevent latent homogenization collapse. Based on LaDiR (arXiv:2510.04573).

The repulsion force pushes chunk latents apart in proportion to their similarity:
  F_h = Σ_{j≠h} γ_t · exp(-‖z_h - z_j‖²/σ²) · (z_h - z_j)

Where γ_t ramps up mid-schedule (max at t=0.5) and σ² is the median pairwise
squared distance. No training required - just a modified sampler.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval/sample_trajectory_ladire.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --prompt "The history of artificial intelligence" \
    --output /tmp/ladire_sample.json \
    --gamma_max 0.3 --steps 100 --cfg_scale 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import cosine_alpha_bar
from relay_utils import load_relay_model
from sample_prefix_suffix_trajectory_cfg import (
    apply_repetition_penalty,
    apply_top_p,
    encode_prefix,
    generate,
)


def _median_pairwise_sq_dist(z: torch.Tensor) -> float:
    """Median squared distance between all chunk latents in a single trajectory."""
    H = z.shape[0]
    dists = []
    for h in range(H):
        for j in range(h + 1, H):
            d = float((z[h] - z[j]).pow(2).sum())
            dists.append(d)
    if not dists:
        return 1.0
    dists.sort()
    return dists[len(dists) // 2]


@torch.no_grad()
def sample_trajectory_ddim_ladire(
    model, cond, steps, cfg_scale, device, dtype,
    gamma_max=0.3, sigma_frac=0.5,
):
    """DDiM sampler with LaDiR repulsion guidance between chunk latents.

    gamma_max: peak repulsion strength (applied at t=0.5, ramped sinusoidally)
    sigma_frac: sigma² = sigma_frac × median_pairwise_sq_dist
    """
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)

    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        t_mid = float((t_cur + t_nxt) / 2)

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
        z_next = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps

        # --- LaDiR repulsion guidance ---
        if gamma_max > 0 and horizon > 1:
            # sinusoidal schedule: peak at t=0.5, zero at t=0 and t=1
            gamma_t = gamma_max * math.sin(math.pi * t_mid)

            med_sq = _median_pairwise_sq_dist(z_next[0])
            sigma_sq = max(sigma_frac * med_sq, 1e-4)

            for h in range(horizon):
                force = torch.zeros_like(z_next[0, h])
                for j in range(horizon):
                    if h == j:
                        continue
                    diff = z_next[0, h] - z_next[0, j]
                    dist_sq = float((diff * diff).sum())
                    # RBF kernel
                    weight = math.exp(-dist_sq / sigma_sq)
                    force = force + weight * diff
                z_next[0, h] = z_next[0, h] + gamma_t * force

        z = z_next

    return z


@torch.no_grad()
def compute_chunk_ce(model, z_traj, input_ids, attention_mask, blend=0.7):
    """Compute teacher-forced CE for the full suffix under injected states.

    Returns mean CE (lower = better).
    """
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    S = H * C

    ids = input_ids[0, :S]
    am = attention_mask[0, :S]
    if ids.numel() < S:
        pad = torch.zeros(S - ids.numel(), device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, pad])
        am = torch.cat([am, torch.zeros_like(pad, dtype=am.dtype)])

    states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(
        input_ids=ids.unsqueeze(0),
        attention_mask=am.bool().unsqueeze(0),
        output_hidden_states=False,
        use_cache=True,
        return_dict=True,
    )
    pkv = model.blend_into_cache(out.past_key_values, [s[:, 0] for s in states], blend)
    out2 = model.rwkv_model(
        input_ids=ids.unsqueeze(0),
        attention_mask=am.bool().unsqueeze(0),
        past_key_values=pkv,
        use_cache=False,
        return_dict=True,
    )
    logits = out2.logits[:, :-1, :]
    targets = ids[1:].unsqueeze(0)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(ce.item())


def effective_rank(z: torch.Tensor) -> float:
    """Effective rank via entropy of singular values."""
    x = z.float().reshape(-1, z.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    ent = -(p * p.log()).sum()
    return float(ent.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--gamma_max", type=float, default=0.3,
                    help="Peak LaDiR repulsion strength (0=off)")
    ap.add_argument("--sigma_frac", type=float, default=0.5,
                    help="sigma² as fraction of median pairwise sq dist")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_diag_samples", type=int, default=16,
                    help="Number of diagnostic samples for eff_rank/CE sweep")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2

    # --- Diagnostic: sweep gamma_max vs eff_rank and CE ---
    if args.num_diag_samples > 0:
        print(f"\n=== LaDiR diagnostic sweep ({args.num_diag_samples} seeds) ===")
        base_prompt = args.prompt
        input_ids_base = tokenizer(base_prompt, return_tensors="pt").input_ids.to(args.device)
        am_base = torch.ones_like(input_ids_base)

        for gm in [0.0, 0.1, 0.3, 0.5]:
            ranks = []
            ces = []
            for s in range(args.num_diag_samples):
                torch.manual_seed(args.seed + s)
                z_prefix, _, _ = encode_prefix(model, input_ids_base, am_base)
                z_traj = sample_trajectory_ddim_ladire(
                    model, z_prefix, args.steps, args.cfg_scale, args.device, dtype,
                    gamma_max=gm, sigma_frac=args.sigma_frac,
                )
                # per-chunk effective rank
                ranks.append(effective_rank(z_traj[0]))
                # CE (uses full prefix+suffix as single passage — consistent with training)
                ces.append(compute_chunk_ce(
                    model, z_traj, input_ids_base, am_base, blend=0.7,
                ))
            print(
                f"  gamma={gm:.1f}: eff_rank={sum(ranks)/len(ranks):.2f}±{torch.tensor(ranks).float().std():.2f}  "
                f"CE={sum(ces)/len(ces):.2f}±{torch.tensor(ces).float().std():.2f}",
                flush=True,
            )

    # --- Single sample with LaDiR ---
    print(f"\n=== LaDiR sample gamma={args.gamma_max} ===")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
    z_traj = sample_trajectory_ddim_ladire(
        model, z_prefix, args.steps, args.cfg_scale, args.device, dtype,
        gamma_max=args.gamma_max, sigma_frac=args.sigma_frac,
    )
    text, state_norm = generate(
        model, tokenizer, input_ids, attention_mask, prefix_cache, prefix_logits,
        z_traj, args,
    )

    payload = {
        "method": "LaDiR",
        "ckpt_dir": args.ckpt_dir,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "gamma_max": args.gamma_max,
        "sigma_frac": args.sigma_frac,
        "z_prefix_norm": float(z_prefix.float().norm(dim=-1).mean().item()),
        "z_traj_norm": float(z_traj.float().norm(dim=-1).mean().item()),
        "z_traj_eff_rank": effective_rank(z_traj[0]),
        "state_norm": state_norm,
        "text": text,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  eff_rank={payload['z_traj_eff_rank']:.2f}")
    print(f"  text preview: {text[:200]}...")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
