from __future__ import annotations

# Wall-clock benchmark: blend=1 parallel trajectory decode vs pure autoregressive.
#
# Why blend=1 is the only parallelizable setting:
#   The trajectory model predicts one injected recurrent state per chunk. At
#   blend=1 the injected state FULLY REPLACES the running state (see
#   model.blend_into_cache: the else-branch drops `current`), so chunk h's state
#   does NOT depend on chunk h-1. The H chunk states are therefore independent and
#   the H chunks can be decoded as a BATCH of H independent length-C sequences in
#   a single forward pass, instead of H serial chunk passes. At blend<1 each chunk
#   mixes in the previous running state, forcing a serial chain.
#
# This script measures REAL wall-clock time for:
#   AR baseline : the same frozen renderer decodes H*C tokens autoregressively
#                 (one token per forward step) from the same prompt state.
#   Parallel    : predict all H chunk states once, inject at blend=1, and run the
#                 H chunks as a single batched forward (one pass, teacher-forced
#                 over the C positions of each chunk).
#
# Both produce H*C token positions of logits; we report time and tokens/s. This
# is an honest planning+decode throughput comparison, not an accuracy claim.

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_proj.parent.parent))
sys.path.insert(0, str(_proj))

from model import StateHijackingRELAY  # noqa: E402
from relay_utils import load_relay_model  # noqa: E402


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def autoregressive_time(model: StateHijackingRELAY, prompt_ids: torch.Tensor, n_tokens: int, device: str) -> tuple[float, int]:
    rwkv = model.rwkv_model
    out = rwkv(input_ids=prompt_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    cur = out.logits[:, -1:].argmax(dim=-1)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = rwkv(input_ids=cur, past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        cur = out.logits[:, -1:].argmax(dim=-1)
    _sync(device)
    return time.perf_counter() - t0, n_tokens


@torch.no_grad()
def parallel_generate_time(
    model: StateHijackingRELAY,
    z: torch.Tensor,
    H: int,
    C: int,
    device: str,
) -> tuple[float, int]:
    # Honest parallel GENERATION at blend=1: the H chunk-states are independent,
    # so the H chunks are generated as a BATCH of H concurrent sequences, each only
    # C tokens long. Generating H*C tokens therefore costs ~ the time to
    # autoregressively generate ONE C-token chunk at batch size H (the H-fold
    # sequence-length reduction is the real speedup source), plus one batched
    # planning call. This is real token generation (greedy argmax), not scoring.
    rwkv = model.rwkv_model
    _sync(device)
    t0 = time.perf_counter()
    # 1) plan all H chunk states in one batched call (independent at blend=1)
    layer_states = model.predict_trajectory_states(z)  # per layer: [1, H, heads, hd, hd]
    # 2) fresh cache batched over H, inject each chunk's state (blend=1 => pure replace)
    seed = torch.zeros(H, 1, dtype=torch.long, device=device)
    empty = rwkv(input_ids=seed, use_cache=True, return_dict=True)
    cache = empty.past_key_values
    for l, st in enumerate(layer_states):
        cache.layers[l].state["recurrent_state"] = st[0].to(torch.float32)  # [H, heads, hd, hd]
    # 3) autoregressively GENERATE C tokens for all H chunks concurrently (batch=H)
    out = rwkv(input_ids=seed, past_key_values=cache, use_cache=True, return_dict=True)
    cache = out.past_key_values
    cur = out.logits[:, -1:].argmax(dim=-1)  # [H, 1]
    for _ in range(C - 1):
        out = rwkv(input_ids=cur, past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        cur = out.logits[:, -1:].argmax(dim=-1)
    _sync(device)
    return time.perf_counter() - t0, H * C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    model, rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model.eval()
    H, C = args.horizon, args.chunk
    D = int(cfg.model.latent_dim)
    dev = args.device

    z_dtype = next(model.parameters()).dtype
    prompt_ids = torch.zeros(1, 8, dtype=torch.long, device=dev)
    chunk_tokens = torch.zeros(H, C, dtype=torch.long, device=dev)
    z = torch.randn(1, H, D, device=dev, dtype=z_dtype)

    for _ in range(args.warmup):
        autoregressive_time(model, prompt_ids, H * C, dev)
        parallel_generate_time(model, z, H, C, dev)

    ar_times, par_times = [], []
    for _ in range(args.iters):
        t_ar, n_ar = autoregressive_time(model, prompt_ids, H * C, dev)
        t_par, n_par = parallel_generate_time(model, z, H, C, dev)
        ar_times.append(t_ar)
        par_times.append(t_par)

    ar = sum(ar_times) / len(ar_times)
    par = sum(par_times) / len(par_times)
    ntok = H * C
    print("=" * 60)
    print(f"ckpt            : {args.ckpt_dir}")
    print(f"geometry        : H={H} x C={C} = {ntok} tokens, blend=1 (parallel)")
    print(f"AR baseline     : {ar*1000:.1f} ms  ({ntok/ar:.0f} tok/s)")
    print(f"Parallel decode : {par*1000:.1f} ms  ({ntok/par:.0f} tok/s)")
    print(f"WALL-CLOCK SPEEDUP: {ar/par:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
