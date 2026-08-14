"""Offline larger-batch sampler eval for the BiRWKV token-diffusion denoiser.

Settles the code/plan section 4.6 "iterative beats single-step" gate with real
sample sizes: grid of mask_ratio x denoise-steps over held-out sequences from
one or more packed-pkl dirs, emitting qz_capability_sampler_shard_v1 records
so merge_eval.py produces bootstrap CIs per (arm = ratio/steps combo).

One GPU per shard; the launcher strides shards across GPUs like the other
capability evals.

Usage:
  python -m eval.capability.sampler_eval_offline \
    --ckpt_dir <step_dir> --model_dir <hf-dir> \
    --token_dirs dirA,dirB --per_dir 512 --window 1024 \
    --shard 0 --num_shards 8 --output <shard.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_SCALE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

MASK_RATIOS = (0.3, 0.5, 0.7, 0.9)
STEP_GRID = (1, 8, 16, 32, 64)


def _load_tail_sequences(token_dir: str, count: int, window: int) -> list[torch.Tensor]:
    """Take `count` windows from the TAIL of the pack (training holds out the tail)."""
    from data.fineweb4096_packed import FineWebPackedPickleDataset

    ds = FineWebPackedPickleDataset(data_dir=token_dir, max_length=4096,
                                    pad_token_id=0, cache_shards=1)
    n = len(ds)
    out = []
    for i in range(max(0, n - count), n):
        ids = ds[i]["input_ids"][:window]
        out.append(ids)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--token_dirs", required=True, help="comma-separated packed-pkl dirs")
    p.add_argument("--per_dir", type=int, default=512)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--self_correction", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
    from models.birwkv7_diffusion import MASK_TOKEN_ID, iterative_denoise, kendall_tau_commit_order

    device = torch.device("cuda")
    loaded = load_birwkv_diffusion(args.ckpt_dir, args.model_dir)
    model = loaded.model

    # gather sequences, tagged by source dir name
    seqs: list[tuple[str, torch.Tensor]] = []
    for d in args.token_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        tag = Path(d).name
        for ids in _load_tail_sequences(d, args.per_dir, args.window):
            seqs.append((tag, ids))
    # shard by index
    seqs = [sv for i, sv in enumerate(seqs) if i % args.num_shards == args.shard]
    print(f"[sampler-eval] shard {args.shard}/{args.num_shards}: {len(seqs)} sequences", flush=True)

    records: list[dict] = []
    t0 = time.time()
    for ratio in MASK_RATIOS:
        for steps in STEP_GRID:
            arm = f"r{int(ratio * 100)}_s{steps}" + ("_sc" if args.self_correction else "")
            for start in range(0, len(seqs), args.batch):
                chunk = seqs[start:start + args.batch]
                tags = [t for t, _ in chunk]
                ids = torch.stack([v for _, v in chunk]).to(device)
                gen = torch.Generator(device=device).manual_seed(
                    args.seed * 100000 + int(ratio * 100) * 1000 + steps)
                eligible = ids.ne(0)
                mask = (torch.rand(ids.shape, device=device, generator=gen) < ratio) & eligible
                corrupted = torch.where(mask, torch.full_like(ids, MASK_TOKEN_ID), ids)
                with torch.no_grad():
                    denoised, commit_step = iterative_denoise(
                        model, corrupted, mask, steps=steps,
                        self_correction=args.self_correction,
                    )
                match = (denoised == ids) & mask
                for row in range(ids.shape[0]):
                    m = int(mask[row].sum())
                    em = float(match[row].sum()) / max(1, m)
                    tau = kendall_tau_commit_order(
                        commit_step[row:row + 1], mask[row:row + 1]) if steps > 1 else 0.0
                    residue = int((denoised[row] == MASK_TOKEN_ID).sum())
                    records.append({
                        "document_id": f"{tags[row]}:{start + row}",
                        "arm": arm,
                        "seed": args.seed,
                        "metrics": {"em": em, "tau": tau, "residue": float(residue)},
                        "isolation_mode": "none",
                        "failure": None,
                    })
            print(f"[sampler-eval] {arm}: done ({time.time() - t0:.0f}s elapsed)", flush=True)

    payload = {
        "schema": "qz_capability_sampler_shard_v1",
        "task": "sampler_reconstruction",
        "task_kind": "sampler",
        "checkpoint": str(Path(args.ckpt_dir).resolve()),
        "checkpoint_step": loaded.step,
        "registry_hash": os.environ.get("CAPABILITY_REGISTRY_HASH", "unpinned"),
        "profile_sha256": os.environ.get("CAPABILITY_PROFILE_SHA256", "unpinned"),
        "condition_profile_hash": os.environ.get("CAPABILITY_CONDITION_PROFILE_HASH", "unpinned"),
        "seeds": [args.seed],
        "metric_schema": ["em", "tau", "residue"],
        "shard_index": args.shard,
        "num_shards": args.num_shards,
        "n_records": len(records),
        "records": records,
        "grid": {"mask_ratios": list(MASK_RATIOS), "steps": list(STEP_GRID),
                 "window": args.window, "self_correction": args.self_correction},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, str(out))
    print(f"[sampler-eval] wrote {out} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
