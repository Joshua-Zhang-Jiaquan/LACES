"""v5.3 Phase 2b — latent prior training on 2.9B boundary-capture fields.

Trains ``LatentGridDiffRWKV`` (the RNN denoiser) directly on
`boundary_state_capture_v1` shards, bypassing the sealed-mixture
semantic_hierarchy.v2 pipeline: the capture IS the E_target output at the
current decision-tree state (candidate S, frozen readout), and the prior's
job is to model its distribution. Direct-2.9B lane per user directive
2026-08-25.

Geometry: capture fields [R, 64 blocks, 64, 256] -> per-row chunk grid
[outer=64, inner=8, d_z=32] via the same mean-pool + reshape as the B8 probe
(`field_to_chunk_latents`), followed by the T5 normalization (unit sphere
x sqrt(d_z) per chunk) — ONE written convention, shared with the probe and
any future downlink (v5_3_plan §1b item 5).

Context condition: the mean chunk latent of the row (document-level summary,
[d_z]) — the cheapest v1 context; revealed-slot conditioning arms come with
the slot-mask work.

Arms (one flag each, v5.3 C1–C3): --schedule {cosine,gumbel},
--objective {eps,rf}, --self-conditioning. Collapse monitors (C4) always on.
Val: held-out rows' diffusion loss at fixed seeds + NND/corr monitors +
a ẑ sample dump for the B9 MMD gate.

Usage:
  python train/train_latent_prior_capture.py --capture-dir <dir> \
      --out <run_dir> [--steps 4000] [--batch 16] [--schedule gumbel] \
      [--objective eps] [--self-conditioning] [--device cuda] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_SCALE_DIR = Path(__file__).resolve().parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

import torch  # noqa: E402

from models.boundary_field import (  # noqa: E402
    CHUNK_DZ,
    FIELD_DIM,
    FIELD_TOKENS,
    N_CHUNKS,
    field_rows_to_grids,
)
from models.latent_grid_diffrwkv import LatentGridConfig, LatentGridDiffRWKV  # noqa: E402
from models.state_hijacking_collapse_monitors import (  # noqa: E402
    nnd_summary,
    norm_loss_correlation,
)

def _load_capture(capture_dir: Path, max_rows: int) -> torch.Tensor:
    shards = sorted(capture_dir.glob("boundary_tokens_*.pt"))
    if not shards:
        msg = f"no boundary_tokens_*.pt shards in {capture_dir}"
        raise FileNotFoundError(msg)
    fields: list[torch.Tensor] = []
    total = 0
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        fields.append(payload["fields"])
        total += payload["fields"].shape[0]
        if total >= max_rows:
            break
    return torch.cat(fields, dim=0)[:max_rows]


def main() -> int:  # noqa: PLR0915
    """Train the prior on capture grids; write ckpt + metrics + ẑ dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--holdout-rows", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--schedule", choices=("cosine", "gumbel"), default="cosine")
    parser.add_argument("--objective", choices=("eps", "rf"), default="eps")
    parser.add_argument("--self-conditioning", action="store_true")
    parser.add_argument("--self-cond-rate", type=float, default=0.25)
    parser.add_argument("--zhat-samples", type=int, default=64,
                        help="rows of prior rollouts dumped for the B9 MMD gate")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.capture_dir is None:
        parser.error("--capture-dir required unless --smoke")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.smoke:
        rows, outer = 24, 4
        fields = torch.randn(rows, outer, FIELD_TOKENS, FIELD_DIM, dtype=torch.float16)
        args.steps = min(args.steps, 30)
        args.holdout_rows = 4
        args.zhat_samples = 4
        args.sample_steps = 4
        args.val_every = 10
        hidden, depth = 32, 2
    else:
        fields = _load_capture(args.capture_dir, args.max_rows)
        outer = fields.shape[1]
        hidden, depth = args.hidden_size, args.depth

    grids = field_rows_to_grids(fields)                 # [R, outer, 8, 32]
    del fields
    holdout = args.holdout_rows
    val_grids, train_grids = grids[:holdout], grids[holdout:]
    if train_grids.shape[0] < args.batch:
        msg = "not enough rows for one batch"
        raise ValueError(msg)

    config = LatentGridConfig(
        outer_slots=outer,
        inner_slots=N_CHUNKS,
        latent_dim=CHUNK_DZ,
        hidden_size=hidden,
        depth=depth,
        schedule=args.schedule,
        objective=args.objective,
        self_conditioning=args.self_conditioning,
        self_cond_rate=args.self_cond_rate,
    )
    model = LatentGridDiffRWKV(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    full_mask = torch.ones(args.batch, outer, N_CHUNKS, dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    z_norms: list[float] = []
    val_curve: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(args.steps):
        index = torch.randint(0, train_grids.shape[0], (args.batch,), generator=generator)
        z_star = train_grids[index].to(device)
        context = z_star.mean(dim=(1, 2))
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.diffusion_loss(z_star, context, full_mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        z_norms.append(float(z_star.norm()))
        if step % args.val_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                torch.manual_seed(7)  # fixed val corruption for comparability
                vb = val_grids.to(device)
                vmask = torch.ones(vb.shape[0], outer, N_CHUNKS, dtype=torch.bool,
                                   device=device)
                vloss, vmetrics = model.diffusion_loss(vb, vb.mean(dim=(1, 2)), vmask)
            torch.manual_seed(args.seed + step + 1)
            model.train()
            entry = {
                "step": step,
                "train_loss": losses[-1],
                "val_loss": float(vloss),
                "val_eps_loss": float(vmetrics["epsilon_loss"]),
            }
            if "scheduler_loss" in vmetrics:
                entry["val_scheduler_loss"] = float(vmetrics["scheduler_loss"])
            val_curve.append(entry)
            print(f"[step {step}] train={losses[-1]:.4f} val={float(vloss):.4f} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)  # noqa: T201

    monitors: dict[str, float] = {
        "norm_loss_corr": norm_loss_correlation(
            torch.tensor(z_norms), torch.tensor(losses)
        ),
    }
    monitors.update(nnd_summary(train_grids[: min(64, train_grids.shape[0])]))

    # ---- ẑ rollout dump for the B9 MMD gate ----
    model.eval()
    n_dump = min(args.zhat_samples, val_grids.shape[0])
    context = val_grids[:n_dump].to(device).mean(dim=(1, 2))
    zhat, _ = model.sample(
        context, segment_count=outer * N_CHUNKS, steps=args.sample_steps, seed=args.seed
    )
    torch.save({"chunks": zhat.reshape(-1, CHUNK_DZ).cpu()}, args.out / "zhat_dump.pt")
    torch.save({"chunks": val_grids[:n_dump].reshape(-1, CHUNK_DZ)},
               args.out / "z_holdout_dump.pt")
    torch.save({"model": model.state_dict(), "config": vars(args) | {"outer": outer}},
               args.out / "prior_ckpt.pt")

    report = {
        "schema": "latent_prior_capture_report_v1",
        "arms": {
            "schedule": args.schedule,
            "objective": args.objective,
            "self_conditioning": bool(args.self_conditioning),
        },
        "steps": args.steps,
        "final_train_loss": losses[-1],
        "val_curve": val_curve,
        "collapse_monitors": monitors,
        "rows": int(grids.shape[0]),
        "geometry": {"outer": outer, "inner": N_CHUNKS, "d_z": CHUNK_DZ,
                     "hidden": hidden, "depth": depth},
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "capture_dir": str(args.capture_dir) if args.capture_dir else None,
    }
    (args.out / "prior_report.json").write_text(
        json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"PRIOR TRAINED: final={losses[-1]:.4f} "  # noqa: T201
          f"corr={monitors['norm_loss_corr']:.3f} nnd={monitors.get('nnd_mean', 0):.3f} "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
