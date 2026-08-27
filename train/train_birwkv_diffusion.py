"""Token-level masked-diffusion training for BiRWKV7ForMaskedDiffusion.

The authentic-diffusion-LM trainer (code/plan sections 4.1-4.5):
  - absorbing-mask corruption with the noise-level mixture
    (30% ratio 0.05-0.30, 40% 0.30-0.70, 30% 0.70-1.00), applied per
    256-token block so one 4096-token sample spans several noise levels;
  - corruption mixture v0: uniform-token or contiguous-span per block;
  - masked-CE objective, token-normalized with the global-ratio DDP rule
    (local CE sum * world_size / global masked count);
  - causal-replay aux loss (forward-only stream, next-token CE) at
    schedule weight lambda_c to protect pretrained knowledge;
  - in-loop sampler metrics: high-mask reconstruction, 1/8/16/32-step
    comparison, commit-order Kendall tau.

Distributed: FSDP FULL_SHARD (use_orig_params), bf16 mixed precision with
fp32 master weights. Checkpoints are written atomically (tmp dir + rename)
every --save-every steps; never stop the job between "CKPT BEGIN" and
"CKPT DONE" log lines.

Launch (per host, via rendezvous_gpfs.sh or torchrun --standalone):
  torchrun --standalone --nproc_per_node=8 scale/train/train_birwkv_diffusion.py \
      --model-dir base_models/rwkv7-0.4B-world \
      --token-dir .../fineweb_4096_packed_full \
      --save-root .../outputs_birwkv_diff_0p4b --run-name pilot-0p4b \
      --steps 4000 --microbatch 4 --grad-accum 2
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

_SCALE_DIR = Path(__file__).resolve().parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

from data.fineweb4096_packed import FineWebPackedPickleDataset  # noqa: E402
from data.fineweb4096_sampler import ShardContiguousSampler  # noqa: E402
from models.birwkv7_diffusion import (  # noqa: E402
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    BiRWKV7Block,
    BiRWKV7ForMaskedDiffusion,
    iterative_denoise,
    kendall_tau_commit_order,
)
from models.boundary_field import CAPTURE_BLOCK_SIZE, capture_z_slots  # noqa: E402
from models.latent_plan import (  # noqa: E402
    PlanEncoder,
    PlanPredictor,
    derange,
    distribution_keep_loss,
    latent_align_loss,
    make_state_cache,
    plan_contrastive_loss,
    read_state_cache,
    state_norm_features,
)
from models.state_hijacking_cache import inject_predicted_states  # noqa: E402

# noise-level mixture (code/plan section 4.1)
NOISE_BUCKETS = ((0.05, 0.30, 0.30), (0.30, 0.70, 0.40), (0.70, 1.00, 0.30))
BUCKET_NAMES = ("low", "med", "high")
_TAU_EVAL_STEPS = 16
_MATRIX_NDIM = 2


# Steps at which --dump-states writes a shard. Deliberately SPARSE and spread across
# training: one early, one mid, one late. 21 MB/sample bf16 means a dense dump would
# fill GPFS, and states from a single step would understate variation across training.
DUMP_STATE_STEPS: frozenset[int] = frozenset((150, 300, 600))


def _dump_captured_states(
    outdir: Path, step: int, rank: int, captured: list[torch.Tensor],
    ids: torch.Tensor, mask: torch.Tensor, source: str,
) -> None:
    """Persist one shard of real captured recurrent states for the retention gate.

    ``read_state_cache`` returns a LAYER-first list of ``[B,40,64,64]``. The Step-5
    encoder (``StateTokenConditioner``) validates BATCH-first ``[B,32,40,64,64]``, so the
    stack is on ``dim=1``. That axis order is the documented trap for this pair; getting
    it wrong yields a shape that still has the right element count for B==32.

    ``ids``/``mask`` travel with the state because the retention gate needs a REAL target
    to predict. The first attempt used a synthetic near-constant target (relative std
    3.2e-04) and its R^2 was therefore meaningless.
    """
    state = torch.stack(captured, dim=1)
    if state.shape[1] != len(captured):
        msg = f"expected layer axis at dim=1, got {tuple(state.shape)}"
        raise RuntimeError(msg)
    outdir.mkdir(parents=True, exist_ok=True)
    dpath = outdir / f"states_step{step:06d}_rank{rank}.pt"
    torch.save(
        {
            "state": state.to(torch.bfloat16).cpu(),
            "ids": ids.cpu(),
            "mask": mask.cpu(),
            "source": source,
            "step": step,
            "layer_axis": 1,
        },
        dpath,
    )
    log(f"[step {step}] DUMPED states {tuple(state.shape)} source={source} -> {dpath}")


def log(msg: str) -> None:
    """Log with a timestamp on rank 0 only."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)  # noqa: T201


# ----------------------------------------------------------------------
# Corruption
# ----------------------------------------------------------------------
def sample_corruption(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
    span_prob: float,
    generator: torch.Generator,
    gen_prob: float = 0.0,
    docgen_prob: float = 0.0,
    doc_spans: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(sample, block) noise-bucket corruption + optional generation lane.

    With probability ``gen_prob`` a row uses FREE-GENERATION corruption
    instead: a clean prefix (uniform 5-60% of the sequence) followed by an
    all-masked contiguous canvas to the end. This is the corruption shape of
    prompt->completion generation, which block-local bucket masking never
    produces (Stage-C finding: strong infill, no synthesis).

    With probability ``docgen_prob`` a row uses DOC-GENERATION corruption: pick ONE
    packed document and mask exactly its completion span
    (``prompt_ends[k] .. doc_ends[k]``), leaving that document's own prompt and all
    ~65 sibling documents visible. This is the targeted fix for ``gen_prob``: a
    4096-token row packs ~66 short SFT docs (measured mean 66.2), so masking "a
    random prefix to the end of the row" covers ~40-57 UNRELATED documents and never
    shows the model "one prompt -> its own completion" -- the shape HumanEval/MBPP
    ask for. ``--gen-prob 0.3`` was measured to be a no-op on every axis (code 0.0
    unchanged, LM ppl 21/21 n.s., RACE identical), consistent with that diagnosis.

    ``doc_spans`` is ``(doc_starts, prompt_ends, doc_ends)``, each ``[B, D]`` int64
    padded with -1, from the codegen corpus sidecar. Rows without usable spans keep
    bucket corruption, so mixing corpora is safe.

    Returns (corrupted_ids, mask [B,T] bool, bucket_idx [B, n_blocks] long,
    gen_rows [B] bool, block_t [B, n_blocks] float) where ``block_t`` is the REALISED
    per-block masked fraction -- the v5.2 B3 per-block timestep coordinate.
    """
    bsz, seq = input_ids.shape
    device = input_ids.device
    n_blocks = (seq + block_size - 1) // block_size

    probs = torch.tensor([b[2] for b in NOISE_BUCKETS], device=device)
    bucket = torch.multinomial(
        probs.expand(bsz * n_blocks, -1), 1, generator=generator
    ).view(bsz, n_blocks)
    lo = torch.tensor([b[0] for b in NOISE_BUCKETS], device=device)[bucket]
    hi = torch.tensor([b[1] for b in NOISE_BUCKETS], device=device)[bucket]
    ratio = lo + (hi - lo) * torch.rand(bsz, n_blocks, device=device, generator=generator)

    eligible = attention_mask.bool() & input_ids.ne(PAD_TOKEN_ID)
    mask = torch.zeros_like(eligible)
    use_span = torch.rand(bsz, n_blocks, device=device, generator=generator) < span_prob
    rnd = torch.rand(bsz, seq, device=device, generator=generator)

    for blk in range(n_blocks):
        s, e = blk * block_size, min((blk + 1) * block_size, seq)
        r = ratio[:, blk].unsqueeze(1)
        blk_mask = rnd[:, s:e] < r
        # contiguous span variant: mask positions s+off .. s+off+span_len
        width = e - s
        span_len = (r.squeeze(1) * width).long().clamp(1, width)
        off = (
            torch.rand(bsz, device=device, generator=generator) * (width - span_len + 1).float()
        ).long()
        pos = torch.arange(width, device=device).unsqueeze(0)
        span_mask = (pos >= off.unsqueeze(1)) & (pos < (off + span_len).unsqueeze(1))
        blk_mask = torch.where(use_span[:, blk].unsqueeze(1), span_mask, blk_mask)
        mask[:, s:e] = blk_mask & eligible[:, s:e]

    gen_rows = torch.rand(bsz, device=device, generator=generator) < gen_prob
    if gen_rows.any():
        prefix_frac = 0.05 + 0.55 * torch.rand(bsz, device=device, generator=generator)
        prefix_len = (prefix_frac * seq).long().clamp(min=16)
        pos = torch.arange(seq, device=device).unsqueeze(0)
        gen_mask = (pos >= prefix_len.unsqueeze(1)) & eligible
        mask = torch.where(gen_rows.unsqueeze(1), gen_mask, mask)
        # label gen rows as high-noise for bucket diagnostics
        bucket = torch.where(
            gen_rows.unsqueeze(1), torch.full_like(bucket, len(NOISE_BUCKETS) - 1), bucket
        )

    # --- DOC-GENERATION lane: mask exactly one packed document's completion. ---
    if docgen_prob > 0.0 and doc_spans is not None:
        doc_starts, prompt_ends, doc_ends = doc_spans
        valid = (doc_starts >= 0) & (prompt_ends >= 0) & (doc_ends > prompt_ends)
        n_valid = valid.sum(dim=1)
        docgen_rows = (
            torch.rand(bsz, device=device, generator=generator) < docgen_prob
        ) & (n_valid > 0)
        if docgen_rows.any():
            weights = valid.float()
            # rows with no valid doc would make multinomial throw; give them a dummy
            # uniform row and discard the draw via docgen_rows/apply below.
            weights = torch.where(
                (n_valid > 0).unsqueeze(1), weights, torch.ones_like(weights)
            )
            choice = torch.multinomial(weights, 1, generator=generator).squeeze(1)
            rows = torch.arange(bsz, device=device)
            lo_sel = prompt_ends[rows, choice].clamp(min=0)
            hi_sel = doc_ends[rows, choice].clamp(max=seq)
            pos = torch.arange(seq, device=device).unsqueeze(0)
            doc_mask = (pos >= lo_sel.unsqueeze(1)) & (pos < hi_sel.unsqueeze(1)) & eligible
            # require a non-empty span after intersecting eligible, so padding-only
            # tails cannot produce a zero-token "generation" row
            apply = docgen_rows & doc_mask.any(dim=1)
            mask = torch.where(apply.unsqueeze(1), doc_mask, mask)
            bucket = torch.where(
                apply.unsqueeze(1), torch.full_like(bucket, len(NOISE_BUCKETS) - 1), bucket
            )
            # report docgen rows as generation rows so existing diagnostics and the
            # gen-lane loss weighting see them with no further plumbing
            gen_rows = gen_rows | apply

    corrupted = torch.where(mask, torch.full_like(input_ids, MASK_TOKEN_ID), input_ids)
    # Per-block REALISED mask fraction, [B, n_blocks] float in [0,1]. This is the v5.2 B3
    # per-block `t` coordinate and it replaces the sampled `ratio` drawn above, which was
    # computed and then DISCARDED here (trainer line ~177 pre-2026-08-18).
    #
    # It must be the realised fraction, not the draw, for three reasons:
    #   * the span variant masks a contiguous run whose length is quantised, so the
    #     achieved fraction differs from `r`;
    #   * `eligible` removes PAD and out-of-attention positions, lowering it further;
    #   * the gen and docgen lanes OVERWRITE `mask` wholesale after the draw (and force
    #     `bucket` to the high bin), so the draw describes a corruption that no longer
    #     exists on those rows.
    # Conditioning on the draw would hand the model a number that disagrees with the
    # canvas it can see -- the same class of mismatch that made the corrupted-state
    # injection actively harmful in Phase 0b.
    blk_elig = eligible.float()
    blk_mask_f = mask.float()
    pad_to = n_blocks * block_size
    if pad_to != seq:
        pad = pad_to - seq
        blk_elig = functional.pad(blk_elig, (0, pad))
        blk_mask_f = functional.pad(blk_mask_f, (0, pad))
    elig_per_blk = blk_elig.view(bsz, n_blocks, block_size).sum(dim=2)
    mask_per_blk = blk_mask_f.view(bsz, n_blocks, block_size).sum(dim=2)
    block_t = mask_per_blk / elig_per_blk.clamp_min(1.0)
    return corrupted, mask, bucket, gen_rows, block_t


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def masked_diffusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    bucket: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Global-ratio token-normalized masked CE + per-bucket diagnostics."""
    ce = functional.cross_entropy(
        logits.view(-1, logits.shape[-1]).float(), targets.view(-1), reduction="none"
    ).view_as(targets)
    ce = ce * mask

    local_sum = ce.sum()
    local_cnt = mask.sum().to(torch.float32)
    if dist.is_initialized():
        stats = torch.stack([local_sum.detach(), local_cnt])
        dist.all_reduce(stats)
        global_cnt = stats[1].clamp_min(1.0)
        world = dist.get_world_size()
        loss = local_sum * (world / global_cnt)
        mean_ce = (stats[0] / global_cnt).item()
    else:
        loss = local_sum / local_cnt.clamp_min(1.0)
        mean_ce = loss.item()

    diag = {"mask_ce": mean_ce}
    with torch.no_grad():
        bsz, seq = targets.shape
        n_blocks = bucket.shape[1]
        for bi, name in enumerate(BUCKET_NAMES):
            bmask = torch.zeros_like(mask)
            for blk in range(n_blocks):
                s, e = blk * block_size, min((blk + 1) * block_size, seq)
                bmask[:, s:e] = mask[:, s:e] & (bucket[:, blk] == bi).unsqueeze(1)
            cnt = bmask.sum()
            bucket_ce = (ce * bmask).sum().div(cnt.clamp_min(1)).item()
            diag[f"ce_{name}"] = bucket_ce if cnt > 0 else float("nan")
    return loss, diag


def causal_replay_loss(model: torch.nn.Module, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """Next-token CE on the forward-only stream (pretrained-knowledge anchor)."""
    logits = model(input_ids, force_forward=True)
    tgt = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool() & tgt.ne(PAD_TOKEN_ID)
    ce = functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="none"
    ).view_as(tgt)
    return (ce * valid).sum() / valid.sum().clamp_min(1)


def joint_commit_loss(
    model: torch.nn.Module,
    ids: torch.Tensor,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator,
    group_size: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """v5.2 B4 / Q1 U3b: teach the model to consume its own within-step commits.

    ``v4_plan`` section 2.4 requires that a grouped commit be "a Joint-Commit RWKV over an
    ordered token group (never a product of independent per-position marginals)". The
    trained objective has only ever been per-position masked CE, so nothing in training
    ever asked the model to condition on tokens *it committed in the same step*. The
    sampler does exactly that, which is the train/sample mismatch this loss closes.

    Construction, per row:

    1. Pick an ordered group of up to ``group_size`` masked positions (left-to-right, which
       matches nothing the sampler does -- see the note below).
    2. REVEAL the first ``j`` of them by writing their TRUE ids into the canvas.
    3. Score CE on member ``j+1`` only, from that partially-revealed canvas.

    The revealed prefix uses **true** ids, not model predictions. That is teacher forcing,
    and it is the deliberate choice: sampling the model's own commits here would make the
    target distribution depend on the current policy and reintroduce the error-accumulation
    dynamic that cost 23% em at r90/s1 in the Phase-1a infilling sweep. Closing the
    conditioning gap and closing the error-accumulation gap are separate problems; this
    loss addresses only the first, and says so.

    Order note: members are taken in position order rather than confidence order because a
    confidence ranking would require a forward pass to compute, doubling this loss's cost,
    and because the sampler's own order varies with the canvas. Position order gives the
    model *practice at conditioning on same-step commits* without pretending to reproduce
    the sampler's exact ordering. If the pilot's gate (ii) fails -- fast mode not matching
    exact mode -- confidence-ordered grouping is the pre-registered escalation.

    Returns ``(loss, diag)``; the loss is 0.0 with an empty diag when no row has at least
    two masked positions, which is the only case where the construction is undefined.
    """
    bsz, seq = ids.shape
    device = ids.device
    j = max(1, group_size // 2)  # reveal half the group, score the next member

    # Positions to reveal, and the single scored position, per row.
    reveal = torch.zeros_like(mask)
    score = torch.zeros_like(mask)
    n_used = 0
    # Draw all window offsets in ONE kernel. A per-row `randint` on a CUDA generator forces
    # a device sync per row, i.e. `bsz` syncs every microbatch, which would dominate this
    # loss's cost. Uniform in [0,1) is scaled per row below because each row has a
    # different number of masked positions.
    offs = torch.rand(bsz, device=device, generator=generator)
    for row in range(bsz):
        pos = mask[row].nonzero(as_tuple=True)[0]
        if pos.numel() < 2:
            continue
        span = min(group_size, int(pos.numel()))
        max_start = int(pos.numel()) - span
        start = int(offs[row].item() * (max_start + 1)) if max_start > 0 else 0
        start = min(start, max_start)
        grp = pos[start:start + span]
        k = min(j, int(grp.numel()) - 1)  # leave at least one member to score
        reveal[row, grp[:k]] = True
        score[row, grp[k]] = True
        n_used += 1

    if n_used == 0:
        return ids.new_zeros((), dtype=torch.float32), {}

    # Canvas with the group's prefix revealed to its TRUE values.
    canvas = torch.where(reveal, ids, corrupted)
    logits = model(canvas, False).float()  # noqa: FBT003
    b_idx, t_idx = score.nonzero(as_tuple=True)
    ce = functional.cross_entropy(
        logits[b_idx, t_idx], ids[b_idx, t_idx], reduction="mean"
    )
    diag = {
        "jc_rows": float(n_used) / float(bsz),
        "jc_revealed": float(int(reveal.sum())) / float(max(1, n_used)),
        "jc_ce": float(ce.detach()),
    }
    return ce, diag


def lambda_c_schedule(
    tokens_seen: float, warm: float = 0.30, main: float = 0.10, warm_tokens: float = 0.5e9
) -> float:
    """Return the causal-replay weight for the current token count."""
    return warm if tokens_seen < warm_tokens else main


# ----------------------------------------------------------------------
# Staged unfreeze (code/plan Phase 2): grad-nulling keyed on param names.
# Stage A (< stage_a frac): only reverse attention + fusion gates train.
# Stage B (< stage_b frac): + forward branch of the TOP HALF of layers.
# Stage C: everything. Implemented by nulling frozen grads before the
# optimizer step -- robust under FSDP use_orig_params (no requires_grad
# flips after wrapping).
# ----------------------------------------------------------------------
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def grad_frozen(name: str, frac: float, n_layers: int, stage_a: float, stage_b: float) -> bool:
    """Return True if this parameter's gradient should be nulled at ``frac``."""
    if stage_b <= 0 or frac >= stage_b:
        return False
    if "attn_bwd" in name or "fuse_" in name:
        return False
    if frac >= stage_a:
        m = _LAYER_RE.search(name)
        if m is not None and int(m.group(1)) >= n_layers // 2:
            return False
    return True


# ----------------------------------------------------------------------
# Checkpointing (atomic; lesson from job-ad95cb44)
# ----------------------------------------------------------------------
def save_checkpoint(  # noqa: PLR0913
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: float,
    save_dir: Path,
    rank: int,
    keep_last: int = 3,
    plan_modules: dict[str, torch.nn.Module] | None = None,
    plan_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write one atomic full-state checkpoint and prune old ones on rank 0.

    ``plan_modules`` holds the replicated latent encoder/predictor and
    ``plan_optimizer`` their (separate, non-FSDP) optimizer. Neither is inside the
    FSDP wrap, so both would be silently dropped from ``model.pt``/``optim.pt`` —
    and a Gate-D0 checkpoint without its predictor cannot produce ``Ẑ``, which
    would make the run unevaluable. They go to ``latent.pt``.
    """
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)

    if rank == 0:
        log(f"CKPT BEGIN step={step}")
        tmp = save_dir / f"step_{step:08d}.tmp"
        final = save_dir / f"step_{step:08d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(model_state, tmp / "model.pt")
        torch.save(optim_state, tmp / "optim.pt")
        if plan_modules:
            latent_blob: dict[str, object] = {
                k: m.state_dict() for k, m in plan_modules.items()
            }
            if plan_optimizer is not None:
                latent_blob["plan_optimizer"] = plan_optimizer.state_dict()
            torch.save(latent_blob, tmp / "latent.pt")
        (tmp / "meta.json").write_text(json.dumps({"step": step, "tokens_seen": tokens_seen}))
        tmp.rename(final)
        log(f"CKPT DONE step={step} -> {final}")
        olds = sorted(p for p in save_dir.glob("step_????????") if p.is_dir())
        for p in olds[:-keep_last]:
            shutil.rmtree(p, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()


def find_resume(save_dir: Path) -> Path | None:
    """Return the newest complete checkpoint directory, if any."""
    dirs = sorted(p for p in save_dir.glob("step_????????") if (p / "model.pt").exists())
    return dirs[-1] if dirs else None


# ----------------------------------------------------------------------
# In-loop sampler eval
# ----------------------------------------------------------------------
@torch.no_grad()
def sampler_eval(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    max_len: int = 1024,
    mask_ratio: float = 0.7,
) -> dict[str, float]:
    """Denoise one batch at 1/8/16/32 steps and report accuracy diagnostics."""
    was_training = model.training
    model.eval()
    ids = batch["input_ids"][:2, :max_len].to(device)
    am = batch["attention_mask"][:2, :max_len].to(device)
    gen = torch.Generator(device=device).manual_seed(1234)
    eligible = am.bool() & ids.ne(PAD_TOKEN_ID)
    mask = (torch.rand(ids.shape, device=device, generator=gen) < mask_ratio) & eligible
    corrupted = torch.where(mask, torch.full_like(ids, MASK_TOKEN_ID), ids)

    out: dict[str, float] = {}
    for steps in (1, 8, 16, 32):
        denoised, commit_step = iterative_denoise(model, corrupted, mask, steps=steps)
        acc = ((denoised == ids) & mask).sum().item() / max(1, mask.sum().item())
        out[f"em@{steps}"] = acc
        if steps == _TAU_EVAL_STEPS:
            out["tau@16"] = kendall_tau_commit_order(commit_step, mask)
            out["mask_residue"] = (denoised == MASK_TOKEN_ID).sum().item()
    if was_training:
        model.train()
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Parse args, build the FSDP trainer, and run the diffusion loop."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--token-dir", required=True)
    p.add_argument("--save-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--block-t-cond", action="store_true",
                   help="attach BlockTimestepConditioner and feed the REALISED per-block mask "
                        "fraction as an additive embedding (v5.2 B3 / Q1 U3a). Zero-init, so "
                        "attaching it alone does not change the model's output.")
    p.add_argument("--block-t-freqs", type=int, default=8,
                   help="Fourier features for the timestep embedding")
    p.add_argument("--block-t-scale", type=float, default=0.5,
                   help="tanh bound on the additive timestep term")
    p.add_argument("--lambda-joint-commit", type=float, default=0.0,
                   help="weight on L_JointCommit (v5.2 B4 / Q1 U3b): reveal the first half of "
                        "an ordered masked group with TRUE ids, score CE on the next member. "
                        "0.0 disables it, which is the default.")
    p.add_argument("--joint-commit-group", type=int, default=4,
                   help="ordered group size for L_JointCommit")
    p.add_argument("--joint-commit-every", type=int, default=2,
                   help="run L_JointCommit every N microbatches (it costs one extra forward)")
    p.add_argument("--span-prob", type=float, default=0.15)
    p.add_argument("--p-self-cond", type=float, default=0.0,
                   help="x-hat-0 feedback self-conditioning (v5.3 C5 / PT T4): with this "
                        "probability per microbatch, run a no-grad forward, write its argmax "
                        "into the masked positions of a second canvas, and take the loss on "
                        "the second forward. Half of the firing steps keep the canvas "
                        "unmodified (the zeroed-channel arm of the standard recipe; "
                        "identical to a plain step, so the extra forward is skipped). "
                        "0.0 disables it, which is the default.")
    p.add_argument("--gen-prob", type=float, default=0.0,
                   help="fraction of rows using free-generation corruption "
                        "(clean prefix + all-masked tail) instead of block buckets")
    p.add_argument("--docgen-prob", type=float, default=0.0,
                   help="fraction of rows using DOC-GENERATION corruption: mask exactly one "
                        "packed document's completion span (prompt_end..doc_end), leaving its "
                        "own prompt and all sibling docs visible. Needs a corpus with the "
                        "doc-boundary sidecar; rows without spans fall back to bucket masking.")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lambda-c-warm", type=float, default=0.30)
    p.add_argument("--lambda-c-main", type=float, default=0.10)
    p.add_argument("--gate-bias-init", type=float, default=4.0)
    p.add_argument("--force-forward", action="store_true",
                   help="forward-only control arm: reverse stream disabled")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--stage-a-frac", type=float, default=0.0,
                   help="until this fraction of steps, train only reverse attn + fusion gates")
    p.add_argument("--stage-b-frac", type=float, default=0.0,
                   help="until this fraction, additionally train top-half forward layers; "
                        "0 disables staged unfreeze entirely")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--sampler-every", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume-from", default=None,
                   help="external step_* dir for a weights-only warm start "
                        "(fresh optimizer/schedule; ignored when this run's own "
                        "save_dir already has checkpoints)")
    # ---- latent conditioning (v4_plan Plan-1 M0, Gate D0) ----
    p.add_argument("--latent-cond", choices=("none", "film", "softprefix", "crossattn"),
                   default="none",
                   help="Gate-D0 downlink arm: 'film' = A1 PerLayerFiLM (exact identity at "
                        "init), 'softprefix' = A3 soft plan tokens. 'none' trains the "
                        "unconditioned baseline.")
    p.add_argument("--latent-dim", type=int, default=32, help="d_z, latent slot width")
    p.add_argument("--latent-slots", type=int, default=8, help="H_z, number of typed slots")
    p.add_argument("--latent-prefix", type=int, default=8,
                   help="soft prefix length (--latent-cond softprefix only)")
    p.add_argument("--latent-out-scale", type=float, default=0.5,
                   help="FiLM gamma bound: gamma in [-s, s] via s*tanh(.)")
    # ---- Phase-3 state-derived Z source (v5_3_plan §4 Phase 3) ----
    p.add_argument("--latent-z-source", choices=("plan", "state"), default="plan",
                   help="'plan' = legacy token-derived Z (PlanEncoder/PlanPredictor; the "
                        "D0 scope-error lane, kept for comparability). 'state' = chunked "
                        "block-latent field captured from the model's OWN forward "
                        "recurrence through the frozen StateTokenConditioner "
                        "(boundary_field.capture_z_slots): H=64 blocks x n=8 chunks = "
                        "512 slots x d_z=32. Requires --latent-cond film.")
    p.add_argument("--latent-prior-ckpt", default=None,
                   help="frozen LatentGridDiffRWKV prior_ckpt.pt (Phase-2b winner). "
                        "Enables the zhat val arm: context field captured from the "
                        "corrupted canvas, repaired by the prior (sample_repair), fed "
                        "as the deployable condition. Optional; without it the val "
                        "loop reports oracle/deranged/zero arms only.")
    p.add_argument("--z-noise-sigma", type=float, default=0.0,
                   help="train-time Gaussian noise std on the oracle z_star (state "
                        "source only) — set to the prior's held-out zhat RMSE so the "
                        "FiLM trains on the error distribution it sees at inference")
    p.add_argument("--z-drop", type=float, default=0.0,
                   help="train-time condition dropout prob (state source only): the "
                        "micro trains unconditioned, the CFG hook")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="train ONLY the latent conditioner (backbone requires_grad "
                        "False, excluded from the optimizer). Makes LM non-erosion "
                        "structural and isolates the downlink question.")
    p.add_argument("--readout-seed", type=int, default=20260825,
                   help="seed for the FROZEN StateTokenConditioner readout (must match "
                        "the Phase-2a capture / Phase-2b prior convention)")
    p.add_argument("--lambda-z-align", type=float, default=1.0,
                   help="weight on the Ẑ->Z alignment loss")
    p.add_argument("--lambda-keep", type=float, default=0.1,
                   help="weight on KL[p_base || p_conditioned] (distribution preservation)")
    p.add_argument("--latent-lr-mult", type=float, default=10.0,
                   help="LR multiplier for latent params (adapter-first: the backbone is "
                        "warm-started, the adapters are not)")
    p.add_argument("--keep-every", type=int, default=4,
                   help="compute the distribution-keep KL every N microbatches (it needs an "
                        "extra unconditioned forward, so every step would cost ~33%%)")
    # Puts gradient directly on the quantity Gate D0.2 measures. The gain sweep
    # showed the own-vs-wrong-plan margin (s) is ~9x smaller than the generic cost
    # of conditioning (h), and that amplifying the modulation widens h faster than
    # s -- so the missing ingredient is a term that rewards discrimination, not
    # more amplitude. Costs one extra forward every contrast_every microbatches.
    # Arm C (v4_plan section 5.3): inject the FULL per-layer recurrent state as the
    # condition. This is the ABLATION CEILING -- it upper-bounds every compressed
    # E_i: H_i -> Z arm, because it hands the denoiser the whole state rather than a
    # 256-number summary. If C cannot beat its own shuffle control, no compressed arm
    # can, and the conditioning direction closes.
    #
    # The state is captured from CLEAN ids, so it is ORACLE by construction. Per
    # prohibition P2 this arm is ablation-only and must never appear in a headline
    # number; per P1 the predicted-Zhat requirement applies to the compressed arms.
    p.add_argument("--state-cond", choices=("none", "full"), default="none",
                   help="arm C: inject the full recurrent state (ablation ceiling)")
    # WHERE the injected state is captured from. This is the difference between a
    # ceiling and a deployable arm:
    #   clean     -> capture on the true ids. ORACLE: the state encodes the answer, so
    #                the result is an ablation ceiling only (P1/P2).
    #   corrupted -> capture on the same masked canvas the model is conditioned on. No
    #                oracle information can enter, so this is what a real E_i: H_i -> Z
    #                would have access to.
    # Measured motivation: with a clean-trained conditioner, injecting a
    # corrupted-canvas state at eval time scored -0.3100 nats [CI -0.3200,-0.2994] --
    # WORSE than injecting zeros (exactly 0.0) and worse than another document's clean
    # state (+0.1203). Being worse than zeros means the input is not weakly informative
    # but actively MISLEADING, i.e. distribution mismatch rather than absent signal.
    # Training on the corrupted capture removes that mismatch and is the only way to
    # tell the two apart.
    p.add_argument("--state-source", choices=("clean", "corrupted"), default="clean",
                   help="capture the injected state from the clean ids (oracle ceiling) "
                        "or the corrupted canvas (deployable, no oracle)")
    p.add_argument("--dump-states", type=str, default="",
                   help="dir to persist real captured states at DUMP_STATE_STEPS "
                        "(for the Step-5 retention gate; ~21 MB/sample bf16)")
    p.add_argument("--state-blend", type=float, default=1.0,
                   help="1.0 = replace the state outright; <1 convex-blends with the "
                        "model's own (blend_predicted_states semantics)")
    p.add_argument("--lambda-contrast", type=float, default=0.0,
                   help="weight on the own-plan-vs-wrong-plan hinge (0 disables)")
    p.add_argument("--contrast-margin", type=float, default=0.05,
                   help="nats of own-vs-wrong margin at which the hinge saturates; "
                        "defaults to the Gate D0.1 threshold")
    p.add_argument("--contrast-every", type=int, default=2,
                   help="apply the contrastive term every N microbatches")
    args = p.parse_args()

    # ---- Phase-3 state-Z lane validation (fail at parse, not at step 1) ----
    if args.latent_z_source == "state":
        if args.latent_cond != "film":
            p.error("--latent-z-source state requires --latent-cond film (P2: FiLM "
                    "downlink only; full-state injection is the ablation ceiling)")
        if args.state_cond != "none":
            p.error("--latent-z-source state and --state-cond full are mutually "
                    "exclusive (single-consumption fla cache discipline)")
        if args.p_self_cond > 0.0:
            p.error("--latent-z-source state and --p-self-cond are mutually "
                    "exclusive (both add a no-grad forward; compose in Phase 4)")
        # The chunk geometry is fixed by the frozen readout: 64 blocks x 8 chunks
        # of d_z=32. Silently accepting other values would train a downlink whose
        # slots do not correspond to captured chunks.
        if args.latent_slots != 512 or args.latent_dim != 32:  # noqa: PLR2004
            p.error("--latent-z-source state requires --latent-slots 512 "
                    "--latent-dim 32 (H=64 blocks x n=8 chunks, d_z=32)")
        if args.max_length % 64 != 0 or (args.max_length // 64) * 8 != 512:  # noqa: PLR2004
            p.error("--latent-z-source state requires --max-length 4096 (the frozen "
                    "capture geometry: 64 blocks of 64 tokens x 8 chunks = 512 slots)")
    elif args.latent_prior_ckpt or args.z_noise_sigma > 0.0 or args.z_drop > 0.0:
        p.error("--latent-prior-ckpt/--z-noise-sigma/--z-drop require "
                "--latent-z-source state")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    save_dir = Path(args.save_root) / args.run_name
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    log(f"BiRWKV masked-diffusion: run={args.run_name} world={world} "
        f"force_forward={args.force_forward}")
    log(f"model={args.model_dir} data={args.token_dir}")
    log(f"steps={args.steps} microbatch={args.microbatch} grad_accum={args.grad_accum} "
        f"global_batch={world * args.microbatch * args.grad_accum} "
        f"tokens/step={world * args.microbatch * args.grad_accum * args.max_length}")

    # ---- model (fp32 master; FSDP casts to bf16 for compute) ----
    model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
        args.model_dir, dtype=torch.float32,
        gate_bias_init=args.gate_bias_init,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    n_params = sum(x.numel() for x in model.parameters())
    n_layers = model.config.num_hidden_layers
    if args.state_cond != "none" and rank == 0:
        # Echo arm C explicitly: this is the one line that proves, after the fact,
        # that the ceiling arm was actually active. Three separate bugs this session
        # came from parameters that changed behaviour without appearing in a log.
        # The provenance note must track --state-source. Hard-coding "ORACLE ... clean
        # ids" made arm D's own boot log assert the opposite of what it was doing, and a
        # log line that contradicts the run is worse than no log line: it is the sort of
        # thing that gets quoted into a writeup months later.
        prov = ("ORACLE state from clean ids; P2 ablation-only"
                if args.state_source == "clean"
                else "DEPLOYABLE state from the corrupted canvas; no oracle access")
        log(f"STATE ARM arm={args.state_cond} source={args.state_source} "
            f"blend={args.state_blend} layers={n_layers} ({prov})")
    log(f"params={n_params / 1e9:.3f}B  mean_forward_alpha={model.mean_forward_alpha():.4f}")

    # ---- latent conditioning (v4_plan Plan-1 M0, Gate D0) ----
    # Attached BEFORE the FSDP wrap so the conditioner's params are sharded with
    # the rest. Kept OUT of the transformer_layer_cls wrap set: FiLM parameters
    # are consumed by every layer, so wrapping them separately would force a
    # gather per block.
    #
    # Seed the GLOBAL RNG here. `--seed` previously fed only the per-rank
    # corruption Generator, so every randomly-initialised module (the plan
    # encoder/predictor, and the conditioner's own trunk) drew from an unseeded
    # global RNG. Two runs with identical flags then produced z_mse 3.49 vs 4.12
    # at step 10 -- an ~18% spread on the very metrics Gate D0 is judged on,
    # while the token path reproduced to 0.0005 nats. Without this, a D0 number
    # is not reproducible and cannot be compared across arms.
    # Same value on every rank ON PURPOSE: the plan modules are replicated, so
    # their initial weights must match or the hand-written grad all-reduce would
    # be averaging gradients of different functions.
    torch.manual_seed(args.seed)
    latent_on = args.latent_cond != "none"
    # Attach the timestep arm here, i.e. BEFORE the FSDP wrap, so its parameters are
    # sharded with the rest of the model rather than replicated. Zero-init means the
    # attached-but-untrained model is numerically identical to the warm start, which is
    # what keeps every pre-timestep measurement comparable.
    if args.block_t_cond:
        _ = model.attach_block_timestep_conditioner(
            num_freqs=args.block_t_freqs,
            out_scale=args.block_t_scale,
        )
        log(f"block-t conditioner ATTACHED (freqs={args.block_t_freqs} "
            f"scale={args.block_t_scale}, zero-init => identity until trained)")
    plan_enc: PlanEncoder | None = None
    plan_pred: PlanPredictor | None = None
    if latent_on:
        _ = model.attach_latent_conditioner(
            args.latent_cond,
            latent_dim=args.latent_dim,
            num_prefix=args.latent_prefix,
            out_scale=args.latent_out_scale,
        )
        vocab = model.config.vocab_size
        if args.latent_z_source == "plan":
            plan_enc = PlanEncoder(vocab, num_slots=args.latent_slots, d_z=args.latent_dim)
            plan_pred = PlanPredictor(vocab, num_slots=args.latent_slots,
                                      d_z=args.latent_dim)
        n_lat = (
            sum(x.numel() for x in model.latent_parameters())
            + sum(x.numel() for x in (plan_enc.parameters() if plan_enc else []))
            + sum(x.numel() for x in (plan_pred.parameters() if plan_pred else []))
        )
        log(f"LATENT arm={args.latent_cond} z_source={args.latent_z_source} "
            f"d_z={args.latent_dim} H_z={args.latent_slots} "
            f"latent_params={n_lat / 1e6:.1f}M lambda_z={args.lambda_z_align} "
            f"lambda_keep={args.lambda_keep} keep_every={args.keep_every} "
            f"lambda_contrast={args.lambda_contrast} margin={args.contrast_margin}")

    # ---- Phase-3 state-Z machinery: frozen readout + optional frozen prior ----
    # The readout MUST be the same frozen module the Phase-2a capture and the
    # Phase-2b prior saw (same seed => same weights); a re-drawn readout would
    # train the downlink against a different E_target than the prior models.
    state_readout = None
    latent_prior = None
    if latent_on and args.latent_z_source == "state":
        from models.state_token_conditioner import StateTokenConditioner  # noqa: PLC0415

        torch.manual_seed(args.readout_seed)
        state_readout = StateTokenConditioner()
        state_readout.eval()
        for prm in state_readout.parameters():
            prm.requires_grad_(False)  # noqa: FBT003
        state_readout = state_readout.to(device=device, dtype=torch.bfloat16)
        log(f"STATE-Z readout FROZEN seed={args.readout_seed} "
            f"params={sum(x.numel() for x in state_readout.parameters()) / 1e3:.1f}K "
            f"z_noise_sigma={args.z_noise_sigma} z_drop={args.z_drop}")
        if args.latent_prior_ckpt:
            from models.latent_grid_diffrwkv import (  # noqa: PLC0415
                LatentGridConfig,
                LatentGridDiffRWKV,
            )

            payload = torch.load(args.latent_prior_ckpt, map_location="cpu",
                                 weights_only=True)
            pconf = payload.get("config", {})
            latent_prior = LatentGridDiffRWKV(LatentGridConfig(
                outer_slots=int(pconf.get("outer", 64)),
                inner_slots=8,
                latent_dim=32,
                hidden_size=int(pconf.get("hidden_size", 768)),
                depth=int(pconf.get("depth", 8)),
                schedule=str(pconf.get("schedule", "cosine")),
                objective=str(pconf.get("objective", "eps")),
                self_conditioning=bool(pconf.get("self_conditioning", False)),
            ))
            latent_prior.load_state_dict(payload["model"])
            latent_prior.eval()
            for prm in latent_prior.parameters():
                prm.requires_grad_(False)  # noqa: FBT003
            latent_prior = latent_prior.to(device=device, dtype=torch.bfloat16)
            log(f"LATENT PRIOR loaded from {args.latent_prior_ckpt} "
                f"(schedule={latent_prior.config.schedule} "
                f"objective={latent_prior.config.objective} "
                f"sc={latent_prior.config.self_conditioning}) — FROZEN; "
                f"zhat val arm ACTIVE")
        else:
            log("LATENT PRIOR absent — zhat val arm INACTIVE (oracle/deranged/zero "
                "arms only); pass --latent-prior-ckpt when the Phase-2b winner lands")

    # ---- Phase-3 freeze: backbone params out of the optimizer ----
    if args.freeze_backbone:
        n_frozen = 0
        for name, prm in model.named_parameters():
            if "latent_cond" not in name:
                prm.requires_grad_(False)  # noqa: FBT003
                n_frozen += prm.numel()
        log(f"BACKBONE FROZEN: {n_frozen / 1e9:.3f}B params requires_grad=False; "
            f"training latent_cond only")

    if world > 1:
        model = FSDP(
            model.to(device),
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={BiRWKV7Block}
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            use_orig_params=True,
            device_id=local_rank,
        )
    else:
        model = model.to(device=device, dtype=torch.bfloat16)

    # The plan encoder/predictor are small (tens of M) and are replicated rather
    # than sharded: they read token ids only, so every rank can hold a full copy
    # and DDP-style grad all-reduce below keeps them in sync.
    if plan_enc is not None and plan_pred is not None:
        plan_enc = plan_enc.to(device=device, dtype=torch.bfloat16)
        plan_pred = plan_pred.to(device=device, dtype=torch.bfloat16)

    latent_param_ids: set[int] = set()
    if latent_on:
        latent_param_ids = {
            id(p)
            for n, p in model.named_parameters()
            if "latent_cond" in n
        }

    decay, no_decay, cond_group = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # --freeze-backbone: frozen params stay out of the optimizer so AdamW
            # allocates no state for them (saves ~2x model size in optimizer RAM)
            # and the schedule cannot touch them even by accident.
            continue
        if id(param) in latent_param_ids:
            cond_group.append(param)
        elif param.ndim < _MATRIX_NDIM or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups: list[dict[str, object]] = [
        {"params": decay, "weight_decay": args.weight_decay, "lr_mult": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
    ]
    if cond_group:
        # Adapter-first: the backbone is warm-started and needs a gentle LR; the
        # conditioner starts at zero and needs a much larger one. The conditioner
        # IS inside the FSDP wrap, so it belongs in this optimizer.
        param_groups.append(
            {"params": cond_group, "weight_decay": 0.0, "lr_mult": args.latent_lr_mult}
        )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # The plan encoder/predictor are REPLICATED, not FSDP-sharded, so they need a
    # SEPARATE optimizer. Mixing them into the one above makes
    # FSDP.optim_state_dict raise `KeyError: Parameter containing: ...` at the
    # first checkpoint, because FSDP cannot map a parameter it does not manage
    # back to a flat-param slot. (Observed in job-405d3306 at step 10.)
    plan_optimizer: torch.optim.Optimizer | None = None
    if plan_enc is not None and plan_pred is not None:
        plan_params = [*plan_enc.parameters(), *plan_pred.parameters()]
        plan_optimizer = torch.optim.AdamW(
            [{"params": plan_params, "weight_decay": 0.0, "lr_mult": args.latent_lr_mult}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )
        log(f"plan modules on a separate optimizer ({sum(p.numel() for p in plan_params) / 1e6:.1f}M "
            f"params, replicated across ranks)")

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # ---- data: --token-dir accepts comma-separated dirs (blended by shard);
    # val is held out from the tail of the FIRST dir ----
    token_dirs = [d for d in args.token_dir.split(",") if d]

    def build_ds(dirs: list[str], max_samples_first: int | None) -> tuple[
        torch.utils.data.ConcatDataset, list[int]
    ]:
        parts, boundaries, offset = [], [], 0
        for i, d in enumerate(dirs):
            ds = FineWebPackedPickleDataset(
                data_dir=d, max_length=args.max_length,
                pad_token_id=PAD_TOKEN_ID, cache_shards=2,
                max_samples=max_samples_first if i == 0 else None,
            )
            parts.append(ds)
            boundaries.extend(offset + b for b in ds.shard_boundaries())
            offset += len(ds)
        return torch.utils.data.ConcatDataset(parts), boundaries

    probe = FineWebPackedPickleDataset(
        data_dir=token_dirs[0], max_length=args.max_length,
        pad_token_id=PAD_TOKEN_ID, cache_shards=1,
    )
    first_total = len(probe)
    train_first = first_total - args.val_samples
    if args.max_samples:
        train_first = min(train_first, args.max_samples)
    train_ds, train_boundaries = build_ds(token_dirs, train_first)
    val_ds = Subset(probe, list(range(first_total - args.val_samples, first_total)))

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }
        # Carry the codegen doc-boundary sidecar. MIXED batches are supported: rows
        # from a sidecar-less corpus (e.g. blended fineweb) are filled with -1, which
        # `sample_corruption` already treats as "no valid span" and falls back to
        # bucket corruption for that row. Requiring the sidecar on *every* row instead
        # would silently disable the docgen lane for the whole run the moment a general
        # corpus is blended in — and blending is what protects LM ability, so the two
        # requirements would otherwise be mutually exclusive.
        span_keys = ("doc_starts", "prompt_ends", "doc_ends")
        if any(k in b for b in batch for k in span_keys):
            width = next(
                b[k].shape[-1] for b in batch for k in span_keys if k in b
            )
            for key in span_keys:
                out[key] = torch.stack([
                    b[key] if key in b else torch.full((width,), -1, dtype=torch.int64)
                    for b in batch
                ])
            out["n_docs"] = torch.stack([
                b.get("n_docs", torch.tensor(0, dtype=torch.int64)) for b in batch
            ])
        return out

    train_loader = DataLoader(
        train_ds, batch_size=args.microbatch,
        sampler=ShardContiguousSampler(train_boundaries, world_size=world,
                                       rank=rank, shuffle=True, seed=args.seed),
        drop_last=True, num_workers=4, collate_fn=collate, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(val_ds, batch_size=args.microbatch, shuffle=False,
                            num_workers=0, collate_fn=collate, pin_memory=True)
    log(f"data: train={len(train_ds)} val={len(val_ds)} samples")

    # ---- resume ----
    start_step, tokens_seen = 0, 0.0

    def _merge_latent_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Backfill optional-module keys from the live model when the ckpt predates them.

        A warm-start checkpoint from before a conditioning arm existed (e.g.
        `codecpt-2p9b`) has none of that arm's entries, so a strict load raises
        `Missing key(s)`. Blanket `strict=False` would fix the symptom and also
        silence a genuinely mis-mapped backbone key, so instead take the missing
        keys from the freshly-initialised (zero-init ⇒ identity) module and
        keep the load strict. Anything else missing still raises.

        Covers BOTH optional arms:

        * ``latent_cond.*`` -- the Gate-D0 latent conditioners;
        * ``block_t_cond.*`` -- the v5.2 B3 per-block timestep arm.

        The timestep arm had to be added here. The 0.4B pilot happens to start COLD, so its
        keys never enter `from_hf_pretrained`'s strict check (the module is attached after
        that load). But any WARM start with ``--block-t-cond`` -- which is exactly how a
        2.9B U3 run begins, from `codecpt-2p9b` -- would have raised on `block_t_cond.*`
        before reaching the first step. Same class of failure as the original
        `Missing key(s) latent_cond.*` that killed the first M0 smoke arm.
        """
        prefixes = tuple(
            pre for pre, on in (("latent_cond.", latent_on),
                                ("block_t_cond.", bool(args.block_t_cond)))
            if on
        )
        if not prefixes:
            return state
        live = model.state_dict()
        added = [k for k in live if k.startswith(prefixes) and k not in state]
        if not added:
            return state
        merged = dict(state)
        for k in added:
            merged[k] = live[k].detach().clone().cpu()
        by_pre = {pre: sum(1 for k in added if k.startswith(pre)) for pre in prefixes}
        log(f"warm start: backfilled {len(added)} optional-module key(s) from fresh "
            f"zero-init {by_pre} (ckpt predates them) -> arms start as identity")
        return merged

    resume = find_resume(save_dir)
    if resume is not None:
        meta = json.loads((resume / "meta.json").read_text())
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        state = torch.load(resume / "model.pt", map_location="cpu", weights_only=True)
        state = _merge_latent_keys(state)
        if world > 1:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state)
            osd = torch.load(resume / "optim.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, osd))
        else:
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
        start_step, tokens_seen = meta["step"], meta["tokens_seen"]
        log(f"resumed from {resume} (step={start_step})")
        # Own-run resume: the plan modules live outside FSDP, so restore them too.
        lat_file = resume / "latent.pt"
        if latent_on and plan_enc is not None and plan_pred is not None:
            if lat_file.exists():
                lstate = torch.load(lat_file, map_location="cpu", weights_only=True)
                _ = plan_enc.load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in lstate["plan_encoder"].items()}
                )
                _ = plan_pred.load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in lstate["plan_predictor"].items()}
                )
                if plan_optimizer is not None and "plan_optimizer" in lstate:
                    plan_optimizer.load_state_dict(lstate["plan_optimizer"])
                    log(f"restored plan encoder/predictor + optimizer from {lat_file}")
                else:
                    log(f"restored plan encoder/predictor from {lat_file} "
                        "(no plan_optimizer state: Adam moments restart)")
            else:
                # Silently continuing with a fresh predictor would make every
                # post-resume latent metric meaningless, so refuse.
                msg = (
                    f"{lat_file} missing but --latent-cond={args.latent_cond}: resuming "
                    "would silently reset the plan encoder/predictor while keeping the "
                    "trained conditioner, making all latent metrics meaningless."
                )
                raise FileNotFoundError(msg)
    elif args.resume_from:
        # weights-only warm start from an external run (fresh optimizer, step 0)
        seed_dir = Path(args.resume_from)
        state = torch.load(seed_dir / "model.pt", map_location="cpu", weights_only=True)
        state = _merge_latent_keys(state)
        if world > 1:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state)
        else:
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
        log(f"weights-only warm start from {seed_dir} (optimizer fresh, step 0)")
        # An external warm start intentionally starts the latent modules fresh:
        # the whole point of M0 is to train them. Say so rather than leave it implicit.
        if latent_on:
            ext_lat = seed_dir / "latent.pt"
            if ext_lat.exists():
                lstate = torch.load(ext_lat, map_location="cpu", weights_only=True)
                _ = plan_enc.load_state_dict(  # type: ignore[union-attr]
                    {k: v.to(torch.bfloat16) for k, v in lstate["plan_encoder"].items()}
                )
                _ = plan_pred.load_state_dict(  # type: ignore[union-attr]
                    {k: v.to(torch.bfloat16) for k, v in lstate["plan_predictor"].items()}
                )
                log(f"warm start also carried plan modules from {ext_lat}")
            else:
                log("warm start has no latent.pt: plan encoder/predictor start FRESH "
                    "(expected for the first latent run off a pre-latent checkpoint)")

    gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)
    tokens_per_micro = args.microbatch * args.max_length

    model.train()
    if plan_enc is not None and plan_pred is not None:
        plan_enc.train()
        plan_pred.train()
    data_iter = iter(train_loader)
    run_diag: dict[str, float] = {}
    run_n = 0
    t0 = time.time()

    for step in range(start_step + 1, args.steps + 1):
        base_lr = lr_at(step)
        for group in optimizer.param_groups:
            # lr_mult implements adapter-first: warm-started backbone at base_lr,
            # from-scratch latent modules at latent_lr_mult x base_lr.
            group["lr"] = base_lr * float(group.get("lr_mult", 1.0))
        if plan_optimizer is not None:
            for group in plan_optimizer.param_groups:
                group["lr"] = base_lr * float(group.get("lr_mult", 1.0))
        lam_c = lambda_c_schedule(tokens_seen, args.lambda_c_warm, args.lambda_c_main)

        optimizer.zero_grad(set_to_none=True)
        if plan_optimizer is not None:
            plan_optimizer.zero_grad(set_to_none=True)
        for _micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)

            doc_spans = None
            if args.docgen_prob > 0.0 and "doc_starts" in batch:
                doc_spans = (
                    batch["doc_starts"].to(device, non_blocking=True),
                    batch["prompt_ends"].to(device, non_blocking=True),
                    batch["doc_ends"].to(device, non_blocking=True),
                )
            corrupted, mask, bucket, gen_rows, block_t = sample_corruption(
                ids, am, args.block_size, args.span_prob, gen, gen_prob=args.gen_prob,
                docgen_prob=args.docgen_prob, doc_spans=doc_spans)

            # ---- latent conditioning (v4_plan Plan-1 M0) ----
            # P1: the denoiser is conditioned on the PREDICTED Ẑ, which sees the
            # corrupted canvas only. The clean-token encoder produces the target
            # for the alignment loss and is never fed to the denoiser.
            z_for_model = None
            loss_latent = None
            if latent_on and args.latent_z_source == "state":
                # ---- Phase-3 state-derived Z (v5_3_plan §4 Phase 3) ----
                # Oracle z_star from the model's own forward recurrence on the
                # CLEAN ids (E_deploy is CUT: the target is never captured from
                # corrupted tokens), through the FROZEN readout and the shared
                # chunk/T5 geometry. Robustness knobs close the train/test gap
                # to the prior's zhat: gaussian noise at the prior's error
                # scale, and condition dropout (the CFG hook).
                assert state_readout is not None  # noqa: S101
                # Draw the drop FIRST: a dropped micro skips the capture forward
                # entirely (the capture is the expensive part, ~+0.5x step).
                drop_draw = float(torch.rand((), generator=gen, device=device))
                if args.z_drop > 0.0 and drop_draw < args.z_drop:
                    z_for_model = None
                    diag_latent = {"z_dropped": 1.0}
                else:
                    with torch.no_grad():
                        z_star = capture_z_slots(
                            model, state_readout, ids,
                            block_size=CAPTURE_BLOCK_SIZE, n_layers=n_layers,
                        ).to(torch.bfloat16)                   # [B, 512, 32]
                    if args.z_noise_sigma > 0.0:
                        z_star = z_star + args.z_noise_sigma * torch.randn_like(z_star)
                    z_for_model = z_star
                    # Liveness: a conditioning lane that silently does nothing is
                    # the round-4a failure mode. For a captured (not learned) Z
                    # the tell is the field going constant across the batch.
                    diag_latent = {
                        "z_absmean": float(z_star.detach().float().abs().mean()),
                        "z_batch_spread": float(
                            z_star.detach().float().flatten(1).norm(dim=-1).std()
                        ),
                    }
            elif latent_on and plan_enc is not None and plan_pred is not None:
                z_target = plan_enc(ids)
                z_hat, z_u = plan_pred(corrupted)
                z_for_model = z_hat
                loss_align, ldiag = latent_align_loss(z_hat, z_target, z_u)
                loss_latent = args.lambda_z_align * loss_align
                diag_latent = dict(ldiag)
                # Instrumentation is not optional: a conditioning lane that
                # silently does nothing is the round-4a failure mode. z_absmean
                # going to 0 (or staying flat at its init value) means the lane
                # is dead even while every other metric looks healthy.
                with torch.no_grad():
                    diag_latent["z_absmean"] = float(z_hat.detach().float().abs().mean())
                    diag_latent["z_std"] = float(z_hat.detach().float().std())
            else:
                diag_latent = {}

            # ---- arm C: full recurrent-state injection (v4_plan section 5.3) ----
            # Capture the state the model itself reaches on the CLEAN ids, then hand
            # that state to the forward that sees only the corrupted canvas. No second
            # frozen copy is needed: under no_grad the model IS its own frozen copy,
            # which saves 8.2 GB and a second FSDP wrap.
            #
            # The capture forward deliberately passes z_slots=None -- conditioning the
            # capture would measure the model conditioning itself, not a ceiling.
            state_cache = None
            if args.state_cond == "full":
                with torch.no_grad():
                    # REUSE the capture cache as the injection cache. A fresh
                    # fla ``Cache()`` starts with ZERO layers and grows only inside
                    # ``.update()``, so handing one to ``inject_predicted_states``
                    # (which indexes ``cache.layers[i]`` directly) raises IndexError
                    # on layer 0 -- it did, on every rank, at step 1. The capture
                    # forward has already allocated all ``n_layers`` and filled them
                    # with exactly the states we want, so at blend=1.0 it IS the
                    # injection cache and no second allocation is needed.
                    cap = make_state_cache()
                    src_ids = ids if args.state_source == "clean" else corrupted
                    model(src_ids, args.force_forward, None, cap, True)  # noqa: FBT003
                    captured = read_state_cache(cap, n_layers)
                diag_latent.update(state_norm_features(captured))
                # Optionally persist real captured states so the Step-5 retention gate
                # runs on the ACTUAL distribution. The first attempt at that gate was
                # VOID partly because its target was synthetic and near-constant
                # (relative std 3.2e-04), which made its R^2 meaningless. A compression
                # gate can only be read against real states.
                if args.dump_states and step in DUMP_STATE_STEPS:
                    _dump_captured_states(
                        Path(args.dump_states), step, rank, captured, ids, mask,
                        args.state_source,
                    )
                # Re-assert the cache contract on the captured object: fp32, zeroed
                # conv/ffn transients, and _seen_tokens reset. The capture forward
                # left a token count behind that the injected recurrence must not see.
                state_cache = inject_predicted_states(cap, captured)

            bt = block_t if args.block_t_cond else None
            # ---- self-conditioning (v5.3 C5 / PT T4): x-hat-0 feedback ----
            # With prob p_self_cond a micro is a self-cond step; half of those
            # zero the feedback channel (the standard recipe), and a zeroed
            # channel is exactly a plain step here because the feedback IS the
            # canvas — so only p/2 of micros pay the extra no-grad forward.
            # The loss stays on the original mask positions: the model learns
            # to keep predicting the target while seeing its own previous
            # commits there, which is the sampler's actual input distribution.
            # Skipped when a state cache is injected (arm C, P2 ablation-only):
            # the fla cache mutates in the first forward and must not be
            # consumed twice.
            canvas = corrupted
            if args.p_self_cond > 0.0 and state_cache is None:
                sc_draw = float(torch.rand((), generator=gen, device=device))
                if sc_draw < args.p_self_cond * 0.5:
                    with torch.no_grad():
                        sc_logits = model(canvas, args.force_forward, z_for_model,
                                          None, False, bt, args.block_size)  # noqa: FBT003
                        sc_pred = sc_logits.argmax(dim=-1)
                    canvas = torch.where(mask, sc_pred, corrupted)
                    diag_latent["self_cond_fired"] = 1.0
            logits = model(canvas, args.force_forward, z_for_model, state_cache,
                           False, bt, args.block_size)  # noqa: FBT003
            loss_mask, diag = masked_diffusion_loss(logits, ids, mask, bucket, args.block_size)
            diag.update(diag_latent)
            # Always record the gen-row share: without this, a log with no ce_gen is
            # ambiguous between "the docgen lane never fired" and "it fired but the
            # metric was not printed" — which is exactly the blind spot that nearly
            # sent a 4k-step round out unverified.
            diag_gen_frac = float(gen_rows.float().mean())
            if gen_rows.any():
                with torch.no_grad():
                    gm = mask & gen_rows.unsqueeze(1)
                    gce = functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]).float(), ids.view(-1),
                        reduction="none").view_as(ids) * gm
                    cnt = gm.sum()
                    if cnt > 0:
                        diag["ce_gen"] = float(gce.sum() / cnt)
            diag["gen_frac"] = diag_gen_frac
            loss_causal = causal_replay_loss(model, ids, am)
            loss = loss_mask + lam_c * loss_causal
            if loss_latent is not None:
                loss = loss + loss_latent
            # ---- joint commit (v5.2 B4 / Q1 U3b) ----
            # Costs one extra forward, hence the `every` gate. WITH grad on purpose: the
            # point is to teach the backbone to read same-step commits, so a no-grad branch
            # would train nothing.
            if args.lambda_joint_commit > 0.0 and (_micro % args.joint_commit_every == 0):
                loss_jc, diag_jc = joint_commit_loss(
                    model, ids, corrupted, mask, gen, args.joint_commit_group,
                )
                if diag_jc:
                    loss = loss + args.lambda_joint_commit * loss_jc
                    diag.update(diag_jc)
            # ---- plan discrimination (v4_plan section 5, D0.2) ----
            # The wrong-plan forward is a DERANGEMENT of the same Zhat batch, so
            # this trains exactly the comparison d0_eval's `shuffle` arm scores.
            # It needs grad (unlike the keep-term's base forward): the point is to
            # push the conditioner's response to plan CONTENT, and a no-grad wrong
            # branch would only supply a detached constant.
            if (
                latent_on
                and args.lambda_contrast > 0.0
                and z_for_model is not None
                and z_for_model.shape[0] >= 2
                and (_micro % args.contrast_every == 0)
            ):
                perm = derange(z_for_model.shape[0], z_for_model.device, gen)
                wrong_logits = model(canvas, args.force_forward, z_for_model[perm])
                loss_contrast, cdiag = plan_contrastive_loss(
                    logits, wrong_logits, ids, mask, margin=args.contrast_margin)
                loss = loss + args.lambda_contrast * loss_contrast
                diag.update(cdiag)

            # ---- distribution preservation (v4_plan section 10.5) ----
            # KL against the model's OWN unconditioned forward, computed every
            # keep_every microbatches (it costs one extra forward). This is the
            # guard that lets conditioning sharpen predictions without walking
            # away from the language distribution section 1.1 measured.
            if latent_on and args.lambda_keep > 0.0 and (_micro % args.keep_every == 0):
                with torch.no_grad():
                    base_logits = model(canvas, args.force_forward, None)
                valid_keep = am.float()
                loss_keep, kdiag = distribution_keep_loss(logits, base_logits, valid_keep)
                loss = loss + args.lambda_keep * loss_keep
                diag.update(kdiag)
            loss = loss / args.grad_accum
            loss.backward()

            tokens_seen += tokens_per_micro * world
            diag["causal_ce"] = float(loss_causal.detach())
            for k, v in diag.items():
                if not math.isnan(v):
                    run_diag[k] = run_diag.get(k, 0.0) + v
                    run_diag[f"__n_{k}"] = run_diag.get(f"__n_{k}", 0) + 1
            run_n += 1

        # The plan encoder/predictor are replicated, not FSDP-sharded, so their
        # grads must be all-reduced by hand to stay identical across ranks.
        plan_param_list: list[torch.nn.Parameter] = []
        if plan_enc is not None and plan_pred is not None:
            plan_param_list = [*plan_enc.parameters(), *plan_pred.parameters()]
        if world > 1 and plan_param_list:
            for p in plan_param_list:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.div_(world)

        if world > 1:
            grad_norm = model.clip_grad_norm_(args.clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if plan_param_list:
            _ = torch.nn.utils.clip_grad_norm_(plan_param_list, args.clip)
        if args.stage_b_frac > 0:
            frac = step / max(1, args.steps)
            for name, param in model.named_parameters():
                if param.grad is not None and grad_frozen(
                    name, frac, n_layers, args.stage_a_frac, args.stage_b_frac
                ):
                    # zero (not None): AdamW must create state for every param
                    # or FSDP.optim_state_dict raises at checkpoint save.
                    param.grad.detach().zero_()
        optimizer.step()
        if plan_optimizer is not None:
            plan_optimizer.step()

        if step % args.log_every == 0 and rank == 0:
            avg = {k: run_diag[k] / max(1, run_diag.get(f"__n_{k}", 1))
                   for k in run_diag if not k.startswith("__")}
            dt = time.time() - t0
            alpha = (model.module.mean_forward_alpha() if hasattr(model, "module")
                     else model.mean_forward_alpha()) if world <= 1 else float("nan")
            mem_gb = torch.cuda.max_memory_allocated() / 2**30
            lat_txt = ""
            if latent_on:
                # z_absmean is the liveness signal: if it is ~0 or frozen at its
                # init value the conditioning lane is dead, no matter how healthy
                # mask_ce looks.
                lat_txt = (
                    f" z_mse={avg.get('z_align_mse', 0):.4f} z_u={avg.get('z_u_mean', 0):.3f}"
                    f" z_absmean={avg.get('z_absmean', 0):.4f} z_std={avg.get('z_std', 0):.4f}"
                    f" keep_kl={avg.get('keep_kl', 0):.4f}"
                    f" zgap={avg.get('z_contrast_gap', 0):+.4f}"
                )
            # U3 (v5.2 B3/B4) observability. Without these the joint-commit lane could
            # train for hours with no evidence it ever fired, and the timestep arm could sit
            # at exact identity unnoticed -- the A3 failure mode, which cost 43 GPU-h.
            # jc_rows is the FRACTION OF ROWS the loss could use (needs >=2 masked
            # positions); jc_ce is its CE; bt_absmean is the mean |additive term| the
            # timestep arm contributes, which is exactly 0 while it remains at zero-init and
            # must become non-zero once it trains.
            if args.lambda_joint_commit > 0.0:
                lat_txt += (
                    f" jc_rows={avg.get('jc_rows', 0):.2f}"
                    f" jc_rev={avg.get('jc_revealed', 0):.2f}"
                    f" jc_ce={avg.get('jc_ce', 0):.4f}"
                )
            if args.block_t_cond:
                # Filter the TOP-LEVEL named_parameters by name. The previous version drilled
                # into `model.module.block_t_cond` and summed `.parameters()`, which reported
                # bt_wsum=0.0000 -- indistinguishable from "weights are zero" but actually
                # meaning it saw NO parameters (a fresh module sums to ~2228, since only
                # `out` is zero-init while the trunk is randomly initialised). Name-filtering
                # works identically wrapped or unwrapped, and bt_n makes an empty read
                # impossible to mistake for a zero weight: 0 params and zero weights now
                # print differently.
                with torch.no_grad():
                    bt_ps = [
                        prm for nm, prm in model.named_parameters()
                        if "block_t_cond" in nm
                    ]
                    bt_w = float(sum(float(prm.detach().abs().sum()) for prm in bt_ps))
                lat_txt += f" bt_n={len(bt_ps)} bt_wsum={bt_w:.4f}"
            # The state diagnostics must be OUTSIDE the `latent_on` block: arm C runs
            # with --latent-cond none, so latent_on is False and these never printed --
            # the liveness guard was silently absent from exactly the run it exists to
            # protect. Caught at step 40 of the first healthy arm-C attempt.
            if args.state_cond != "none":
                lat_txt += (
                    f" st_abs={avg.get('state_absmean', 0):.4f}"
                    f" st_spread={avg.get('state_batch_spread', 0):.3f}"
                )
            log(f"[step {step}] mask_ce={avg.get('mask_ce', 0):.4f} "
                f"ce_low={avg.get('ce_low', 0):.4f} ce_med={avg.get('ce_med', 0):.4f} "
                f"ce_high={avg.get('ce_high', 0):.4f} causal={avg.get('causal_ce', 0):.4f} "
                f"lam_c={lam_c:.2f} grad={float(grad_norm):.3f} lr={lr_at(step):.2e} "
                f"gen_frac={avg.get('gen_frac', 0):.3f} ce_gen={avg.get('ce_gen', 0):.4f} "
                f"alpha={alpha:.3f} tokens={tokens_seen / 1e9:.3f}B "
                f"step/s={args.log_every / dt:.3f} mem={mem_gb:.1f}GB{lat_txt}")
            run_diag.clear()
            run_n = 0
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()

        if step % args.val_every == 0:
            model.eval()
            if plan_enc is not None and plan_pred is not None:
                plan_enc.eval()
                plan_pred.eval()
            vtot, vcnt = torch.zeros(2, device=device), 0
            # Paired conditioned-vs-unconditioned val: the SAME corrupted batch is
            # scored with Ẑ and with None, so the delta is the in-training
            # estimate of Gate D0.1 rather than a comparison across runs.
            vtot_uncond = torch.zeros(2, device=device)
            varm_totals: dict[str, torch.Tensor] = {}
            vbuckets = {n: [0.0, 0] for n in BUCKET_NAMES}
            with torch.no_grad():
                vgen = torch.Generator(device=device).manual_seed(777)
                for vb in val_loader:
                    ids = vb["input_ids"].to(device)
                    am = vb["attention_mask"].to(device)
                    corrupted, mask, bucket, _, v_block_t = sample_corruption(
                        ids, am, args.block_size, args.span_prob, vgen)
                    z_val = None
                    v_bt = v_block_t if args.block_t_cond else None
                    arm_ce: dict[str, torch.Tensor] | None = None
                    if latent_on and args.latent_z_source == "state":
                        # ---- Phase-3 gate arms (v5_3_plan §4 Phase 3) ----
                        # oracle: clean-capture z_star (ceiling; Delta_oracle-pred)
                        # zhat: prior-repaired corrupted-capture field (HEADLINE, P1)
                        # deranged: batch-deranged z_star (<=1/3-of-gain control)
                        # zero: zeros (injection-path-alone control)
                        # none: unconditioned (reference; scored below as ce_u)
                        assert state_readout is not None  # noqa: S101
                        z_star_v = capture_z_slots(
                            model, state_readout, ids,
                            block_size=CAPTURE_BLOCK_SIZE, n_layers=n_layers,
                        ).to(torch.bfloat16)
                        arms: dict[str, torch.Tensor | None] = {
                            "oracle": z_star_v,
                            "deranged": z_star_v[
                                derange(z_star_v.shape[0], device, vgen)
                            ] if z_star_v.shape[0] > 1 else None,
                            "zero": torch.zeros_like(z_star_v),
                        }
                        if latent_prior is not None:
                            # Deployable path: context field from the CORRUPTED
                            # canvas (arm-D mechanism), repaired by the frozen
                            # prior toward the clean-field distribution. The
                            # repair depth tracks the batch's realised mask
                            # fraction: heavier corruption => deeper re-noise.
                            z_ctx = capture_z_slots(
                                model, state_readout, corrupted,
                                block_size=CAPTURE_BLOCK_SIZE, n_layers=n_layers,
                            ).to(torch.bfloat16)
                            m_frac = float(mask.float().mean())
                            zhat_v = latent_prior.sample_repair(
                                z_ctx,
                                z_ctx.reshape(z_ctx.shape[0], -1,
                                              z_ctx.shape[-1]).mean(dim=1),
                                noise_frac=min(0.9, max(0.1, m_frac)),
                                steps=32,
                                seed=777,
                            ).to(torch.bfloat16)
                            arms["zhat"] = zhat_v
                        arm_ce = {}
                        for arm_name, arm_z in arms.items():
                            if arm_z is None:
                                continue
                            a_logits = model(corrupted, args.force_forward, arm_z,
                                             None, False, v_bt, args.block_size)  # noqa: FBT003
                            a_ce = functional.cross_entropy(
                                a_logits.view(-1, a_logits.shape[-1]).float(),
                                ids.view(-1), reduction="none",
                            ).view_as(ids) * mask
                            pair = torch.stack([a_ce.sum(), mask.sum().float()])
                            arm_ce[arm_name] = pair
                            if arm_name not in varm_totals:
                                varm_totals[arm_name] = torch.zeros(2, device=device)
                            varm_totals[arm_name] += pair
                        # headline forward below scores the ORACLE arm as the
                        # in-training trend line (cheap); the zhat gate number
                        # comes from arm_ce["zhat"] when the prior is loaded.
                        z_val = z_star_v
                    elif latent_on and plan_pred is not None:
                        z_val, _ = plan_pred(corrupted)  # P1: predicted, never target
                    # Arm C: the val d0_delta must compare INJECTED-state vs plain,
                    # or a state-only run would report no delta at all and the arm
                    # would look inert when it is merely unmeasured.
                    v_state = None
                    if args.state_cond == "full":
                        # Same reuse as the train path: a fresh Cache() has no layers.
                        cap_v = make_state_cache()
                        # Same source as training: a val arm capturing from a different
                        # distribution than the trained one would measure a mismatch
                        # rather than the arm.
                        vsrc = ids if args.state_source == "clean" else corrupted
                        model(vsrc, args.force_forward, None, cap_v, True)  # noqa: FBT003
                        v_state = inject_predicted_states(
                            cap_v, read_state_cache(cap_v, n_layers)
                        )
                    logits = model(corrupted, args.force_forward, z_val, v_state,
                                   False, v_bt, args.block_size)  # noqa: FBT003
                    if z_val is not None or v_state is not None:
                        # Unconditioned control: no z AND no injected state.
                        base_logits_v = model(corrupted, args.force_forward, None)
                        ce_u = functional.cross_entropy(
                            base_logits_v.view(-1, base_logits_v.shape[-1]).float(),
                            ids.view(-1), reduction="none",
                        ).view_as(ids) * mask
                        vtot_uncond += torch.stack([ce_u.sum(), mask.sum().float()])
                    flat_ce = functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]).float(),
                        ids.view(-1),
                        reduction="none",
                    )
                    ce = flat_ce.view_as(ids) * mask
                    vtot += torch.stack([ce.sum(), mask.sum().float()])
                    n_blocks = bucket.shape[1]
                    for bi, nme in enumerate(BUCKET_NAMES):
                        bm = torch.zeros_like(mask)
                        for blk in range(n_blocks):
                            v_end = min((blk + 1) * args.block_size, ids.shape[1])
                            s, e = blk * args.block_size, v_end
                            bm[:, s:e] = mask[:, s:e] & (bucket[:, blk] == bi).unsqueeze(1)
                        vbuckets[nme][0] += float((ce * bm).sum())
                        vbuckets[nme][1] += int(bm.sum())
                    vcnt += 1
            # NOTE: vtot is deliberately NOT all-reduced. The pre-latent trainer
            # reported val_mask_ce as a rank-0-local figure, and every historical
            # baseline (e.g. round-4b val_mask_ce 4.3877) is on that definition.
            # Making it global here would silently change the metric's meaning
            # and break comparability. The d0_delta below is a difference of two
            # figures computed on the same rank and the same batches, so it is
            # unaffected by this choice.
            if rank == 0:
                bstr = " ".join(f"val_{n}={vbuckets[n][0] / max(1, vbuckets[n][1]):.4f}"
                                for n in BUCKET_NAMES)
                val_ce = float(vtot[0] / vtot[1].clamp_min(1))
                d0_txt = ""
                if latent_on or args.state_cond == "full":
                    val_u = float(vtot_uncond[0] / vtot_uncond[1].clamp_min(1))
                    # D0.1 in-training estimate. POSITIVE = conditioning helps.
                    # This is the number the whole milestone exists to move; the
                    # Gate-D0 threshold is >= 0.05 nats measured properly offline
                    # with CIs, so treat this as a trend, not the verdict.
                    d0_txt = (
                        f" val_uncond={val_u:.4f} d0_delta={val_u - val_ce:+.4f}"
                    )
                arm_txt = ""
                if varm_totals:
                    # POSITIVE delta = the arm helps vs unconditioned. The gate
                    # reads zhat (headline, P1) with oracle as the ceiling and
                    # deranged/zero as controls; thresholds live in v5_3_plan §4.
                    val_u_arm = float(vtot_uncond[0] / vtot_uncond[1].clamp_min(1))
                    parts = []
                    for arm_name in ("oracle", "zhat", "deranged", "zero"):
                        if arm_name in varm_totals:
                            tot = varm_totals[arm_name]
                            a_val = float(tot[0] / tot[1].clamp_min(1))
                            parts.append(
                                f"arm_{arm_name}={a_val:.4f}"
                                f"(d={val_u_arm - a_val:+.4f})"
                            )
                    arm_txt = " " + " ".join(parts)
                log(f"[val step {step}] val_mask_ce={val_ce:.4f} {bstr}{d0_txt}{arm_txt}")
            model.train()
            if plan_enc is not None and plan_pred is not None:
                plan_enc.train()
                plan_pred.train()

        if args.sampler_every > 0 and step % args.sampler_every == 0 and rank == 0 and world <= 1:
            sm = sampler_eval(model, next(iter(val_loader)), device)
            log(f"[sampler step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in sm.items()))
        elif args.sampler_every > 0 and step % args.sampler_every == 0 and world > 1:
            # FSDP: all ranks must participate in the forward passes
            sm = sampler_eval(model, next(iter(val_loader)), device)
            if rank == 0:
                log(f"[sampler step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in sm.items()))

        if step % args.save_every == 0 or step == args.steps:
            plan_mods = None
            if plan_enc is not None and plan_pred is not None:
                plan_mods = {"plan_encoder": plan_enc, "plan_predictor": plan_pred}
            save_checkpoint(
                model, optimizer, step, tokens_seen, save_dir, rank,
                plan_modules=plan_mods, plan_optimizer=plan_optimizer,
            )

    log("training complete")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
