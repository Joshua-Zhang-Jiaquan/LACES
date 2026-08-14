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

from data.fineweb4096_packed import FineWebPackedPickleDataset, ShardContiguousSampler  # noqa: E402
from models.birwkv7_diffusion import (  # noqa: E402
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    BiRWKV7Block,
    BiRWKV7ForMaskedDiffusion,
    iterative_denoise,
    kendall_tau_commit_order,
)

# noise-level mixture (code/plan section 4.1)
NOISE_BUCKETS = ((0.05, 0.30, 0.30), (0.30, 0.70, 0.40), (0.70, 1.00, 0.30))
BUCKET_NAMES = ("low", "med", "high")
_TAU_EVAL_STEPS = 16
_MATRIX_NDIM = 2


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(sample, block) noise-bucket corruption + optional generation lane.

    With probability ``gen_prob`` a row uses FREE-GENERATION corruption
    instead: a clean prefix (uniform 5-60% of the sequence) followed by an
    all-masked contiguous canvas to the end. This is the corruption shape of
    prompt->completion generation, which block-local bucket masking never
    produces (Stage-C finding: strong infill, no synthesis).

    Returns (corrupted_ids, mask [B,T] bool, bucket_idx [B, n_blocks] long,
    gen_rows [B] bool).
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

    corrupted = torch.where(mask, torch.full_like(input_ids, MASK_TOKEN_ID), input_ids)
    return corrupted, mask, bucket, gen_rows


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
) -> None:
    """Write one atomic full-state checkpoint and prune old ones on rank 0."""
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
    p.add_argument("--span-prob", type=float, default=0.15)
    p.add_argument("--gen-prob", type=float, default=0.0,
                   help="fraction of rows using free-generation corruption "
                        "(clean prefix + all-masked tail) instead of block buckets")
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
    args = p.parse_args()

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
    log(f"params={n_params / 1e9:.3f}B  mean_forward_alpha={model.mean_forward_alpha():.4f}")

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

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < _MATRIX_NDIM or "norm" in name.lower() else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

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
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

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
    resume = find_resume(save_dir)
    if resume is not None:
        meta = json.loads((resume / "meta.json").read_text())
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        state = torch.load(resume / "model.pt", map_location="cpu", weights_only=True)
        if world > 1:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state)
            osd = torch.load(resume / "optim.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, osd))
        else:
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
        start_step, tokens_seen = meta["step"], meta["tokens_seen"]
        log(f"resumed from {resume} (step={start_step})")
    elif args.resume_from:
        # weights-only warm start from an external run (fresh optimizer, step 0)
        seed_dir = Path(args.resume_from)
        state = torch.load(seed_dir / "model.pt", map_location="cpu", weights_only=True)
        if world > 1:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state)
        else:
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
        log(f"weights-only warm start from {seed_dir} (optimizer fresh, step 0)")

    gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)
    tokens_per_micro = args.microbatch * args.max_length

    model.train()
    data_iter = iter(train_loader)
    run_diag: dict[str, float] = {}
    run_n = 0
    t0 = time.time()

    for step in range(start_step + 1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        lam_c = lambda_c_schedule(tokens_seen, args.lambda_c_warm, args.lambda_c_main)

        optimizer.zero_grad(set_to_none=True)
        for _micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)

            corrupted, mask, bucket, gen_rows = sample_corruption(
                ids, am, args.block_size, args.span_prob, gen, gen_prob=args.gen_prob)
            logits = model(corrupted, force_forward=args.force_forward)
            loss_mask, diag = masked_diffusion_loss(logits, ids, mask, bucket, args.block_size)
            if gen_rows.any():
                with torch.no_grad():
                    gm = mask & gen_rows.unsqueeze(1)
                    gce = functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]).float(), ids.view(-1),
                        reduction="none").view_as(ids) * gm
                    cnt = gm.sum()
                    if cnt > 0:
                        diag["ce_gen"] = float(gce.sum() / cnt)
            loss_causal = causal_replay_loss(model, ids, am)
            loss = (loss_mask + lam_c * loss_causal) / args.grad_accum
            loss.backward()

            tokens_seen += tokens_per_micro * world
            diag["causal_ce"] = float(loss_causal.detach())
            for k, v in diag.items():
                if not math.isnan(v):
                    run_diag[k] = run_diag.get(k, 0.0) + v
                    run_diag[f"__n_{k}"] = run_diag.get(f"__n_{k}", 0) + 1
            run_n += 1

        if world > 1:
            grad_norm = model.clip_grad_norm_(args.clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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

        if step % args.log_every == 0 and rank == 0:
            avg = {k: run_diag[k] / max(1, run_diag.get(f"__n_{k}", 1))
                   for k in run_diag if not k.startswith("__")}
            dt = time.time() - t0
            alpha = (model.module.mean_forward_alpha() if hasattr(model, "module")
                     else model.mean_forward_alpha()) if world <= 1 else float("nan")
            mem_gb = torch.cuda.max_memory_allocated() / 2**30
            log(f"[step {step}] mask_ce={avg.get('mask_ce', 0):.4f} "
                f"ce_low={avg.get('ce_low', 0):.4f} ce_med={avg.get('ce_med', 0):.4f} "
                f"ce_high={avg.get('ce_high', 0):.4f} causal={avg.get('causal_ce', 0):.4f} "
                f"lam_c={lam_c:.2f} grad={float(grad_norm):.3f} lr={lr_at(step):.2e} "
                f"alpha={alpha:.3f} tokens={tokens_seen / 1e9:.3f}B "
                f"step/s={args.log_every / dt:.3f} mem={mem_gb:.1f}GB")
            run_diag.clear()
            run_n = 0
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()

        if step % args.val_every == 0:
            model.eval()
            vtot, vcnt = torch.zeros(2, device=device), 0
            vbuckets = {n: [0.0, 0] for n in BUCKET_NAMES}
            with torch.no_grad():
                vgen = torch.Generator(device=device).manual_seed(777)
                for vb in val_loader:
                    ids = vb["input_ids"].to(device)
                    am = vb["attention_mask"].to(device)
                    corrupted, mask, bucket, _ = sample_corruption(
                        ids, am, args.block_size, args.span_prob, vgen)
                    logits = model(corrupted, force_forward=args.force_forward)
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
            if rank == 0:
                bstr = " ".join(f"val_{n}={vbuckets[n][0] / max(1, vbuckets[n][1]):.4f}"
                                for n in BUCKET_NAMES)
                val_ce = float(vtot[0] / vtot[1].clamp_min(1))
                log(f"[val step {step}] val_mask_ce={val_ce:.4f} {bstr}")
            model.train()

        if step % args.sampler_every == 0 and rank == 0 and world <= 1:
            sm = sampler_eval(model, next(iter(val_loader)), device)
            log(f"[sampler step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in sm.items()))
        elif step % args.sampler_every == 0 and world > 1:
            # FSDP: all ranks must participate in the forward passes
            sm = sampler_eval(model, next(iter(val_loader)), device)
            if rank == 0:
                log(f"[sampler step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in sm.items()))

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, optimizer, step, tokens_seen, save_dir, rank)

    log("training complete")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
