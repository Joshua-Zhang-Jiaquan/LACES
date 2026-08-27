"""Shared boundary-state chunk-latent geometry (v5.3 Phase 2a/3).

ONE written convention (v5_3_plan §1b item 5) for turning the 2.9B forward
recurrence into the chunked block-latent field, shared by:

* the offline Phase-2a capture (``eval/capability/boundary_state_capture.py``),
* the Phase-2b latent-prior trainer (``train/train_latent_prior_capture.py``),
* the B8 probe decoder, and
* the Phase-3 token-plane trainer's in-loop z_target capture
  (``train/train_birwkv_diffusion.py`` ``--latent-z-source state``).

Any change to the pooling, chunking, or T5 normalization below changes the
E_target interface for every consumer at once — which is the point: a capture
and a downlink that silently disagree on normalization is the D0-dissociation
failure mode (a real signal destroyed at an interface).

Geometry: per 4096-token row, H=64 blocks × 64 tokens. The frozen
``StateTokenConditioner`` reads the boundary WKV state as 64 layer-tokens ×
256 dims; the 64 layer-tokens are mean-pooled to one 256-float block
descriptor; 256 = n·d_z reshapes exactly into n=8 chunks × d_z=32; each chunk
is normalized to the unit sphere × sqrt(d_z) (T5) so data variance matches the
prior's unit-Gaussian noise scale.
"""

from __future__ import annotations

import math

import torch

# The E_target capture geometry is FROZEN at 64-token blocks (v5.2 §1a: H=64
# blocks per 4096 row). It is deliberately independent of the trainer's
# corruption --block-size (F2 corrupts at 256): the readout/prior/probe all
# trained on 64-token boundaries, so the downlink must capture at the same
# granularity regardless of the corruption schedule.
CAPTURE_BLOCK_SIZE = 64
N_CHUNKS = 8
CHUNK_DZ = 32
FIELD_TOKENS = 64
FIELD_DIM = 256


def capture_boundary_tokens(
    model: object,
    conditioner: object,
    ids: torch.Tensor,
    *,
    block_size: int,
    n_layers: int,
    to_cpu_fp16: bool = True,
) -> torch.Tensor:
    """Return [B, n_blocks, 64, 256] boundary token fields for one batch.

    One incremental forward per block with the carried fla cache; after block
    h the cache holds the recurrent state at boundary h, which the frozen
    conditioner reads out as 64 tokens x 256 dims.

    ``to_cpu_fp16=True`` (the offline-capture default) moves each snapshot to
    fp16 CPU as it is produced, bounding GPU memory at one block's field. The
    in-loop trainer passes ``False`` to keep the field on device in the
    conditioner's dtype for immediate consumption.
    """
    from models.latent_plan import make_state_cache, read_state_cache  # noqa: PLC0415

    _batch, total = ids.shape
    n_blocks = total // block_size
    fields: list[torch.Tensor] = []
    cache = make_state_cache()
    with torch.no_grad():
        for block in range(n_blocks):
            chunk = ids[:, block * block_size : (block + 1) * block_size]
            # force_forward=True: boundary state is a FORWARD-stream object; the
            # reverse stream is anti-causal and has no defined boundary state.
            _ = model(chunk, True, None, cache, True)  # noqa: FBT003
            captured = read_state_cache(cache, n_layers)
            snapshot = torch.stack(captured, dim=1)  # [B, L, 40, 64, 64]
            tokens = conditioner(snapshot.to(next(conditioner.parameters()).dtype))
            fields.append(
                tokens.detach().to(torch.float16).cpu() if to_cpu_fp16
                else tokens.detach()
            )
    return torch.stack(fields, dim=1)  # [B, n_blocks, 64, 256]


def field_rows_to_grids(fields: torch.Tensor) -> torch.Tensor:
    """[R, H, 64, 256] capture fields -> [R, H, n=8, d_z=32] normalized grids.

    Mean-pool the 64 layer-tokens into the 256-float block descriptor, reshape
    to chunks (v5.2 §1a: 256 = n*d_z exactly), then T5-normalize each chunk to
    the unit sphere scaled by sqrt(d_z) so data variance matches the prior's
    unit-Gaussian noise scale.
    """
    if fields.shape[-2:] != (FIELD_TOKENS, FIELD_DIM):
        msg = f"expected [*, {FIELD_TOKENS}, {FIELD_DIM}], got {tuple(fields.shape)}"
        raise ValueError(msg)
    descriptor = fields.float().mean(dim=-2)                      # [R, H, 256]
    chunks = descriptor.reshape(*descriptor.shape[:-1], N_CHUNKS, CHUNK_DZ)
    norm = chunks.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return chunks / norm * math.sqrt(CHUNK_DZ)


def capture_z_slots(
    model: object,
    conditioner: object,
    ids: torch.Tensor,
    *,
    block_size: int,
    n_layers: int,
) -> torch.Tensor:
    """ids [B, T] -> flattened normalized chunk-latent slots [B, H*n, d_z].

    The Phase-3 trainer's one-call path: capture (on device) -> descriptor ->
    chunk grid -> T5 normalization -> flatten blocks x chunks into the slot
    axis that ``slots_to_positions`` consumes (slot h*n+c conditions chunk c of
    block h; at T=4096, H*n=512 slots => 8 tokens per slot, exactly one chunk).
    """
    field = capture_boundary_tokens(
        model, conditioner, ids,
        block_size=block_size, n_layers=n_layers, to_cpu_fp16=False,
    )                                                             # [B, H, 64, 256]
    grids = field_rows_to_grids(field)                            # [B, H, 8, 32]
    return grids.reshape(grids.shape[0], -1, CHUNK_DZ)            # [B, 512, 32]


__all__ = [
    "CAPTURE_BLOCK_SIZE",
    "CHUNK_DZ",
    "FIELD_DIM",
    "FIELD_TOKENS",
    "N_CHUNKS",
    "capture_boundary_tokens",
    "capture_z_slots",
    "field_rows_to_grids",
]
