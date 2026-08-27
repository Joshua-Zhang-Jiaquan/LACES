"""Latent collapse monitors for the v5.3 latent prior (C4).

Two complementary alarms, both halting alarms in Phase 2b:

- ``norm_loss_correlation`` — the corr(z_norm, loss) "leakage" alarm. The
  representation-collapse incident on job-ad95cb44 showed pearson r = 0.994
  between falling latent norm and falling diffusion loss (the optimizer was
  shrinking its own regression target, not denoising). The target is detached
  now, but the alarm stays as the cheap regression test of that contract.

- ``nnd_summary`` — the LangFlow-style nearest-neighbor-distance collapse
  diagnostic. A chunk field whose vectors crowd one another (NND distribution
  contracting toward zero) is losing diversity even when norms are stable.
  The threshold is derived from the 0.4B pilot's own healthy-run distribution
  (v5.3 §0-V6: the published Plaid figure is asserted-grade, so no gate reads
  it); this module only measures and reports.

Both functions are pure torch, CPU-safe, and stateless so the trainer's val
loop and offline audits consume the same code.
"""

from __future__ import annotations

import torch

__all__: tuple[str, ...] = ("nnd_summary", "norm_loss_correlation")

_MIN_POINTS = 2


def norm_loss_correlation(z_norms: torch.Tensor, losses: torch.Tensor) -> float:
    """Return pearson corr between per-step latent norms and losses.

    Values near +1 reproduce the ad95cb44 collapse signature (norm shrinking
    in lockstep with loss). Series shorter than two points, or constant
    series, return 0.0 (no evidence either way).
    """
    if z_norms.numel() != losses.numel():
        msg = f"series length mismatch: {z_norms.numel()} vs {losses.numel()}"
        raise ValueError(msg)
    if z_norms.numel() < _MIN_POINTS:
        return 0.0
    x = z_norms.detach().flatten().to(torch.float64)
    y = losses.detach().flatten().to(torch.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.norm() * y_centered.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float((x_centered @ y_centered) / denominator)


def nnd_summary(
    chunk_field: torch.Tensor,
    *,
    normalize: bool = True,
    max_points: int = 4096,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Summarize the nearest-neighbor-distance distribution of a chunk field.

    ``chunk_field`` is any ``[..., d_z]`` tensor; leading dimensions are
    flattened to a point cloud. With ``normalize`` the points are projected to
    the unit sphere first (the LangFlow convention — collapse then reads as
    small angular NND independent of norm drift). Point clouds larger than
    ``max_points`` are subsampled for the O(N^2) distance matrix.

    Returns mean / p05 / p50 of the NND distribution plus the effective点数.
    """
    points = chunk_field.detach().reshape(-1, chunk_field.shape[-1]).to(torch.float32)
    count = points.shape[0]
    if count < _MIN_POINTS:
        return {"nnd_mean": 0.0, "nnd_p05": 0.0, "nnd_p50": 0.0, "nnd_points": float(count)}
    if count > max_points:
        index = torch.randperm(count, generator=generator, device=points.device)[:max_points]
        points = points[index]
        count = max_points
    if normalize:
        points = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    nnd = distances.min(dim=1).values
    return {
        "nnd_mean": float(nnd.mean()),
        "nnd_p05": float(nnd.quantile(0.05)),
        "nnd_p50": float(nnd.quantile(0.5)),
        "nnd_points": float(count),
    }
