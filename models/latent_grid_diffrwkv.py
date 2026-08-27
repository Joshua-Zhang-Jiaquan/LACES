"""Context-conditioned DDPM over a masked semantic latent grid."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, final

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override

from models.state_hijacking_dit import TrajectoryLatentRWKV
from models.state_hijacking_latent_grid_errors import (
    context_shape_error,
    distinct_storage_error,
    flat_grid_mask_error,
    flat_grid_shape_error,
    grid_layout_error,
    grid_mask_error,
    grid_shape_error,
    positive_steps_error,
    segment_count_error,
)
from models.state_hijacking_latent_grid_lineage import (
    UPPER_LINEAGE_SCHEMA_VERSION,
    UPPER_RUNTIME_ROLE,
    UpperLineage,
    UpperLineageRequest,
    UpperSourceBlocked,
    UpperSourceBlockedError,
    build_upper_lineage,
    publish_upper_lineage,
    trajectory_state_from_checkpoint,
)
from models.state_hijacking_gumbel_schedule import GumbelNoiseSchedule
from models.state_hijacking_schedule import cosine_alpha_bar

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

HLMS_SIZE_JUSTIFICATION = {
    "kind": "indivisible_state_machine",
    "reason": (
        "One coupled diffusion state machine: the masked semantic grid, "
        "noise schedule, and RWKV denoiser transition functions all share "
        "the same per-step latent state, so an artificial split would "
        "duplicate that state plumbing across module boundaries."
    ),
}

__all__: tuple[str, ...] = (
    "UPPER_LINEAGE_SCHEMA_VERSION",
    "UPPER_RUNTIME_ROLE",
    "LatentGridConfig",
    "LatentGridDiffRWKV",
    "UpperLineage",
    "UpperSourceBlocked",
    "assert_distinct_parameter_storage",
    "parameter_storage_ids",
)

_GRID_DIMENSIONS: Final = 4
_FLAT_DIMENSIONS: Final = 3
_CONTEXT_DIMENSIONS: Final = 2

GridMetric: TypeAlias = dict[str, torch.Tensor]
TensorState: TypeAlias = Mapping[str, torch.Tensor]


class TensorDenoiser(Protocol):
    """A registered module with a typed tensor-forward interface."""

    def __call__(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a denoised latent tensor."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield the registered trainable parameters."""
        ...

    def state_dict(self) -> TensorState:
        """Return the registered tensor state."""
        ...

class TensorHead(Protocol):
    """A registered tensor-to-tensor head."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one tensor result for the given tensor input."""
        ...


class StateLoader(Protocol):
    """A typed state-loading callback that preserves Torch module registration."""

    def __call__(self, state_dict: TensorState, *, strict: bool) -> None:
        """Load a tensor state map using the supplied strictness."""
        ...


@dataclass(frozen=True, slots=True)
class LatentGridConfig:
    """The fixed dimensions and denoiser width for the upper latent grid.

    v5.3 additions (all defaults preserve the historical eps/cosine/no-SC
    behavior bit-for-bit; see DAN/v5_3_plan C1-C3):

    - ``schedule``: ``"cosine"`` (historical control arm) or ``"gumbel"``
      (learnable information-uniform schedule with gamma conditioning).
    - ``objective``: ``"eps"`` (historical) or ``"rf"`` (rectified-flow arm,
      same convention as ``state_hijacking_diffusion.compute_diffusion``).
    - ``self_conditioning``: enables the [z_t, z0_prev] input channel
      (zero-init projection; p=0.25 training recipe lives in
      ``diffusion_loss``).
    """

    outer_slots: int = 64
    inner_slots: int = 64
    latent_dim: int = 32
    hidden_size: int = 768
    depth: int = 8
    bidirectional: bool = True
    schedule: str = "cosine"
    objective: str = "eps"
    self_conditioning: bool = False
    self_cond_rate: float = 0.25

    @property
    def horizon(self) -> int:
        """Return the flattened number of latent positions."""
        return self.outer_slots * self.inner_slots


@final
class LatentGridDiffRWKV(nn.Module):
    """Diffuse semantic latents while preserving grid and flat input layouts."""

    config: LatentGridConfig
    denoiser: TensorDenoiser
    outer_pos: torch.Tensor
    inner_pos: torch.Tensor
    count_head: TensorHead
    gumbel_schedule: GumbelNoiseSchedule | None
    self_cond_proj: nn.Linear | None
    _load_denoiser_state: StateLoader

    def __init__(self, config: LatentGridConfig | None = None) -> None:
        """Build the latent-grid denoiser with its historical default configuration."""
        if TYPE_CHECKING:
            pass
        else:
            nn.Module.__init__(self)
        self.config = LatentGridConfig() if config is None else config
        if self.config.schedule not in ("cosine", "gumbel"):
            msg = f"unknown schedule {self.config.schedule!r}; expected 'cosine' or 'gumbel'"
            raise ValueError(msg)
        if self.config.objective not in ("eps", "rf"):
            msg = f"unknown objective {self.config.objective!r}; expected 'eps' or 'rf'"
            raise ValueError(msg)
        denoiser = TrajectoryLatentRWKV(
            latent_dim=self.config.latent_dim,
            horizon=self.config.horizon,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            bidirectional=self.config.bidirectional,
        )
        self.denoiser = denoiser

        def load_denoiser_state(state_dict: TensorState, *, strict: bool) -> None:
            _ = denoiser.load_state_dict(state_dict, strict=strict)

        self._load_denoiser_state = load_denoiser_state
        self.outer_pos = nn.Parameter(torch.zeros(self.config.outer_slots, self.config.latent_dim))
        self.inner_pos = nn.Parameter(torch.zeros(self.config.inner_slots, self.config.latent_dim))
        _ = nn.init.normal_(self.outer_pos, std=0.02)
        _ = nn.init.normal_(self.inner_pos, std=0.02)
        self.count_head = nn.Linear(self.config.latent_dim, self.config.horizon + 1)
        # v5.3 C1: learnable Gumbel schedule — registered only when selected so the
        # default cosine configuration keeps the historical parameter set exactly.
        self.gumbel_schedule = (
            GumbelNoiseSchedule() if self.config.schedule == "gumbel" else None
        )
        # v5.3 C2: self-conditioning input channel, zero-init so an SC-enabled
        # model is function-identical to the SC-off model at initialization.
        if self.config.self_conditioning:
            self_cond_proj = nn.Linear(self.config.latent_dim, self.config.latent_dim)
            _ = nn.init.zeros_(self_cond_proj.weight)
            _ = nn.init.zeros_(self_cond_proj.bias)
            self.self_cond_proj: nn.Linear | None = self_cond_proj
        else:
            self.self_cond_proj = None

    def count_active_params(self) -> int:
        """Return the number of physically trainable upper parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    def count_stored_params(self) -> int:
        """Return the number of stored parameters without optimizer-state inflation."""
        return self.count_active_params()

    def _as_grid(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if values.ndim == _GRID_DIMENSIONS:
            expected = (
                values.shape[0],
                self.config.outer_slots,
                self.config.inner_slots,
                self.config.latent_dim,
            )
            if tuple(values.shape) != expected:
                raise grid_shape_error(expected, tuple(values.shape))
            if tuple(mask.shape) != expected[:_FLAT_DIMENSIONS] or mask.dtype != torch.bool:
                raise grid_mask_error()
            return values, mask, False
        if values.ndim == _FLAT_DIMENSIONS:
            expected = (values.shape[0], self.config.horizon, self.config.latent_dim)
            if tuple(values.shape) != expected:
                raise flat_grid_shape_error(expected, tuple(values.shape))
            if tuple(mask.shape) != expected[:_CONTEXT_DIMENSIONS] or mask.dtype != torch.bool:
                raise flat_grid_mask_error()
            return (
                values.reshape(
                    values.shape[0],
                    self.config.outer_slots,
                    self.config.inner_slots,
                    self.config.latent_dim,
                ),
                mask.reshape(values.shape[0], self.config.outer_slots, self.config.inner_slots),
                True,
            )
        raise grid_layout_error()

    def _validate(self, grid: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> None:
        grid_values, _, _ = self._as_grid(grid, mask)
        if tuple(context.shape) != (grid_values.shape[0], self.config.latent_dim):
            raise context_shape_error()

    def _positioned(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        position = self.outer_pos[:, None, :] + self.inner_pos[None, :, :]
        return (grid + position.unsqueeze(0)) * mask.unsqueeze(-1).to(grid.dtype)

    @override
    def forward(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor,
        context_condition: torch.Tensor,
        grid_mask: torch.Tensor,
        self_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masked noise and count logits in the caller's input layout.

        ``timestep`` is the raw t in [0, 1] under the cosine schedule and the
        log noise-to-signal ratio gamma under the gumbel schedule — the
        denoiser conditions on whichever coordinate the schedule uses.

        ``self_cond`` is the previous prediction of the clean grid (same
        layout as ``z_t``); it enters additively through the zero-init
        projection (equivalent to input concat + zero-init block, without
        changing the denoiser input width). Ignored unless the config enables
        self-conditioning.
        """
        grid, mask, input_was_flat = self._as_grid(z_t, grid_mask)
        self._validate(z_t, context_condition, grid_mask)
        if self.self_cond_proj is not None and self_cond is not None:
            cond_grid, _, _ = self._as_grid(self_cond, grid_mask)
            grid = grid + self.self_cond_proj(cond_grid.detach())
        flat = self._positioned(grid, mask).reshape(
            grid.shape[0], self.config.horizon, self.config.latent_dim
        )
        prediction = self.denoiser(flat, timestep, cond=context_condition).reshape_as(grid)
        prediction = prediction * mask.unsqueeze(-1).to(grid.dtype)
        if input_was_flat:
            prediction = prediction.reshape(
                z_t.shape[0], self.config.horizon, self.config.latent_dim
            )
        return prediction, self.count_head(context_condition)

    def _draw_schedule(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (denoiser time coordinate, alpha_bar) for one training batch."""
        if self.gumbel_schedule is not None:
            gamma = self.gumbel_schedule.sample_gamma(batch_size, device=device, dtype=dtype)
            return gamma, GumbelNoiseSchedule.alpha_bar(gamma).to(dtype)
        timestep = torch.rand(batch_size, device=device, dtype=dtype)
        return timestep, cosine_alpha_bar(timestep).to(dtype)

    def _predict_clean(
        self,
        noisy_grid: torch.Tensor,
        prediction: torch.Tensor,
        alpha: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean-grid estimate from an eps or rf prediction."""
        if self.config.objective == "rf":
            expanded = timestep.view(-1, 1, 1, 1).to(noisy_grid.dtype)
            return noisy_grid + (1.0 - expanded) * prediction
        return (noisy_grid - (1.0 - alpha).sqrt() * prediction) / alpha.sqrt().clamp_min(1e-4)

    def diffusion_loss(
        self,
        z_star: torch.Tensor,
        context_condition: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, GridMetric]:
        """Compute the masked denoising and active-count objectives.

        Default config reproduces the historical eps/cosine objective exactly.
        With ``schedule="gumbel"`` the batch draws gamma from the learnable
        information-uniform Gumbel density and adds the stop-grad scheduler
        fit loss; with ``objective="rf"`` the regression target is the
        rectified-flow velocity (``compute_diffusion`` convention: noisy =
        t*z + (1-t)*eps, target = z - eps); with self-conditioning enabled,
        a fraction ``self_cond_rate`` of batches runs a no-grad first pass
        whose clean-grid estimate feeds the graded second pass.
        """
        grid, mask, _ = self._as_grid(z_star, grid_mask)
        self._validate(z_star, context_condition, grid_mask)
        batch_size = grid.shape[0]
        weight = mask.unsqueeze(-1).to(grid.dtype)
        if self.config.objective == "rf":
            timestep = torch.sigmoid(
                torch.randn(batch_size, device=grid.device, dtype=grid.dtype) * 0.8 - 0.8
            )
            noise = torch.randn_like(grid) * weight
            expanded = timestep.view(batch_size, 1, 1, 1)
            noisy_grid = (expanded * grid + (1.0 - expanded) * noise) * weight
            target = (grid - noise) * weight
            alpha = torch.ones(batch_size, device=grid.device, dtype=grid.dtype).view(
                batch_size, 1, 1, 1
            )
        else:
            # RNG draw order (timestep, then noise) matches the historical
            # objective exactly, so the default configuration is bit-identical
            # to the pre-v5.3 code under a fixed seed.
            timestep, alpha_bar = self._draw_schedule(
                batch_size, device=grid.device, dtype=grid.dtype
            )
            alpha = alpha_bar.view(batch_size, 1, 1, 1)
            noise = torch.randn_like(grid) * weight
            noisy_grid = alpha.sqrt() * grid + (1.0 - alpha).sqrt() * noise
            target = noise
        self_cond: torch.Tensor | None = None
        if (
            self.self_cond_proj is not None
            and torch.rand((), device=grid.device).item() < self.config.self_cond_rate
        ):
            with torch.no_grad():
                first_prediction, _ = self.forward(
                    noisy_grid, timestep, context_condition, mask, self_cond=None
                )
                self_cond = (
                    self._predict_clean(noisy_grid, first_prediction, alpha, timestep) * weight
                ).detach()
        prediction, count_logits = self.forward(
            noisy_grid,
            timestep,
            context_condition,
            mask,
            self_cond=self_cond,
        )
        denominator = (weight.sum() * grid.shape[-1]).clamp_min(1.0)
        epsilon_loss = ((prediction - target).pow(2) * weight).sum() / denominator
        count = mask.reshape(batch_size, -1).sum(dim=1).to(torch.long)
        count_loss = functional.cross_entropy(count_logits, count)
        metrics: GridMetric = {
            "epsilon_loss": epsilon_loss,
            "count_loss": count_loss,
            "timestep": timestep,
        }
        total = epsilon_loss + count_loss
        if self.gumbel_schedule is not None:
            per_sample_denominator = (
                weight.sum(dim=(1, 2, 3)) * grid.shape[-1]
            ).clamp_min(1.0)
            per_sample_loss = ((prediction - target).pow(2) * weight).sum(
                dim=(1, 2, 3)
            ) / per_sample_denominator
            scheduler_loss = self.gumbel_schedule.scheduler_loss(timestep, per_sample_loss)
            metrics["scheduler_loss"] = scheduler_loss
            total = total + scheduler_loss
        return total, metrics

    def sample(
        self,
        context_condition: torch.Tensor,
        *,
        segment_count: int | None = None,
        steps: int = 50,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the existing deterministic-or-random latent-grid DDPM trajectory."""
        with torch.no_grad():
            return self._sample(
                context_condition,
                segment_count=segment_count,
                steps=steps,
                seed=seed,
            )

    def sample_repair(
        self,
        z_ctx: torch.Tensor,
        context_condition: torch.Tensor,
        *,
        noise_frac: float = 0.5,
        steps: int = 50,
        seed: int | None = None,
    ) -> torch.Tensor:
        """A5 prior-side repair: partially re-noise ``z_ctx`` and denoise it back.

        ``z_ctx`` is a corruption-degraded chunk grid (the deployable context
        field captured from the corrupted canvas — never the clean target;
        E_deploy is CUT so no clean capture exists at inference). The grid is
        diffused to the schedule point a fraction ``noise_frac`` of the way to
        pure noise and the standard reverse pass runs from there, pulling the
        field toward the clean-field distribution the prior was trained on
        while retaining the context's document-specific information. At
        ``noise_frac=1.0`` this degenerates to :meth:`sample` (pure noise
        start); at ``0.0`` it returns ``z_ctx`` unchanged.

        Accepts ``[B, outer, inner, d_z]`` or flat ``[B, horizon, d_z]``;
        returns the same layout it was given. All grid slots are treated as
        active (the Phase-3 downlink conditions on the full field).
        """
        if not 0.0 <= noise_frac <= 1.0:
            msg = f"noise_frac must be in [0, 1], got {noise_frac}"
            raise ValueError(msg)
        if steps < 1:
            raise positive_steps_error()
        with torch.no_grad():
            full_mask = (
                torch.ones(z_ctx.shape[0], self.config.horizon,
                           dtype=torch.bool, device=z_ctx.device)
                if z_ctx.ndim == _FLAT_DIMENSIONS
                else torch.ones(
                    z_ctx.shape[0], self.config.outer_slots, self.config.inner_slots,
                    dtype=torch.bool, device=z_ctx.device,
                )
            )
            grid, mask, was_flat = self._as_grid(z_ctx, full_mask)
            batch_size = grid.shape[0]
            weight = mask.unsqueeze(-1).to(grid.dtype)
            generator = (
                torch.Generator(device=grid.device).manual_seed(seed)
                if seed is not None else None
            )
            noise = torch.randn(grid.shape, device=grid.device, dtype=grid.dtype,
                                generator=generator)
            if noise_frac == 0.0:
                out = grid
            elif self.config.objective == "rf":
                # compute_diffusion convention: z_t = t*z + (1-t)*eps, clean at
                # t=1; start at t0 = 1 - noise_frac and integrate t0 -> 1.
                t0 = 1.0 - noise_frac
                grid = (t0 * grid + (1.0 - t0) * noise) * weight
                times = torch.linspace(t0, 1.0, steps + 1,
                                       device=grid.device, dtype=grid.dtype)
                self_cond: torch.Tensor | None = None
                for step in range(steps):
                    current = times[step]
                    delta = times[step + 1] - current
                    velocity, _ = self.forward(
                        grid, current.expand(batch_size), context_condition, mask,
                        self_cond=self_cond,
                    )
                    if self.self_cond_proj is not None:
                        expanded = current.expand(batch_size).view(batch_size, 1, 1, 1)
                        self_cond = (grid + (1.0 - expanded) * velocity) * weight
                    grid = (grid + delta * velocity) * weight
                out = grid
            else:
                if self.gumbel_schedule is not None:
                    gammas = self.gumbel_schedule.sampling_gammas(
                        steps, device=grid.device, dtype=grid.dtype
                    )
                    coordinates = gammas
                    alpha_bars = GumbelNoiseSchedule.alpha_bar(gammas).to(grid.dtype)
                else:
                    times = torch.linspace(1.0, 0.0, steps + 1,
                                           device=grid.device, dtype=grid.dtype)
                    coordinates = times
                    alpha_bars = cosine_alpha_bar(times).to(grid.dtype)
                # Enter the reverse trajectory at the index matching noise_frac:
                # index 0 = pure noise, index `steps` = clean, so skipping the
                # first (1 - noise_frac) of the indices re-noises less.
                start = min(int(round((1.0 - noise_frac) * steps)), steps - 1)
                alpha_start = alpha_bars[start].clamp_min(1e-4)
                grid = (
                    alpha_start.sqrt() * grid
                    + (1.0 - alpha_start).sqrt() * noise
                ) * weight
                sc: torch.Tensor | None = None
                for step in range(start, steps):
                    current = coordinates[step]
                    epsilon, _ = self.forward(
                        grid, current.expand(batch_size), context_condition, mask,
                        self_cond=sc,
                    )
                    alpha_current = alpha_bars[step].clamp_min(1e-4)
                    alpha_next = alpha_bars[step + 1].clamp_min(1e-4)
                    clean = (
                        grid - (1.0 - alpha_current).sqrt() * epsilon
                    ) / alpha_current.sqrt()
                    if self.self_cond_proj is not None:
                        sc = clean * weight
                    grid = (
                        alpha_next.sqrt() * clean
                        + (1.0 - alpha_next).sqrt() * epsilon
                    ) * weight
                out = grid
            if was_flat:
                return out.reshape(z_ctx.shape[0], self.config.horizon,
                                   self.config.latent_dim)
            return out

    def _sample(
        self,
        context_condition: torch.Tensor,
        *,
        segment_count: int | None,
        steps: int,
        seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            context_condition.ndim != _CONTEXT_DIMENSIONS
            or context_condition.shape[-1] != self.config.latent_dim
        ):
            raise context_shape_error()
        if steps < 1:
            raise positive_steps_error()
        batch_size = context_condition.shape[0]
        generator = (
            torch.Generator(device=context_condition.device).manual_seed(seed)
            if seed is not None
            else None
        )
        grid = torch.randn(
            (batch_size, self.config.outer_slots, self.config.inner_slots, self.config.latent_dim),
            device=context_condition.device,
            dtype=context_condition.dtype,
            generator=generator,
        )
        count = (
            self.count_head(context_condition).argmax(dim=-1)
            if segment_count is None
            else torch.full(
                (batch_size,), segment_count, dtype=torch.long, device=context_condition.device
            )
        )
        if segment_count is not None and not 0 <= segment_count <= self.config.horizon:
            raise segment_count_error()
        flat_index = torch.arange(self.config.horizon, device=context_condition.device).unsqueeze(0)
        mask = (flat_index < count.unsqueeze(1)).reshape(
            batch_size, self.config.outer_slots, self.config.inner_slots
        )
        grid = grid * mask.unsqueeze(-1).to(grid.dtype)
        weight = mask.unsqueeze(-1).to(grid.dtype)
        self_cond: torch.Tensor | None = None
        if self.config.objective == "rf":
            # compute_diffusion convention: z_t = t*z + (1-t)*eps, velocity = z - eps,
            # so integration runs t: 0 -> 1 with Euler steps on the predicted velocity.
            times = torch.linspace(0.0, 1.0, steps + 1, device=grid.device, dtype=grid.dtype)
            for step in range(steps):
                current = times[step]
                delta = times[step + 1] - current
                velocity, _ = self.forward(
                    grid, current.expand(batch_size), context_condition, mask,
                    self_cond=self_cond,
                )
                if self.self_cond_proj is not None:
                    expanded = current.expand(batch_size).view(batch_size, 1, 1, 1)
                    self_cond = (grid + (1.0 - expanded) * velocity) * weight
                grid = (grid + delta * velocity) * weight
            return grid, mask
        if self.gumbel_schedule is not None:
            gammas = self.gumbel_schedule.sampling_gammas(
                steps, device=grid.device, dtype=grid.dtype
            )
            coordinates = gammas
            alpha_bars = GumbelNoiseSchedule.alpha_bar(gammas).to(grid.dtype)
        else:
            times = torch.linspace(1.0, 0.0, steps + 1, device=grid.device, dtype=grid.dtype)
            coordinates = times
            alpha_bars = cosine_alpha_bar(times).to(grid.dtype)
        for step in range(steps):
            current = coordinates[step]
            epsilon, _ = self.forward(
                grid, current.expand(batch_size), context_condition, mask,
                self_cond=self_cond,
            )
            alpha_current = alpha_bars[step].clamp_min(1e-4)
            alpha_next = alpha_bars[step + 1].clamp_min(1e-4)
            clean = (grid - (1.0 - alpha_current).sqrt() * epsilon) / alpha_current.sqrt()
            if self.self_cond_proj is not None:
                self_cond = clean * weight
            grid = (
                alpha_next.sqrt() * clean + (1.0 - alpha_next).sqrt() * epsilon
            ) * weight
        return grid, mask

    def load_release_denoiser(self, source: TensorDenoiser) -> list[str]:
        """Copy compatible released latent-denoiser weights and report key deltas."""
        source_state = source.state_dict()
        missing_keys = sorted(set(self.denoiser.state_dict()) - set(source_state))
        unexpected_keys = sorted(set(source_state) - set(self.denoiser.state_dict()))
        self._load_denoiser_state(source_state, strict=False)
        return [f"missing:{key}" for key in missing_keys] + [
            f"unexpected:{key}" for key in unexpected_keys
        ]

    def load_release_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        release_identity_manifest: Mapping[str, JsonValue],
        output_root: Path,
        role: str = UPPER_RUNTIME_ROLE,
        require_full_architecture: bool = True,
    ) -> UpperLineage:
        """Load a released denoiser and publish its exact lineage record."""
        if require_full_architecture and self.config != LatentGridConfig():
            detail = f"expected full {LatentGridConfig()}, got {self.config}"
            raise UpperSourceBlockedError.architecture_mismatch(detail)
        trajectory_state = trajectory_state_from_checkpoint(checkpoint_path)
        try:
            self._load_denoiser_state(trajectory_state, strict=True)
        except RuntimeError as error:
            raise UpperSourceBlockedError.architecture_mismatch(str(error)) from error
        lineage = build_upper_lineage(
            UpperLineageRequest(
                checkpoint_path,
                release_identity_manifest,
                output_root,
                role,
                asdict(self.config),
                len(trajectory_state),
            )
        )
        publish_upper_lineage(output_root, lineage)
        return lineage


def parameter_storage_ids(module: nn.Module) -> set[int]:
    """Return backing-storage pointers for a module's registered parameters."""
    return {parameter.untyped_storage().data_ptr() for parameter in module.parameters()}


def assert_distinct_parameter_storage(upper: nn.Module, lower: nn.Module) -> None:
    """Reject accidental parameter or storage sharing between upper and lower models."""
    if parameter_storage_ids(upper) & parameter_storage_ids(lower):
        raise distinct_storage_error()
