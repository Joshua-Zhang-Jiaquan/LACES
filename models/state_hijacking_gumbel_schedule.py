"""Learnable Gumbel information-uniform noise schedule (LangFlow T1 transplant).

The information-uniform principle (LangFlow, arXiv:2604.11748): training and
sampling density should concentrate where the conditional-entropy profile
H(gamma) changes fastest. Fitting H(gamma) with a Gumbel-shaped cumulative
curve H_inf * exp(-exp(-(gamma - mu) / beta)) makes that density exactly
Gumbel(mu, beta), so sampling gamma from the learned Gumbel distribution IS
the information-uniform rule. gamma is the log noise-to-signal ratio; the
variance-preserving map is sigma^2 = sigmoid(gamma), alpha^2 = sigmoid(-gamma).

The scheduler parameters (mu, beta, H_inf) are fit online by a separate
squared loss against the observed per-sample objective, with the observation
detached (stop-grad) so the fit never feeds gradient back into the denoiser
(v5.3 C1; threshold/effect verdicts stay with the 0.4B pilot, not the
citation).
"""

from __future__ import annotations

from typing import Final, final

import torch
from torch import nn
from typing_extensions import override

__all__: tuple[str, ...] = ("GumbelNoiseSchedule",)

_QUANTILE_CLIP: Final = 1e-5
_MIN_BETA: Final = 1e-3


@final
class GumbelNoiseSchedule(nn.Module):
    """Learnable Gumbel(mu, beta) schedule over gamma = log(sigma^2 / alpha^2)."""

    mu: torch.Tensor
    log_beta: torch.Tensor
    log_h_inf: torch.Tensor

    def __init__(
        self,
        init_mu: float = 0.0,
        init_beta: float = 2.0,
        init_h_inf: float = 1.0,
    ) -> None:
        """Initialize the three scheduler parameters at their fit seeds."""
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(init_mu)))
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).log())
        self.log_h_inf = nn.Parameter(torch.tensor(float(init_h_inf)).log())

    @property
    def beta(self) -> torch.Tensor:
        """Return the positive Gumbel scale."""
        return self.log_beta.exp().clamp_min(_MIN_BETA)

    @property
    def h_inf(self) -> torch.Tensor:
        """Return the positive entropy asymptote."""
        return self.log_h_inf.exp()

    def gamma_from_uniform(self, uniform: torch.Tensor) -> torch.Tensor:
        """Map clipped uniform draws to Gumbel quantiles of gamma."""
        clipped = uniform.clamp(_QUANTILE_CLIP, 1.0 - _QUANTILE_CLIP)
        return self.mu - self.beta * torch.log(-torch.log(clipped))

    def sample_gamma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw training gammas from the learned information-uniform density."""
        uniform = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        return self.gamma_from_uniform(uniform).to(dtype)

    def sampling_gammas(
        self,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``steps + 1`` gammas from high noise to clean via Gumbel quantiles."""
        uniform = torch.linspace(
            1.0 - _QUANTILE_CLIP, _QUANTILE_CLIP, steps + 1, device=device, dtype=dtype
        )
        return self.gamma_from_uniform(uniform).to(dtype)

    @staticmethod
    def alpha_bar(gamma: torch.Tensor) -> torch.Tensor:
        """Return the variance-preserving cumulative alpha for each gamma."""
        return torch.sigmoid(-gamma).clamp(min=1e-6, max=1.0 - 1e-6)

    def entropy_fit(self, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gumbel-CDF-shaped entropy curve at gamma."""
        detached_gamma = gamma.detach()
        return self.h_inf * torch.exp(
            -torch.exp(-(detached_gamma - self.mu) / self.beta)
        )

    def scheduler_loss(
        self, gamma: torch.Tensor, observed_loss: torch.Tensor
    ) -> torch.Tensor:
        """Fit (mu, beta, H_inf) to the observed per-sample loss, stop-grad on the target."""
        return (self.entropy_fit(gamma) - observed_loss.detach()).pow(2).mean()

    @override
    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`entropy_fit` for module-call convenience."""
        return self.entropy_fit(gamma)
