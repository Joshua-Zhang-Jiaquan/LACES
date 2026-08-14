"""Typed Torch module boundaries for state-hijacking DiT helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class TypedTorchModule(nn.Module):
    """Provide a concrete constructor type over Torch's dynamic module base."""

    if TYPE_CHECKING:
        __init__: Callable[[], None]


class TensorModule(Protocol):
    """A registered module that maps one tensor to one tensor."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Transform one tensor."""
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield parameters for dtype alignment at a module boundary."""
        ...


class GroupNormModule(Protocol):
    """Registered group normalization parameters consumed by functional normalization."""

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the registered normalization scale parameter."""
        ...

    @property
    def bias(self) -> torch.nn.Parameter:
        """Return the registered normalization bias parameter."""
        ...

    eps: float


class AttentionModule(Protocol):
    """A registered attention module with tensor query/key/value inputs."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return attended tensors and optional attention weights."""
        ...


class TimeConditionedModule(Protocol):
    """A registered module conditioned by a per-batch time embedding."""

    def __call__(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Transform a tensor with its conditioning tensor."""
        ...
