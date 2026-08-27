from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


NUM_LAYERS: Final = 32
NUM_HEADS: Final = 40
HEAD_DIM: Final = 64
TOKENS_PER_LAYER: Final = 2
TOKEN_DIM: Final = 256
SUPPORTED_DTYPES: Final = frozenset((torch.float16, torch.bfloat16, torch.float32))


@dataclass(frozen=True, slots=True)
class StateReaderShapeError(ValueError):
    code: str = "expected_state_shape"

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class StateReaderValueError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


def validate_state_token_input(state: torch.Tensor) -> torch.Tensor:
    expected = (NUM_LAYERS, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    if not isinstance(state, torch.Tensor) or state.ndim != 5:
        raise StateReaderShapeError()
    if state.shape[0] <= 0 or tuple(state.shape[1:]) != expected:
        raise StateReaderShapeError()
    if state.dtype not in SUPPORTED_DTYPES:
        raise StateReaderValueError("unsupported_state_dtype")
    return state


class StateTokenConditioner(nn.Module):
    """Stream batch-first WKV matrices into two head-aware tokens per layer."""

    def __init__(self) -> None:
        super().__init__()
        self.row_queries = nn.Parameter(torch.empty(TOKENS_PER_LAYER, HEAD_DIM))
        self.column_queries = nn.Parameter(torch.empty(TOKENS_PER_LAYER, HEAD_DIM))
        self.head_embeddings = nn.Parameter(torch.empty(NUM_HEADS, TOKEN_DIM))
        self.layer_embeddings = nn.Parameter(torch.empty(NUM_LAYERS, TOKENS_PER_LAYER, TOKEN_DIM))
        self.token_embeddings = nn.Parameter(torch.empty(TOKENS_PER_LAYER, TOKEN_DIM))
        self.output_norm = nn.LayerNorm(TOKEN_DIM)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.row_queries, std=HEAD_DIM**-0.5)
        nn.init.normal_(self.column_queries, std=HEAD_DIM**-0.5)
        nn.init.normal_(self.head_embeddings, std=TOKEN_DIM**-0.5)
        nn.init.normal_(self.layer_embeddings, std=0.02)
        nn.init.normal_(self.token_embeddings, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        packed = validate_state_token_input(state)
        if not torch.isfinite(packed).all():
            raise StateReaderValueError("nonfinite_state")
        if packed.device != self.row_queries.device or packed.dtype != self.row_queries.dtype:
            raise StateReaderValueError("state_reader_device_dtype_mismatch")

        layer_tokens: list[torch.Tensor] = []
        for layer_index in range(NUM_LAYERS):
            responses = torch.einsum(
                "tr,bhrc,tc->bht",
                self.row_queries,
                packed[:, layer_index],
                self.column_queries,
            )
            head_aware = torch.einsum("bht,he->bte", responses, self.head_embeddings)
            positioned = head_aware / math.sqrt(NUM_HEADS)
            positioned = positioned + self.layer_embeddings[layer_index] + self.token_embeddings
            layer_tokens.append(self.output_norm(positioned))
        return torch.cat(layer_tokens, dim=1)


__all__ = [
    "HEAD_DIM",
    "NUM_HEADS",
    "NUM_LAYERS",
    "TOKEN_DIM",
    "TOKENS_PER_LAYER",
    "StateReaderShapeError",
    "StateReaderValueError",
    "StateTokenConditioner",
    "validate_state_token_input",
]
