from __future__ import annotations

# pyright: reportAny=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false, reportUnknownMemberType=false

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlbatrossTMixState:
    prev_x: torch.Tensor
    wkv: torch.Tensor


@dataclass
class AlbatrossCMixState:
    prev_x: torch.Tensor


@dataclass
class AlbatrossRWKV7BlockState:
    tmix: AlbatrossTMixState
    cmix: AlbatrossCMixState


def _low_rank_dim(dim: int) -> int:
    return max(4, min(dim * 2, int(round((2.5 * (dim ** 0.5)) / 4) * 4)))


def _validate_sequence(x: torch.Tensor, dim: int) -> None:
    if x.dim() != 3:
        raise ValueError(f"expected a batch-first [batch, seq, dim] tensor, got shape {tuple(x.shape)}")
    if x.shape[-1] != dim:
        raise ValueError(f"expected hidden dim {dim}, got {x.shape[-1]}")


def _shift_delta(x: torch.Tensor, prev_x: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    first_prev = torch.zeros_like(x[:, :1]) if prev_x is None else prev_x.unsqueeze(1).to(device=x.device, dtype=x.dtype)
    shifted = torch.cat([first_prev, x[:, :-1]], dim=1)
    return shifted - x, x[:, -1]


def albatross_recurrent_fallback(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if r.dim() != 4:
        raise ValueError(f"expected [batch, seq, heads, head_dim] tensors, got {tuple(r.shape)}")
    if not (r.shape == w.shape == k.shape == v.shape == a.shape == b.shape):
        raise ValueError("r, w, k, v, a, and b must have matching shapes")
    batch, _, heads, head_dim = r.shape
    if state is None:
        running = torch.zeros(batch, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
    else:
        expected = (batch, heads, head_dim, head_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"expected state shape {expected}, got {tuple(state.shape)}")
        running = state.to(device=r.device, dtype=torch.float32)

    outputs: list[torch.Tensor] = []
    for index in range(r.shape[1]):
        r_i = r[:, index].float()
        w_i = w[:, index].float().clamp(min=-60.0, max=30.0)
        k_i = k[:, index].float()
        v_i = v[:, index].float()
        a_i = a[:, index].float()
        b_i = b[:, index].float()

        recurrent_mix = (running * a_i.unsqueeze(-1)).sum(dim=-2, keepdim=True)
        running = torch.exp(w_i).unsqueeze(-1) * running
        running = running + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
        running = running + b_i.unsqueeze(-1) * recurrent_mix
        outputs.append((running * r_i.unsqueeze(-1)).sum(dim=-2).to(dtype=r.dtype))

    return torch.stack(outputs, dim=1), running


class AlbatrossTMix(nn.Module):
    def __init__(self, dim: int, head_dim: int, layer_id: int, n_layer: int):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.layer_id = layer_id
        self.n_layer = n_layer

        ratio = 1.0 - (layer_id / max(1, n_layer))
        self.norm = nn.LayerNorm(dim)
        self.x_r = nn.Parameter(torch.full((1, 1, dim), 0.08 * ratio))
        self.x_w = nn.Parameter(torch.full((1, 1, dim), 0.12 * ratio))
        self.x_k = nn.Parameter(torch.full((1, 1, dim), 0.10 * ratio))
        self.x_v = nn.Parameter(torch.full((1, 1, dim), 0.07 * ratio))
        self.x_a = nn.Parameter(torch.full((1, 1, dim), 0.09 * ratio))
        self.x_g = nn.Parameter(torch.full((1, 1, dim), 0.06 * ratio))

        rank = _low_rank_dim(dim)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        self.w0 = nn.Parameter(torch.full((dim,), -1.0))
        self.w1 = nn.Linear(dim, rank, bias=False)
        self.w2 = nn.Linear(rank, dim, bias=True)
        self.a0 = nn.Parameter(torch.full((dim,), -0.25))
        self.a1 = nn.Linear(dim, rank, bias=False)
        self.a2 = nn.Linear(rank, dim, bias=True)
        self.v0 = nn.Parameter(torch.full((dim,), 0.05))
        self.v1 = nn.Linear(dim, rank, bias=False)
        self.v2 = nn.Linear(rank, dim, bias=True)
        self.g1 = nn.Linear(dim, rank, bias=False)
        self.g2 = nn.Linear(rank, dim, bias=True)

        self.k_k = nn.Parameter(torch.full((dim,), 0.71))
        self.k_a = nn.Parameter(torch.full((dim,), 1.02))
        self.r_k = nn.Parameter(torch.full((self.num_heads, self.head_dim), -0.04))
        self.group_norm = nn.GroupNorm(self.num_heads, dim, eps=head_dim * 1e-5)

    def forward(self, x: torch.Tensor, *, state: object | None = None) -> tuple[torch.Tensor, AlbatrossTMixState]:
        _validate_sequence(x, self.dim)
        prev_x = state.prev_x if isinstance(state, AlbatrossTMixState) else None
        prev_wkv = state.wkv if isinstance(state, AlbatrossTMixState) else None

        batch, seq, dim = x.shape
        h = self.norm(x)
        delta, next_prev = _shift_delta(h, prev_x)
        xr = h + delta * self.x_r.to(dtype=h.dtype)
        xw = h + delta * self.x_w.to(dtype=h.dtype)
        xk = h + delta * self.x_k.to(dtype=h.dtype)
        xv = h + delta * self.x_v.to(dtype=h.dtype)
        xa = h + delta * self.x_a.to(dtype=h.dtype)
        xg = h + delta * self.x_g.to(dtype=h.dtype)

        r = self.receptance(xr)
        k_raw = self.key(xk)
        value_delta = self.v2(torch.tanh(self.v1(xv))) * self.v0.to(dtype=x.dtype)
        v = self.value(xv) + value_delta
        w = -F.softplus(self.w0.to(dtype=x.dtype) + self.w2(torch.tanh(self.w1(xw)))) - 0.01
        alpha = torch.sigmoid(self.a0.to(dtype=x.dtype) + self.a2(torch.tanh(self.a1(xa))))
        g = torch.sigmoid(self.g2(F.silu(self.g1(xg))))

        kk = F.normalize(k_raw * self.k_k.to(dtype=x.dtype), dim=-1, p=2.0)
        k = k_raw * (1.0 + (alpha - 1.0) * self.k_a.to(dtype=x.dtype))
        r_heads = r.view(batch, seq, self.num_heads, self.head_dim).contiguous()
        w_heads = w.view(batch, seq, self.num_heads, self.head_dim).contiguous()
        k_heads = k.view(batch, seq, self.num_heads, self.head_dim).contiguous()
        v_heads = v.view(batch, seq, self.num_heads, self.head_dim).contiguous()
        a_heads = (-kk).view(batch, seq, self.num_heads, self.head_dim).contiguous()
        b_heads = (kk * alpha).view(batch, seq, self.num_heads, self.head_dim).contiguous()

        y, next_wkv = albatross_recurrent_fallback(
            r_heads,
            w_heads,
            k_heads,
            v_heads,
            a_heads,
            b_heads,
            state=prev_wkv,
        )
        y = y.reshape(batch, seq, dim)
        y = F.group_norm(
            y.reshape(batch * seq, dim).float(),
            self.num_heads,
            self.group_norm.weight.float(),
            self.group_norm.bias.float(),
            self.group_norm.eps,
        ).to(dtype=x.dtype).view(batch, seq, dim)
        correction = ((r_heads * k_heads * self.r_k.to(dtype=x.dtype).view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1, keepdim=True) * v_heads).reshape(batch, seq, dim)
        y = self.output((y + correction) * g)
        return y, AlbatrossTMixState(prev_x=next_prev, wkv=next_wkv)


class AlbatrossCMix(nn.Module):
    def __init__(self, dim: int, layer_id: int, n_layer: int):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.n_layer = n_layer
        hidden = max(dim, int(dim * 4 / 3))
        self.norm = nn.LayerNorm(dim)
        self.x_k = nn.Parameter(torch.full((1, 1, dim), 0.25))
        self.x_r = nn.Parameter(torch.full((1, 1, dim), 0.10))
        self.key = nn.Linear(dim, hidden, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, *, state: object | None = None) -> tuple[torch.Tensor, AlbatrossCMixState]:
        _validate_sequence(x, self.dim)
        prev_x = state.prev_x if isinstance(state, AlbatrossCMixState) else None
        h = self.norm(x)
        delta, next_prev = _shift_delta(h, prev_x)
        xk = h + delta * self.x_k.to(dtype=h.dtype)
        xr = h + delta * self.x_r.to(dtype=h.dtype)
        squared_relu = torch.relu(self.key(xk)).square()
        y = torch.sigmoid(self.receptance(xr)) * self.value(squared_relu)
        return y, AlbatrossCMixState(prev_x=next_prev)


class AlbatrossRWKV7Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, layer_id: int, n_layer: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix = AlbatrossTMix(dim=dim, head_dim=head_dim, layer_id=layer_id, n_layer=n_layer)
        self.channel_mix = AlbatrossCMix(dim=dim, layer_id=layer_id, n_layer=n_layer)
        if mlp_ratio != 4.0:
            hidden = max(dim, int(dim * mlp_ratio))
            self.channel_mix.key = nn.Linear(dim, hidden, bias=False)
            self.channel_mix.value = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, *, state: object | None = None) -> tuple[torch.Tensor, AlbatrossRWKV7BlockState]:
        _validate_sequence(x, self.dim)
        tmix_state = state.tmix if isinstance(state, AlbatrossRWKV7BlockState) else None
        cmix_state = state.cmix if isinstance(state, AlbatrossRWKV7BlockState) else None

        t_out, next_tmix_state = self.time_mix(self.ln1(x), state=tmix_state)
        x = x + t_out
        c_out, next_cmix_state = self.channel_mix(self.ln2(x), state=cmix_state)
        x = x + c_out
        return x, AlbatrossRWKV7BlockState(tmix=next_tmix_state, cmix=next_cmix_state)


__all__ = [
    "AlbatrossTMix",
    "AlbatrossCMix",
    "AlbatrossRWKV7Block",
    "albatross_recurrent_fallback",
]
