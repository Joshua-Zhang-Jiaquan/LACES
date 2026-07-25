"""State-Hijacking RELAY with Denoiser latent diffusion in z-space.

Architecture (training-time):

    text ──no_grad RWKV──► pooled_h
                              │
                          encoder MLP
                              │
                              ▼
                          z_0 ∈ R^d
                              │
       ┌──────────────────────┴──────────────────────┐
       │                                              │
       │ DIFFUSION BRANCH                             │ STATE BRANCH (teacher-forced)
       │ ε ~ N(0,I), t ~ U(0,1)                       │ alpha_predictors(z_0)
       │ z_t = √α_t·z_0 + √(1-α_t)·ε                  │     │
       │ Denoiser(z_t, t) → ε̂                              │     ▼
       │     │                                        │ predicted_state[l]
       │     └── MSE(ε̂, ε)  (latent diffusion loss)   │     │
       │                                              │     ▼
       │                                              │ overwrite cache
       │                                              │     │
       │                                              │ frozen RWKV(text, cache)
       │                                              │     │
       │                                              │     ▼
       │                                              │ text_logits → CE loss (shift-1)
       └──────────────────────────────────────────────┘

Why teacher-force the state branch with clean z_0 (not ẑ_0 derived from ε̂):
  - cleanest gradient path: text CE → encoder
  - matches standard latent-diffusion training: the "decoder" (here the
    state-injected RWKV) sees clean z; the Denoiser is trained separately to
    denoise. At inference, you sample z_T ~ N(0,I), run DDIM to get ẑ_0,
    then plug ẑ_0 into the state branch.

Inference is NOT in this file (single-step ε prediction only). Add a
DDIM sampler when ready to do unconditional generation.
"""
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportConstantRedefinition=false, reportDeprecated=false, reportImplicitOverride=false, reportImplicitStringConcatenation=false
# pyright: reportIndexIssue=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportOperatorIssue=false, reportOptionalCall=false, reportOptionalMemberAccess=false
# pyright: reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportUnusedVariable=false, reportUntypedFunctionDecorator=false
from __future__ import annotations
from typing import List, Optional, Tuple, cast

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import sphere_flow

try:
    from fla.ops.rwkv7 import chunk_rwkv7 as _chunk_rwkv7
    from fla.ops.rwkv7 import fused_mul_recurrent_rwkv7 as _fused_mul_recurrent_rwkv7
except Exception:
    _chunk_rwkv7 = None
    _fused_mul_recurrent_rwkv7 = None


SUPPORTED_LATENT_RWKV_VARIANTS = ("fused_rwkv7", "albatross_goose")


def resolve_latent_rwkv_variant(config: object) -> str:
    variant = str(getattr(config, "latent_rwkv_variant", "fused_rwkv7"))
    if variant not in SUPPORTED_LATENT_RWKV_VARIANTS:
        supported = ", ".join(SUPPORTED_LATENT_RWKV_VARIANTS)
        raise ValueError(
            f"unsupported latent_rwkv_variant={variant!r}; supported variants: {supported}. "
            "Use latent_rwkv_variant=albatross_goose for the native PyTorch Albatross path; "
            "albatross_cuda is intentionally not available in this integration."
        )
    return variant


def _rwkv_head_dim(hidden_size: int) -> int:
    for candidate in (64, 32, 16, 8, 4, 2, 1):
        if hidden_size % candidate == 0:
            return candidate
    raise ValueError(f"hidden_size={hidden_size} must be divisible by a RWKV head dim")


class DenoiserBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 legacy_init: bool = False, use_cond_adaln: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        # Eager build (not lazy) so the optimizer captures it; zero-init = no-op.
        self.adaLN_cond: Optional[nn.Sequential] = None
        if use_cond_adaln:
            self.adaLN_cond = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
            nn.init.zeros_(self.adaLN_cond[1].weight)
            nn.init.zeros_(self.adaLN_cond[1].bias)

        if not legacy_init:
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)
            with torch.no_grad():
                h = self.adaLN[1].out_features // 6
                self.adaLN[1].bias[2 * h:3 * h] = 1.0
                self.adaLN[1].bias[5 * h:6 * h] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod = self.adaLN(t_emb)
        if cond_emb is not None and self.adaLN_cond is not None:
            mod = mod + self.adaLN_cond(cond_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class LatentRWKV7Direction(nn.Module):
    def __init__(self, hidden_size: int, head_dim: Optional[int] = None, low_rank_dim: Optional[int] = None):
        super().__init__()
        if head_dim is None:
            for candidate in (64, 32, 16, 8):
                if hidden_size % candidate == 0:
                    head_dim = candidate
                    break
            else:
                raise ValueError(f"hidden_size={hidden_size} must be divisible by a RWKV head dim")
        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size={hidden_size} not divisible by head_dim={head_dim}")
        if low_rank_dim is None:
            low_rank_dim = max(32, int(round((2.5 * (hidden_size ** 0.5)) / 32) * 32))

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.norm = nn.LayerNorm(hidden_size)
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.r_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_lora = nn.Sequential(
            nn.Linear(hidden_size, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, hidden_size, bias=True),
        )
        self.a_lora = nn.Sequential(
            nn.Linear(hidden_size, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, hidden_size, bias=True),
        )
        self.g_lora = nn.Sequential(
            nn.Linear(hidden_size, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, hidden_size, bias=False),
            nn.Sigmoid(),
        )
        self.k_k = nn.Parameter(torch.full((hidden_size,), 0.71))
        self.k_a = nn.Parameter(torch.full((hidden_size,), 1.02))
        self.r_k = nn.Parameter(torch.full((self.num_heads, self.head_dim), -0.04))
        self.group_norm = nn.GroupNorm(self.num_heads, hidden_size, eps=self.head_dim * 1e-5)

        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.w_lora[-1].weight)
        nn.init.constant_(self.w_lora[-1].bias, -1.0)
        nn.init.zeros_(self.a_lora[-1].weight)
        nn.init.constant_(self.a_lora[-1].bias, -0.2)

    def _token_shift(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        return prev - x

    def _fallback_recurrent(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        B, T, H, K = r.shape
        V = v.shape[-1]
        state = torch.zeros(B, H, K, V, device=r.device, dtype=torch.float32)
        outs = []
        for i in range(T):
            r_i = r[:, i].float()
            w_i = w[:, i].float()
            k_i = k[:, i].float()
            v_i = v[:, i].float()
            kk_i = kk[:, i].float()
            a_i = a[:, i].float()
            state = torch.exp(w_i).unsqueeze(-1) * state
            state = state + (kk_i * a_i).unsqueeze(-1) * ((-kk_i).unsqueeze(-1) * state).sum(dim=-2).unsqueeze(-2)
            state = state + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
            outs.append((state * r_i.unsqueeze(-1)).sum(dim=-2).to(dtype=v.dtype))
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm(x)
        delta = self._token_shift(h)
        xr = h + delta * self.x_r.to(dtype=h.dtype)
        xw = h + delta * self.x_w.to(dtype=h.dtype)
        xk = h + delta * self.x_k.to(dtype=h.dtype)
        xv = h + delta * self.x_v.to(dtype=h.dtype)
        xa = h + delta * self.x_a.to(dtype=h.dtype)
        xg = h + delta * self.x_g.to(dtype=h.dtype)

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = F.normalize((k * self.k_k.to(dtype=k.dtype)).view(B, T, self.num_heads, self.head_dim), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a.to(dtype=k.dtype))
        r = r.view(B, T, self.num_heads, self.head_dim).contiguous()
        w = w.view(B, T, self.num_heads, self.head_dim).contiguous()
        k = k.view(B, T, self.num_heads, self.head_dim).contiguous()
        v = v.view(B, T, self.num_heads, self.head_dim).contiguous()
        a = a.view(B, T, self.num_heads, self.head_dim).contiguous()
        kk = kk.contiguous()

        if self.training and _chunk_rwkv7 is not None and x.is_cuda:
            y, _ = _chunk_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                a=-kk,
                b=kk * a,
                scale=1.0,
                initial_state=None,
                output_final_state=False,
                safe_gate=True,
                chunk_size=64,
            )
        elif _fused_mul_recurrent_rwkv7 is not None and x.is_cuda:
            y, _ = _fused_mul_recurrent_rwkv7(r, w, k, v, kk, a, scale=1.0, output_final_state=False)
        else:
            y = self._fallback_recurrent(r, w, k, v, kk, a)
        y = y.reshape(B, T, C)
        y = F.group_norm(
            y.reshape(B * T, C).float(),
            self.num_heads,
            self.group_norm.weight.float(),
            self.group_norm.bias.float(),
            self.group_norm.eps,
        ).to(dtype=x.dtype).view(B, T, C)
        correction = ((r * k * self.r_k.to(dtype=r.dtype).view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1, keepdim=True) * v).reshape(B, T, C)
        y = (y + correction) * g
        return self.o_proj(y)


class LatentRWKVBlock(nn.Module):
    def __init__(self, hidden_size: int, bidirectional: bool = False, mlp_ratio: float = 4.0,
                 use_cond_adaln: bool = False, variant: str = "fused_rwkv7",
                 layer_id: int = 0, n_layer: int = 1):
        super().__init__()
        self.bidirectional = bidirectional
        self.variant = variant
        if variant == "albatross_goose":
            from albatross_rwkv7 import AlbatrossRWKV7Block

            self.forward_direction = AlbatrossRWKV7Block(
                dim=hidden_size,
                head_dim=_rwkv_head_dim(hidden_size),
                layer_id=layer_id,
                n_layer=n_layer,
                mlp_ratio=mlp_ratio,
            )
        elif variant == "fused_rwkv7":
            self.forward_direction = LatentRWKV7Direction(hidden_size)
        else:
            supported = ", ".join(SUPPORTED_LATENT_RWKV_VARIANTS)
            raise ValueError(f"unsupported latent_rwkv_variant={variant!r}; supported variants: {supported}")
        if bidirectional:
            if variant == "albatross_goose":
                from albatross_rwkv7 import AlbatrossRWKV7Block

                self.backward_direction = AlbatrossRWKV7Block(
                    dim=hidden_size,
                    head_dim=_rwkv_head_dim(hidden_size),
                    layer_id=layer_id,
                    n_layer=n_layer,
                    mlp_ratio=mlp_ratio,
                )
            else:
                self.backward_direction = LatentRWKV7Direction(hidden_size)
            self.direction_fuse = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            nn.init.zeros_(self.direction_fuse.weight)
            with torch.no_grad():
                eye = torch.eye(hidden_size)
                self.direction_fuse.weight[:, :hidden_size].copy_(0.5 * eye)
                self.direction_fuse.weight[:, hidden_size:].copy_(0.5 * eye)
        else:
            self.backward_direction = None
            self.direction_fuse = None
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # Eager build (not lazy) so the optimizer captures it; zero-init = no-op.
        self.adaLN_cond: Optional[nn.Sequential] = None
        if use_cond_adaln:
            self.adaLN_cond = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
            nn.init.zeros_(self.adaLN_cond[1].weight)
            nn.init.zeros_(self.adaLN_cond[1].bias)

    def _direction(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "albatross_goose":
            y, _ = module(x)
            return y - x
        return module(x)

    def forward(self, x: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate_attn = gate_mlp = None
        shift_mlp = scale_mlp = None
        shift_attn = scale_attn = None
        if cond_emb is not None and self.adaLN_cond is not None:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_cond(cond_emb).chunk(6, dim=-1)
        x_attn = x
        if scale_attn is not None:
            x_attn = x * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        y_fwd = self._direction(self.forward_direction, x_attn)
        if self.bidirectional:
            x_rev = torch.flip(x_attn, dims=(1,))
            assert self.backward_direction is not None
            y_bwd = torch.flip(self._direction(self.backward_direction, x_rev), dims=(1,))
            y = self.direction_fuse(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            y = y_fwd
        if gate_attn is not None:
            y = (1 + gate_attn.unsqueeze(1)) * y
        x = x + y
        h = self.norm2(x)
        if scale_mlp is not None:
            h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(h)
        if gate_mlp is not None:
            mlp_out = (1 + gate_mlp.unsqueeze(1)) * mlp_out
        return x + mlp_out


class TrajectoryLatentRWKV(nn.Module):
    def __init__(self, latent_dim: int, horizon: int, hidden_size: int = 256,
                  depth: int = 4, bidirectional: bool = False,
                  use_cond_adaln: bool = False, use_cond_boundary: bool = False,
                  variant: str = "fused_rwkv7"):
        super().__init__()
        if variant not in SUPPORTED_LATENT_RWKV_VARIANTS:
            supported = ", ".join(SUPPORTED_LATENT_RWKV_VARIANTS)
            raise ValueError(f"unsupported latent_rwkv_variant={variant!r}; supported variants: {supported}")
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.latent_rwkv_variant = variant
        self.latent_rwkv_module = "AlbatrossRWKV7Block" if variant == "albatross_goose" else "LatentRWKV7Direction"
        self.latent_rwkv_evidence = f"latent_rwkv_variant={variant} module={self.latent_rwkv_module}"
        self.use_cond_adaln = use_cond_adaln
        self.in_proj = nn.Linear(latent_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.cond_proj = nn.Linear(latent_dim, hidden_size)
        self.cond_proj_adaln = nn.Linear(latent_dim, hidden_size) if use_cond_adaln else None
        if self.cond_proj_adaln is not None:
            nn.init.zeros_(self.cond_proj_adaln.bias)
        self.use_cond_boundary = use_cond_boundary
        if use_cond_boundary:
            self.cond_boundary_proj = nn.Linear(latent_dim, 2 * hidden_size)
        else:
            self.cond_boundary_proj = None
        self.cond_boundary_scale = 1.0  # inference-time knob; default preserves trained behavior
        self.blocks = nn.ModuleList([
            LatentRWKVBlock(
                hidden_size,
                bidirectional=bidirectional,
                use_cond_adaln=use_cond_adaln,
                variant=variant,
                layer_id=index,
                n_layer=depth,
            )
            for index in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, _ = z_t.shape
        if H > self.horizon:
            raise ValueError(f"trajectory length {H} exceeds configured horizon {self.horizon}")
        x = self.in_proj(z_t)
        x = x + self.pos_embed[:, :H].to(dtype=x.dtype)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(self.time_embed(t.to(dtype=x.dtype)))
        cond_emb = None
        if cond is not None:
            if self.use_cond_adaln:
                cond_emb = self.cond_proj_adaln(cond.to(dtype=x.dtype))
            else:
                t_emb = t_emb + self.cond_proj(cond.to(dtype=x.dtype))
        x = x + t_emb.unsqueeze(1)
        if trajectory_mask is not None:
            x = x * trajectory_mask[:, :H].to(dtype=x.dtype).unsqueeze(-1)

        use_boundary = self.use_cond_boundary and self.cond_boundary_proj is not None and cond is not None
        if use_boundary:
            head_tok, tail_tok = self.cond_boundary_proj(cond.to(dtype=x.dtype)).chunk(2, dim=-1)
            scale = float(self.cond_boundary_scale)
            head_tok = head_tok * scale
            tail_tok = tail_tok * scale
            x = torch.cat([head_tok.unsqueeze(1), x, tail_tok.unsqueeze(1)], dim=1)

        for block in self.blocks:
            x = block(x, cond_emb=cond_emb)
            if trajectory_mask is not None:
                if use_boundary:
                    x[:, 1:1 + H] = x[:, 1:1 + H] * trajectory_mask[:, :H].to(dtype=x.dtype).unsqueeze(-1)
                else:
                    x = x * trajectory_mask[:, :H].to(dtype=x.dtype).unsqueeze(-1)

        if use_boundary:
            x = x[:, 1:1 + H]
        return self.out_proj(self.norm(x))


class AuxLatentDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        mid = hidden_size * 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Cosine schedule (same as DDPM++) ──
def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """t in [0,1]; returns ᾱ_t ∈ (0, 1]."""
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f / f0).clamp(min=1e-6, max=1.0)


class LatentDenoiser(nn.Module):
    """Small Denoiser that operates in z-space (R^d).

    Lifts z (R^d) → a length-N hidden sequence → stacked DenoiserBlocks
    conditioned on time embedding → projects back to R^d for ε prediction.
    """

    def __init__(self, latent_dim: int, hidden_size: int = 256,
                 depth: int = 4, num_heads: int = 4, num_tokens: int = 4,
                 legacy_init: bool = False, use_cond_adaln: bool = False,
                 use_cond_boundary: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.use_cond_adaln = use_cond_adaln
        self.use_cond_boundary = use_cond_boundary

        self.in_proj = nn.Linear(latent_dim, hidden_size * num_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.cond_proj = nn.Linear(latent_dim, hidden_size)
        self.cond_proj_adaln = nn.Linear(latent_dim, hidden_size) if use_cond_adaln else None
        if self.cond_proj_adaln is not None:
            nn.init.zeros_(self.cond_proj_adaln.bias)
        if use_cond_boundary:
            self.cond_boundary_proj = nn.Linear(latent_dim, 2 * hidden_size)
        else:
            self.cond_boundary_proj = None
        self.cond_boundary_scale = 1.0

        self.blocks = nn.ModuleList([
            DenoiserBlock(hidden_size, num_heads, legacy_init=legacy_init, use_cond_adaln=use_cond_adaln)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size * num_tokens, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z_t.shape[0]
        x = self.in_proj(z_t).view(B, self.num_tokens, self.hidden_size)
        x = x + self.pos_embed
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(self.time_embed(t.to(dtype=x.dtype)))
        cond_emb = None
        if cond is not None:
            if self.use_cond_adaln:
                cond_emb = self.cond_proj_adaln(cond.to(dtype=x.dtype))
            else:
                t_emb = t_emb + self.cond_proj(cond.to(dtype=x.dtype))

        use_boundary = self.use_cond_boundary and self.cond_boundary_proj is not None and cond is not None
        if use_boundary:
            head_tok, tail_tok = self.cond_boundary_proj(cond.to(dtype=x.dtype)).chunk(2, dim=-1)
            scale = float(self.cond_boundary_scale)
            head_tok = head_tok * scale
            tail_tok = tail_tok * scale
            x = torch.cat([head_tok.unsqueeze(1), x, tail_tok.unsqueeze(1)], dim=1)

        for blk in self.blocks:
            x = blk(x, t_emb, cond_emb=cond_emb)

        if use_boundary:
            x = x[:, 1:1 + self.num_tokens]

        x = self.norm(x).reshape(B, -1)
        return self.out_proj(x)


class TrajectoryLatentTransformer(nn.Module):
    def __init__(self, latent_dim: int, horizon: int, hidden_size: int = 256,
                 depth: int = 4, num_heads: int = 4,
                 legacy_init: bool = False, use_cond_adaln: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.use_cond_adaln = use_cond_adaln
        self.in_proj = nn.Linear(latent_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.cond_proj = nn.Linear(latent_dim, hidden_size)
        self.cond_proj_adaln = nn.Linear(latent_dim, hidden_size) if use_cond_adaln else None
        self.blocks = nn.ModuleList([
            DenoiserBlock(hidden_size, num_heads, legacy_init=legacy_init, use_cond_adaln=use_cond_adaln)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, _ = z_t.shape
        if H > self.horizon:
            raise ValueError(f"trajectory length {H} exceeds configured horizon {self.horizon}")
        key_padding_mask = None
        if trajectory_mask is not None:
            key_padding_mask = ~trajectory_mask[:, :H].bool()
        x = self.in_proj(z_t) + self.pos_embed[:, :H].to(dtype=z_t.dtype)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(self.time_embed(t.to(dtype=x.dtype)))
        cond_emb = None
        if cond is not None:
            if self.use_cond_adaln:
                cond_emb = self.cond_proj_adaln(cond.to(dtype=x.dtype))
            else:
                t_emb = t_emb + self.cond_proj(cond.to(dtype=x.dtype))
        for blk in self.blocks:
            x = blk(x, t_emb, key_padding_mask=key_padding_mask, cond_emb=cond_emb)
        return self.out_proj(self.norm(x))


class TrajectoryStateDecoder(nn.Module):
    def __init__(self, latent_dim: int, horizon: int, num_layers: int, n_basis: int,
                 hidden_size: int = 256, depth: int = 4, num_heads: int = 4,
                 legacy_init: bool = False, block_type: str = "transformer"):
        super().__init__()
        self.horizon = horizon
        self.num_layers = num_layers
        self.n_basis = n_basis
        # "dit" kept as a backward-compat alias for the transformer block type.
        block_type = "transformer" if block_type == "dit" else block_type
        self.block_type = block_type
        self.in_proj = nn.Linear(latent_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.cond = nn.Parameter(torch.zeros(1, hidden_size))
        if block_type == "transformer":
            self.blocks = nn.ModuleList([
                DenoiserBlock(hidden_size, num_heads, legacy_init=legacy_init)
                for _ in range(depth)
            ])
        elif block_type in ("rwkv", "birwkv"):
            self.blocks = nn.ModuleList([
                LatentRWKVBlock(hidden_size, bidirectional=block_type == "birwkv")
                for _ in range(depth)
            ])
        else:
            raise ValueError(f"Unknown trajectory state decoder block_type: {block_type}")
        self.norm = nn.LayerNorm(hidden_size)
        self.alpha_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_basis) for _ in range(num_layers)
        ])
        for head in self.alpha_heads:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        B, H, _ = z.shape
        if H > self.horizon:
            raise ValueError(f"trajectory length {H} exceeds configured horizon {self.horizon}")
        x = self.in_proj(z) + self.pos_embed[:, :H].to(dtype=z.dtype)
        cond = self.cond.to(dtype=x.dtype).expand(B, -1)
        if self.block_type == "transformer":
            for block in self.blocks:
                x = block(x, cond)
        else:
            x = x + cond.unsqueeze(1)
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)
        return [head(x) for head in self.alpha_heads]


class StateHijackingRELAY(nn.Module):
    """Frozen RWKV + Denoiser-diffused latent + direct WKV-state conditioning.

    Returns text logits AND (eps_pred, eps_target) so the train loop can
    compute CE + λ·MSE.
    """

    def __init__(
        self,
        config,
        rwkv_model: nn.Module,
        vocab_size: int,
        latent_dim: int = 32,
        n_basis: int = 16,
        encoder_hidden_mult: int = 4,
        denoiser_hidden: int = 256,
        denoiser_depth: int = 4,
        denoiser_num_heads: int = 4,
        denoiser_num_tokens: int = 4,
        encoder_type: str = "mlp",          # "mlp" | "identity" | "variational"
        alpha_type: str = "linear",         # "linear" | "mlp_trunk"
        alpha_hidden: int = 256,
        latent_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.rwkv_model = rwkv_model
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.n_basis = n_basis
        self.encoder_type = encoder_type
        self._teacher_force_ratio = 1.0
        self._training_stage = 1  # 1=warmup(only clean z), 2=diffusion(only denoised z), 0=blend

        for p in self.rwkv_model.parameters():
            p.requires_grad = False

        rcfg = rwkv_model.config
        self.num_layers = rcfg.num_hidden_layers
        self.hidden_size = rcfg.hidden_size
        self.head_dim = getattr(rcfg, "head_dim", 64)
        if self.hidden_size % self.head_dim != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by head_dim={self.head_dim}"
            )
        self.num_heads = self.hidden_size // self.head_dim

        # ── Encoder: pooled hidden → z_0 ──
        # Same three modes as StateInjectionRELAY: mlp / identity / variational.
        # Variational returns kl_per_dim in extras; trainer applies free_bits +
        # KL warmup and adds eff_kl_weight * KL to total loss.
        mid = self.hidden_size * encoder_hidden_mult
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(self.hidden_size, mid), nn.GELU(),
                nn.Linear(mid, mid), nn.GELU(),
                nn.Linear(mid, latent_dim),
            )
            self.encoder_trunk = self.mu_head = self.logvar_head = None
        elif encoder_type == "identity":
            if latent_dim != self.hidden_size:
                raise ValueError(
                    f"identity encoder requires latent_dim == hidden_size, "
                    f"got {latent_dim} vs {self.hidden_size}"
                )
            self.encoder = self.encoder_trunk = self.mu_head = self.logvar_head = None
            if latent_stats_path:
                stats = torch.load(latent_stats_path, map_location="cpu")
                mu = stats["mu"].float() if "mu" in stats else stats["mean"].float()
                sigma = stats["sigma"].float() if "sigma" in stats else stats["std"].float()
                sigma = sigma.clamp(min=1e-6)
            else:
                mu = torch.zeros(self.hidden_size)
                sigma = torch.ones(self.hidden_size)
            self.register_buffer("latent_mu", mu)
            self.register_buffer("latent_sigma", sigma)
        elif encoder_type == "variational":
            self.encoder = None
            # Direction-2 (multi-backbone shared S0): a per-backbone input adapter maps this
            # backbone's hidden_size to a canonical width, so 0.4B/2.9B/13.3B (hidden 1024/2560/
            # 4096) can feed ONE shared encoder_trunk producing an aligned R^d latent. Only the
            # adapter is backbone-specific; trunk/mu/logvar are shareable. Gated by
            # config.shared_s0_canonical_hidden (default None => adapter is Identity, trunk sized
            # on this backbone's own hidden_size => byte-identical to the single-backbone model).
            canonical_hidden = getattr(config, "shared_s0_canonical_hidden", None)
            if canonical_hidden is not None:
                self.s0_input_adapter = nn.Linear(self.hidden_size, int(canonical_hidden))
                trunk_in = int(canonical_hidden)
                mid = int(canonical_hidden) * encoder_hidden_mult
            else:
                self.s0_input_adapter = None
                trunk_in = self.hidden_size
            self.encoder_trunk = nn.Sequential(
                nn.Linear(trunk_in, mid), nn.LayerNorm(mid), nn.GELU(),
                nn.Linear(mid, mid), nn.LayerNorm(mid), nn.GELU(),
            )
            self.mu_head = nn.Linear(mid, latent_dim)
            self.logvar_head = nn.Linear(mid, latent_dim)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.constant_(self.logvar_head.bias, -3.0)
            # Aux decoder for stage=3 VAE-only pretraining
            self.aux_decoder = AuxLatentDecoder(latent_dim, self.hidden_size)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.use_cond_adaln = bool(getattr(config, "use_cond_adaln", False))
        self.use_cond_boundary = bool(getattr(config, "use_cond_boundary", False))

        # SIM-CoT (arXiv:2509.20317) step-supervision head. Maps each per-chunk latent
        # z_h to its chunk's pooled hidden state; the MSE forces each z_h to carry a
        # distinct step semantic, countering the sampled-latent homogenization/effective-
        # rank collapse measured in Loop-1. Trained jointly, discarded at inference.
        # Gated by config.simcot_enabled (default off => byte-identical, absent from ckpt).
        self._simcot_pooled_cache: Optional[torch.Tensor] = None
        if bool(getattr(config, "simcot_enabled", False)):
            self.simcot_step_decoder = AuxLatentDecoder(latent_dim, self.hidden_size)

        # ── Latent diffusion (Denoiser in z-space) ──
        self.latent_denoiser = LatentDenoiser(
            latent_dim=latent_dim,
            hidden_size=denoiser_hidden,
            depth=denoiser_depth,
            num_heads=denoiser_num_heads,
            num_tokens=denoiser_num_tokens,
            use_cond_adaln=self.use_cond_adaln,
            use_cond_boundary=self.use_cond_boundary,
        )

        # ── State basis [L, K, H, D, D] ──
        self.state_basis = nn.Parameter(
            torch.randn(self.num_layers, n_basis, self.num_heads,
                        self.head_dim, self.head_dim) * 0.02
        )

        # ── Alpha predictors ──
        self.alpha_type = alpha_type
        if alpha_type == "linear":
            self.alpha_trunk = None
            self.alpha_heads = nn.ModuleList([
                nn.Linear(latent_dim, n_basis) for _ in range(self.num_layers)
            ])
            for lin in self.alpha_heads:
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
        elif alpha_type == "mlp_trunk":
            self.alpha_trunk = nn.Sequential(
                nn.Linear(latent_dim, alpha_hidden), nn.GELU(),
                nn.Linear(alpha_hidden, alpha_hidden), nn.GELU(),
            )
            self.alpha_heads = nn.ModuleList([
                nn.Linear(alpha_hidden, n_basis) for _ in range(self.num_layers)
            ])
            for lin in self.alpha_heads:
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
        else:
            raise ValueError(f"Unknown alpha_type: {alpha_type}")

        self.state_scale = nn.Parameter(torch.zeros(1))
        self.xchunk_decay_logit = nn.Parameter(torch.zeros(1))
        self.xchunk_gate = nn.Parameter(torch.zeros(1))

        self.trajectory_enabled = bool(getattr(config, "trajectory_mode", False))
        self.trajectory_chunk_size = int(getattr(config, "trajectory_chunk_size", 32))
        self.trajectory_horizon = int(getattr(config, "trajectory_horizon", 16))
        self.trajectory_s1_mode = str(getattr(config, "trajectory_s1_mode", "independent"))
        self.trajectory_state_blend = float(getattr(config, "trajectory_state_blend", 1.0))
        self.use_learnable_blend = bool(getattr(config, "use_learnable_blend", False))
        self.blend_floor = float(getattr(config, "blend_floor", 0.05))
        if self.use_learnable_blend:
            init_blend = min(max(self.trajectory_state_blend, 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init_blend / (1.0 - init_blend))
            self.blend_gate_logit = nn.Parameter(torch.full((self.num_layers,), init_logit))
        else:
            self.register_parameter("blend_gate_logit", None)
        trajectory_denoiser_type = str(getattr(config, "trajectory_denoiser_type", "transformer"))
        if trajectory_denoiser_type == "dit":
            trajectory_denoiser_type = "transformer"
        self.latent_rwkv_variant = resolve_latent_rwkv_variant(config)
        self.latent_rwkv_module = "AlbatrossRWKV7Block" if self.latent_rwkv_variant == "albatross_goose" else "LatentRWKV7Direction"
        self.latent_rwkv_evidence = f"latent_rwkv_variant={self.latent_rwkv_variant} module={self.latent_rwkv_module}"
        trajectory_state_decoder_type = str(getattr(config, "trajectory_state_decoder_type", "transformer"))
        if trajectory_state_decoder_type == "dit":
            trajectory_state_decoder_type = "transformer"
        if self.trajectory_s1_mode in ("rwkv", "birwkv"):
            trajectory_state_decoder_type = self.trajectory_s1_mode
        trajectory_state_hidden = int(getattr(config, "trajectory_state_hidden", denoiser_hidden))
        trajectory_state_depth = int(getattr(config, "trajectory_state_depth", denoiser_depth))
        trajectory_state_num_heads = int(getattr(config, "trajectory_state_num_heads", denoiser_num_heads))
        if self.trajectory_enabled and trajectory_denoiser_type == "transformer":
            self.trajectory_denoiser = TrajectoryLatentTransformer(
                latent_dim=latent_dim,
                horizon=self.trajectory_horizon,
                hidden_size=denoiser_hidden,
                depth=denoiser_depth,
                num_heads=denoiser_num_heads,
                use_cond_adaln=self.use_cond_adaln,
            )
        elif self.trajectory_enabled and trajectory_denoiser_type in ("rwkv", "birwkv"):
            self.trajectory_denoiser = TrajectoryLatentRWKV(
                latent_dim=latent_dim,
                horizon=self.trajectory_horizon,
                hidden_size=denoiser_hidden,
                depth=denoiser_depth,
                bidirectional=trajectory_denoiser_type == "birwkv",
                use_cond_adaln=self.use_cond_adaln,
                use_cond_boundary=self.use_cond_boundary,
                variant=self.latent_rwkv_variant,
            )
        elif self.trajectory_enabled:
            raise ValueError(f"Unknown trajectory_denoiser_type: {trajectory_denoiser_type}")
        else:
            self.trajectory_denoiser = None
        self.trajectory_state_decoder = TrajectoryStateDecoder(
            latent_dim=latent_dim,
            horizon=self.trajectory_horizon,
            num_layers=self.num_layers,
            n_basis=n_basis,
            hidden_size=trajectory_state_hidden,
            depth=trajectory_state_depth,
            num_heads=trajectory_state_num_heads,
            block_type=trajectory_state_decoder_type,
        ) if self.trajectory_enabled and self.trajectory_s1_mode in ("transformer", "rwkv", "birwkv") else None

    # ── VAE-only forward (stage 3): encoder → z → aux_decoder → recon loss ──
    def _vae_forward(self, text_tokens, attention_mask=None):
        B = text_tokens.shape[0]
        device = text_tokens.device

        with torch.no_grad():
            out = self.rwkv_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            h_last = out.hidden_states[-1]
            if attention_mask is not None:
                m = attention_mask.to(h_last.dtype).unsqueeze(-1)
                pooled = (h_last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                pooled = h_last.mean(dim=1)

        pooled_enc = pooled.to(next(self.encoder_trunk.parameters()).dtype)
        if getattr(self, "s0_input_adapter", None) is not None:
            pooled_enc = self.s0_input_adapter(pooled_enc)
        h = self.encoder_trunk(pooled_enc)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-10.0, max=10.0)
        mu_dec = mu.to(next(self.aux_decoder.parameters()).dtype)
        logvar_dec = logvar.to(next(self.aux_decoder.parameters()).dtype)
        std = (0.5 * logvar_dec).exp()
        z = mu_dec + std * torch.randn_like(std)
        pooled_hat = self.aux_decoder(z)
        recon_loss = F.mse_loss(pooled_hat, pooled.detach().to(pooled_hat.dtype))
        kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - logvar - 1.0)

        dummy_logits = torch.zeros(B, 2, self.vocab_size, device=device)
        dummy_eps = torch.zeros(B, self.latent_dim, device=device)
        with torch.no_grad():
            extras = {
                "recon_loss": recon_loss,
                "recon_val": float(recon_loss.detach().item()),
                "cosine": F.cosine_similarity(pooled_hat.detach(), pooled.detach().to(pooled_hat.dtype), dim=-1).mean().item(),
                "z_norm": z.detach().norm(dim=-1).mean().item(),
                "kl_per_dim": kl_per_dim,
                "state_norm": 0.0,
                "state_scale": 0.0,
                "eps_pred_norm": 0.0,
                "t_mean": 0.0,
            }
        return dummy_logits, dummy_eps, dummy_eps.clone(), extras

    def _trajectory_view(self, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        B, L = text_tokens.shape
        C = self.trajectory_chunk_size
        H = self.trajectory_horizon
        usable = min(L, C * H)
        usable = (usable // C) * C
        if usable < C:
            raise ValueError(f"trajectory_mode requires at least {C} tokens, got {L}")
        H_eff = usable // C
        tokens = text_tokens[:, :usable].reshape(B, H_eff, C)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, :usable].reshape(B, H_eff, C)
        return tokens, mask, H_eff, C

    def _encode_trajectory_chunks(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        chunks, chunk_mask, H_eff, C = self._trajectory_view(text_tokens, attention_mask)
        B = chunks.shape[0]
        flat_tokens = chunks.reshape(B * H_eff, C)
        flat_mask = chunk_mask.reshape(B * H_eff, C) if chunk_mask is not None else None
        with torch.no_grad():
            out = self.rwkv_model(
                input_ids=flat_tokens,
                attention_mask=flat_mask.bool() if flat_mask is not None else None,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            pooled = self._pool_hidden(out.hidden_states[-1], flat_mask)
        # Learned cross-chunk compression (HLST/H-Net-style): before encoding, fuse each
        # chunk's pooled feature with a causal decayed running sum of earlier chunks, so
        # chunk h carries context from chunks 1..h-1 (fixes RACE cross-paragraph loss from
        # isolated per-chunk pooling). Zero-init gate => byte-identical when disabled.
        if bool(getattr(self, "_s1_xchunk_enabled", False)):
            pooled_seq = pooled.reshape(B, H_eff, -1)
            decay = torch.sigmoid(self.xchunk_decay_logit).to(pooled_seq.dtype)
            gate = torch.tanh(self.xchunk_gate).to(pooled_seq.dtype)
            running = torch.zeros(B, pooled_seq.shape[-1], device=pooled_seq.device, dtype=pooled_seq.dtype)
            fused = []
            for h in range(H_eff):
                fused.append(pooled_seq[:, h] + gate * running)
                running = decay * running + pooled_seq[:, h]
            pooled = torch.stack(fused, dim=1).reshape(B * H_eff, -1)
        z_flat, kl_per_dim = self._encode_pooled(pooled)
        z = z_flat.reshape(B, H_eff, self.latent_dim)
        if kl_per_dim is not None:
            kl_per_dim = kl_per_dim.reshape(B, H_eff, self.latent_dim)
        # Global-anchor (REGLUE-style, arXiv:2512.16636): add a whole-sequence global z
        # to each per-chunk z so trajectory strictly generalizes single-z (H=1 => pure
        # global). Fixes RACE cross-paragraph info loss from per-chunk pooling. Gated.
        if bool(getattr(self, "_s1_global_anchor_enabled", False)):
            with torch.no_grad():
                pooled_global = pooled.reshape(B, H_eff, -1).mean(dim=1)
            z_global, _ = self._encode_pooled(pooled_global)
            z = z + z_global.unsqueeze(1)
        if bool(getattr(self, "_simcot_enabled", False)):
            self._simcot_pooled_cache = pooled.reshape(B, H_eff, -1).detach()
        return z, kl_per_dim, H_eff, C

    def _trajectory_vae_forward(self, text_tokens, attention_mask=None):
        chunks, chunk_mask, H_eff, C = self._trajectory_view(text_tokens, attention_mask)
        B = chunks.shape[0]
        flat_tokens = chunks.reshape(B * H_eff, C)
        flat_mask = chunk_mask.reshape(B * H_eff, C) if chunk_mask is not None else None
        device = text_tokens.device
        with torch.no_grad():
            out = self.rwkv_model(
                input_ids=flat_tokens,
                attention_mask=flat_mask.bool() if flat_mask is not None else None,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            pooled = self._pool_hidden(out.hidden_states[-1], flat_mask)
        pooled_enc = pooled.to(next(self.encoder_trunk.parameters()).dtype)
        if getattr(self, "s0_input_adapter", None) is not None:
            pooled_enc = self.s0_input_adapter(pooled_enc)
        h = self.encoder_trunk(pooled_enc)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-10.0, max=10.0)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        pooled_hat = self.aux_decoder(z.to(next(self.aux_decoder.parameters()).dtype))
        recon_loss = F.mse_loss(pooled_hat, pooled.detach().to(pooled_hat.dtype))
        kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - logvar - 1.0)
        dummy_logits = torch.zeros(B * H_eff, C, self.vocab_size, device=device)
        dummy_eps = torch.zeros(B, H_eff, self.latent_dim, device=device, dtype=z.dtype)
        with torch.no_grad():
            extras = {
                "trajectory_mode": True,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(C),
                "recon_loss": recon_loss,
                "recon_val": float(recon_loss.detach().item()),
                "cosine": F.cosine_similarity(pooled_hat.detach(), pooled.detach().to(pooled_hat.dtype), dim=-1).mean().item(),
                "z_norm": z.detach().norm(dim=-1).mean().item(),
                "kl_per_dim": kl_per_dim.reshape(B, H_eff, self.latent_dim),
                "state_norm": 0.0,
                "state_scale": 0.0,
                "eps_pred_norm": 0.0,
                "t_mean": 0.0,
            }
        return dummy_logits, dummy_eps, dummy_eps.clone(), extras

    def predict_states(self, z: torch.Tensor) -> List[torch.Tensor]:
        states = []
        scale = self.state_scale.to(z.dtype)
        if self.alpha_type == "linear":
            features = z
        else:
            features = self.alpha_trunk(z)
        for l in range(self.num_layers):
            alpha = self.alpha_heads[l](features)
            basis_l = self.state_basis[l].to(z.dtype)
            state_l = torch.einsum('bk,khde->bhde', alpha, basis_l)
            states.append(state_l * scale)
        return states

    def _maybe_noise_condition(self, z: torch.Tensor) -> torch.Tensor:
        # Method B (STAR-DLM-style): during S1 training, perturb the clean encoder z
        # with a random cosine-schedule noise level so S1 learns to decode the noisy
        # z it will actually receive from S2 sampling. Gated (default off => no-op).
        if not (self.training and bool(getattr(self, "_s1_noise_cond_enabled", False))):
            return z
        max_sigma = float(getattr(self, "_s1_noise_cond_max_sigma", 1.0))
        B = z.shape[0]
        t = torch.rand(B, device=z.device, dtype=z.dtype)
        ab = cosine_alpha_bar(t).to(z.dtype)
        view = (B,) + (1,) * (z.dim() - 1)
        sqrt_ab = ab.sqrt().view(view)
        sqrt_1mab = (1.0 - ab).sqrt().view(view) * max_sigma
        return sqrt_ab * z + sqrt_1mab * torch.randn_like(z)

    def predict_trajectory_states(self, z: torch.Tensor) -> List[torch.Tensor]:
        if self.trajectory_state_decoder is None:
            B, H, D = z.shape
            flat_states = self.predict_states(z.reshape(B * H, D))
            return [s.reshape(B, H, self.num_heads, self.head_dim, self.head_dim) for s in flat_states]
        alpha_by_layer = self.trajectory_state_decoder(z)
        states = []
        scale = self.state_scale.to(z.dtype)
        for layer_idx, alpha in enumerate(alpha_by_layer):
            basis_l = self.state_basis[layer_idx].to(z.dtype)
            state_l = torch.einsum('btk,knde->btnde', alpha, basis_l)
            states.append(state_l * scale)
        return states

    def _effective_layer_blend(self, layer_idx: int, blend: float):
        blend_gate_logit = getattr(self, "blend_gate_logit", None)
        if self.use_learnable_blend and blend_gate_logit is not None:
            gate = torch.sigmoid(blend_gate_logit[layer_idx])
            return self.blend_floor + (1.0 - self.blend_floor) * gate
        return float(blend)

    def _anchor_loss_for_chunk(self, cache, states_h, chunk_idx: int) -> Optional[torch.Tensor]:
        if chunk_idx == 0:
            return None
        losses = []
        for layer_idx, planned in enumerate(states_h):
            current = cache.layers[layer_idx].state.get("recurrent_state") if cache.layers[layer_idx].state is not None else None
            if isinstance(current, torch.Tensor):
                planned_f = planned.float()
                current_f = current.detach().float()
                denom = current_f.pow(2).mean().clamp(min=1e-6)
                losses.append((planned_f - current_f).pow(2).mean() / denom)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def blend_into_cache(self, cache, predicted_states, blend: float):
        for l, st in enumerate(predicted_states):
            layer = cache.layers[l]
            if layer.state is None:
                layer.state = {
                    "recurrent_state": None, "attn_state": None,
                    "conv_state": None, "ffn_state": None,
                }
            current = layer.state.get("recurrent_state")
            planned = st.to(torch.float32)
            layer_blend = self._effective_layer_blend(l, blend)
            if isinstance(layer_blend, torch.Tensor):
                if isinstance(current, torch.Tensor):
                    layer.state["recurrent_state"] = current.to(torch.float32) * (1.0 - layer_blend) + planned * layer_blend
                else:
                    layer.state["recurrent_state"] = planned * layer_blend
            else:
                if isinstance(current, torch.Tensor) and 0.0 < layer_blend < 1.0:
                    layer.state["recurrent_state"] = current.to(torch.float32) * (1.0 - layer_blend) + planned * layer_blend
                elif layer_blend <= 0.0 and isinstance(current, torch.Tensor):
                    layer.state["recurrent_state"] = current.to(torch.float32)
                else:
                    layer.state["recurrent_state"] = planned
            for sub_key in ("conv_state", "ffn_state"):
                cs = layer.state.get(sub_key)
                if isinstance(cs, torch.Tensor):
                    layer.state[sub_key] = torch.zeros_like(cs)
            layer._seen_tokens = 0
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = 0
        return cache

    def _detach_rwkv_cache(self, cache):
        def detach_obj(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    obj[key] = detach_obj(value)
                return obj
            if isinstance(obj, list):
                for idx, value in enumerate(obj):
                    obj[idx] = detach_obj(value)
                return obj
            if isinstance(obj, tuple):
                return tuple(detach_obj(value) for value in obj)
            return obj

        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "state"):
                    layer.state = detach_obj(layer.state)
            return cache
        return detach_obj(cache)

    def _s1_self_forcing_enabled(self) -> bool:
        return bool(getattr(self, "_s1_self_forcing", False))

    def _s1_self_forcing_ban_eos_enabled(self) -> bool:
        return bool(getattr(self, "_s1_self_forcing_ban_eos", False))

    def _self_forcing_argmax_token(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._s1_self_forcing_ban_eos_enabled():
            return logits.detach().argmax(dim=-1)
        eos_id = int(getattr(self, "_s1_self_forcing_eos_id", 0))
        if eos_id < 0 or eos_id >= logits.shape[-1]:
            return logits.detach().argmax(dim=-1)
        masked_logits = logits.detach().clone()
        masked_logits[:, eos_id] = torch.finfo(masked_logits.dtype).min
        return masked_logits.argmax(dim=-1)

    def _self_forced_chunk_forward(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: Optional[torch.Tensor],
        cache,
    ):
        p = max(0.0, min(1.0, float(getattr(self, "_s1_self_forcing_prob_effective", 0.0))))
        B, C = chunk_tokens.shape
        logits_steps = []
        prev_logits = None
        for t in range(C):
            ground_truth_token = chunk_tokens[:, t]
            if t == 0 or prev_logits is None or p <= 0.0:
                input_token = ground_truth_token
            else:
                own_token = self._self_forcing_argmax_token(prev_logits)
                if p >= 1.0:
                    use_own = torch.ones(B, device=chunk_tokens.device, dtype=torch.bool)
                else:
                    use_own = torch.rand(B, device=chunk_tokens.device) < p
                if chunk_mask is not None:
                    use_own = use_own & chunk_mask[:, t].bool()
                input_token = torch.where(use_own, own_token, ground_truth_token)
            mask_t = chunk_mask[:, t:t + 1].bool() if chunk_mask is not None else None
            out_t = self.rwkv_model(
                input_ids=input_token.unsqueeze(1),
                attention_mask=mask_t,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out_t.past_key_values
            prev_logits = out_t.logits[:, -1, :]
            logits_steps.append(prev_logits)
        return torch.stack(logits_steps, dim=1), cache

    def forward_trajectory_state(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.trajectory_s1_mode in ("transformer", "rwkv", "birwkv"):
            return self.forward_trajectory_state_rollout(text_tokens, attention_mask)
        z, kl_per_dim, H_eff, C = self._encode_trajectory_chunks(text_tokens, attention_mask)
        B = z.shape[0]
        z_flat = z.reshape(B * H_eff, self.latent_dim)
        chunks, chunk_mask, _, _ = self._trajectory_view(text_tokens, attention_mask)
        flat_tokens = chunks.reshape(B * H_eff, C)
        flat_mask = chunk_mask.reshape(B * H_eff, C) if chunk_mask is not None else None

        with torch.no_grad():
            out_pool = self.rwkv_model(
                input_ids=flat_tokens,
                attention_mask=flat_mask.bool() if flat_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = out_pool.past_key_values

        predicted_states = self.predict_states(z_flat)
        cache = self.inject_into_cache(cache, predicted_states)
        out_main = self.rwkv_model(
            input_ids=flat_tokens,
            attention_mask=flat_mask.bool() if flat_mask is not None else None,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        dummy_eps = torch.zeros(B, H_eff, self.latent_dim, device=text_tokens.device, dtype=z.dtype)
        with torch.no_grad():
            extras = {
                "trajectory_mode": True,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(C),
                "z_norm": z.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z.detach().norm(dim=-1).mean().item(),
                "eps_norm": 0.0,
                "eps_pred_norm": 0.0,
                "state_norm": torch.stack([s.detach().float().norm() for s in predicted_states]).mean().item(),
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": 0.0,
                "alpha_bar_mean": 1.0,
            }
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()
        return out_main.logits, dummy_eps, dummy_eps.clone(), extras

    def forward_trajectory_state_rollout(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        z, kl_per_dim, H_eff, C = self._encode_trajectory_chunks(text_tokens, attention_mask)
        chunks, chunk_mask, _, _ = self._trajectory_view(text_tokens, attention_mask)
        B = z.shape[0]
        layer_states = self.predict_trajectory_states(z)
        with torch.no_grad():
            bootstrap = self.rwkv_model(
                input_ids=chunks[:, 0, :1],
                use_cache=True,
                return_dict=True,
            )
            cache = bootstrap.past_key_values
        logits_by_chunk = []
        blend = self.trajectory_state_blend
        anchor_losses = []
        self_forcing = self._s1_self_forcing_enabled()
        bptt_chunks = max(1, int(getattr(self, "_s1_self_forcing_bptt_chunks", 1)))
        for h in range(H_eff):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            anchor_loss = self._anchor_loss_for_chunk(cache, states_h, h)
            if anchor_loss is not None:
                anchor_losses.append(anchor_loss)
            cache = self.blend_into_cache(cache, states_h, blend)
            mask_h = chunk_mask[:, h].bool() if chunk_mask is not None else None
            if self_forcing:
                logits_h, cache = self._self_forced_chunk_forward(chunks[:, h], mask_h, cache)
                logits_by_chunk.append(logits_h)
                if (h + 1) % bptt_chunks == 0:
                    cache = self._detach_rwkv_cache(cache)
            else:
                out_h = self.rwkv_model(
                    input_ids=chunks[:, h],
                    attention_mask=mask_h,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out_h.past_key_values
                logits_by_chunk.append(out_h.logits)
        actual_vocab_size = logits_by_chunk[0].shape[-1]
        logits = torch.stack(logits_by_chunk, dim=1).reshape(B * H_eff, C, actual_vocab_size)
        dummy_eps = torch.zeros(B, H_eff, self.latent_dim, device=text_tokens.device, dtype=z.dtype)
        with torch.no_grad():
            state_norm = torch.stack([
                layer_state.detach().float().norm() for layer_state in layer_states
            ]).mean().item()
            extras = {
                "trajectory_mode": True,
                "trajectory_s1_mode": self.trajectory_s1_mode,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(C),
                "trajectory_state_blend": float(blend),
                "s1_self_forcing": bool(self_forcing),
                "s1_self_forcing_p_effective": float(getattr(self, "_s1_self_forcing_prob_effective", 0.0)) if self_forcing else 0.0,
                "s1_self_forcing_ban_eos": bool(self._s1_self_forcing_ban_eos_enabled()) if self_forcing else False,
                "s1_self_forcing_bptt_chunks": float(bptt_chunks),
                "s1_self_forcing_token_steps": float(H_eff * C) if self_forcing else 0.0,
                "z_norm": z.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z.detach().norm(dim=-1).mean().item(),
                "eps_norm": 0.0,
                "eps_pred_norm": 0.0,
                "state_norm": state_norm,
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": 0.0,
                "alpha_bar_mean": 1.0,
            }
        if anchor_losses:
            extras["_state_anchor_loss_tensor"] = torch.stack(anchor_losses).mean()
            extras["state_anchor_loss"] = float(extras["_state_anchor_loss_tensor"].detach().item())
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()
        return logits, dummy_eps, dummy_eps.clone(), extras

    def forward_prefix_suffix_trajectory_state_rollout(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        split_idx: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, L = text_tokens.shape
        C = self.trajectory_chunk_size
        split_idx = max(1, min(int(split_idx), L - C))

        prefix_tokens = text_tokens[:, :split_idx]
        suffix_tokens = text_tokens[:, split_idx:]
        prefix_mask = attention_mask[:, :split_idx] if attention_mask is not None else None
        suffix_mask = attention_mask[:, split_idx:] if attention_mask is not None else None

        z, kl_per_dim, H_eff, C = self._encode_trajectory_chunks(suffix_tokens, suffix_mask)
        chunks, chunk_mask, _, _ = self._trajectory_view(suffix_tokens, suffix_mask)
        z = self._maybe_noise_condition(z)
        layer_states = self.predict_trajectory_states(z)

        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values

        logits_by_chunk = []
        blend = self.trajectory_state_blend
        anchor_losses = []
        self_forcing = self._s1_self_forcing_enabled()
        bptt_chunks = max(1, int(getattr(self, "_s1_self_forcing_bptt_chunks", 1)))
        for h in range(H_eff):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            anchor_loss = self._anchor_loss_for_chunk(cache, states_h, h)
            if anchor_loss is not None:
                anchor_losses.append(anchor_loss)
            cache = self.blend_into_cache(cache, states_h, blend)
            mask_h = chunk_mask[:, h].bool() if chunk_mask is not None else None
            if self_forcing:
                logits_h, cache = self._self_forced_chunk_forward(chunks[:, h], mask_h, cache)
                logits_by_chunk.append(logits_h)
                if (h + 1) % bptt_chunks == 0:
                    cache = self._detach_rwkv_cache(cache)
            else:
                out_h = self.rwkv_model(
                    input_ids=chunks[:, h],
                    attention_mask=mask_h,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out_h.past_key_values
                logits_by_chunk.append(out_h.logits)

        actual_vocab_size = logits_by_chunk[0].shape[-1]
        logits = torch.stack(logits_by_chunk, dim=1).reshape(B * H_eff, C, actual_vocab_size)
        dummy_eps = torch.zeros(B, H_eff, self.latent_dim, device=text_tokens.device, dtype=z.dtype)
        with torch.no_grad():
            state_norm = torch.stack([
                layer_state.detach().float().norm() for layer_state in layer_states
            ]).mean().item()
            extras = {
                "prefix_suffix_trajectory_s1": True,
                "trajectory_s1_mode": self.trajectory_s1_mode,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(C),
                "trajectory_state_blend": float(blend),
                "s1_self_forcing": bool(self_forcing),
                "s1_self_forcing_p_effective": float(getattr(self, "_s1_self_forcing_prob_effective", 0.0)) if self_forcing else 0.0,
                "s1_self_forcing_ban_eos": bool(self._s1_self_forcing_ban_eos_enabled()) if self_forcing else False,
                "s1_self_forcing_bptt_chunks": float(bptt_chunks),
                "s1_self_forcing_token_steps": float(H_eff * C) if self_forcing else 0.0,
                "z_norm": z.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z.detach().norm(dim=-1).mean().item(),
                "eps_norm": 0.0,
                "eps_pred_norm": 0.0,
                "state_norm": state_norm,
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": 0.0,
                "alpha_bar_mean": 1.0,
                "split_idx": float(split_idx),
            }
        if anchor_losses:
            extras["_state_anchor_loss_tensor"] = torch.stack(anchor_losses).mean()
            extras["state_anchor_loss"] = float(extras["_state_anchor_loss_tensor"].detach().item())
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()
        return logits, dummy_eps, dummy_eps.clone(), extras

    def _prefix_suffix_trajectory_logits_from_z(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        suffix_tokens: torch.Tensor,
        suffix_mask: Optional[torch.Tensor],
        z: torch.Tensor,
        max_chunks: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        chunks, chunk_mask, H_eff, C = self._trajectory_view(suffix_tokens, suffix_mask)
        H_use = min(int(H_eff), int(z.shape[1]))
        if max_chunks > 0:
            H_use = min(H_use, int(max_chunks))
        z = z[:, :H_use]
        chunks = chunks[:, :H_use]
        if chunk_mask is not None:
            chunk_mask = chunk_mask[:, :H_use]

        layer_states = self.predict_trajectory_states(z)
        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values

        logits_by_chunk = []
        blend = self.trajectory_state_blend
        for h in range(H_use):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            cache = self.blend_into_cache(cache, states_h, blend)
            mask_h = chunk_mask[:, h].bool() if chunk_mask is not None else None
            out_h = self.rwkv_model(
                input_ids=chunks[:, h],
                attention_mask=mask_h,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out_h.past_key_values
            logits_by_chunk.append(out_h.logits)

        actual_vocab_size = logits_by_chunk[0].shape[-1]
        logits = torch.stack(logits_by_chunk, dim=1).reshape(-1, C, actual_vocab_size)
        return logits, chunk_mask

    def _prefix_suffix_trajectory_alignment_kl(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        suffix_tokens: torch.Tensor,
        suffix_mask: Optional[torch.Tensor],
        z_clean: torch.Tensor,
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        max_chunks = int(getattr(self, "_s2_align_max_chunks", 2))
        temperature = max(float(getattr(self, "_s2_align_temperature", 1.0)), 1e-6)

        with torch.no_grad():
            teacher_logits, align_mask = self._prefix_suffix_trajectory_logits_from_z(
                prefix_tokens,
                prefix_mask,
                suffix_tokens,
                suffix_mask,
                z_clean.detach(),
                max_chunks=max_chunks,
            )
        student_logits, _ = self._prefix_suffix_trajectory_logits_from_z(
            prefix_tokens,
            prefix_mask,
            suffix_tokens,
            suffix_mask,
            z_pred,
            max_chunks=max_chunks,
        )

        teacher_logits = teacher_logits[:, :-1, :].float() / temperature
        student_logits = student_logits[:, :-1, :].float() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl = kl * (temperature ** 2)

        if align_mask is not None:
            mask = align_mask[:, :, 1:].reshape(-1).bool()
        else:
            mask = torch.ones(kl.numel(), device=kl.device, dtype=torch.bool)
        kl_flat = kl.reshape(-1)
        if mask.sum().item() == 0:
            return kl_flat.mean() * 0.0
        return (kl_flat * mask.to(kl_flat.dtype)).sum() / mask.sum().clamp(min=1)

    def forward_trajectory_diffusion(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        z_0, kl_per_dim, H_eff, C = self._encode_trajectory_chunks(text_tokens, attention_mask)
        trajectory_loss_mask = None
        if attention_mask is not None:
            _, chunk_mask, _, _ = self._trajectory_view(text_tokens, attention_mask)
            if chunk_mask is not None:
                trajectory_loss_mask = chunk_mask.any(dim=-1).to(z_0.dtype).unsqueeze(-1)
        B = z_0.shape[0]
        device = text_tokens.device
        gen_type = getattr(self, '_gen_type', 'ddpm')
        alpha_bar = torch.ones(B, device=device, dtype=z_0.dtype)
        cond = None
        if bool(getattr(self, "_trajectory_condition_first", False)) and H_eff > 1:
            cond = z_0[:, 0].detach()
        if gen_type in ('flow', 'rf'):
            t = torch.sigmoid(torch.randn(B, device=device, dtype=z_0.dtype) * 0.8 - 0.8)
            t_exp = t.view(B, 1, 1)
            eps_noise = torch.randn_like(z_0)
            z_t = t_exp * z_0 + (1.0 - t_exp) * eps_noise
            if trajectory_loss_mask is not None:
                z_t = z_t * trajectory_loss_mask
            eps_pred = self.trajectory_denoiser(
                z_t,
                t,
                cond=cond,
                trajectory_mask=trajectory_loss_mask.squeeze(-1).bool() if trajectory_loss_mask is not None else None,
            )
            eps_target = z_0 - eps_noise if gen_type == 'rf' else z_0
            if trajectory_loss_mask is not None:
                eps_target = eps_target * trajectory_loss_mask
            z_0_pred = z_t + (1.0 - t_exp) * eps_pred if gen_type == 'rf' else eps_pred
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
        else:
            t = torch.rand(B, device=device, dtype=z_0.dtype)
            alpha_bar = cosine_alpha_bar(t).to(z_0.dtype)
            sqrt_ab = alpha_bar.sqrt().view(B, 1, 1)
            sqrt_1mab = (1.0 - alpha_bar).sqrt().view(B, 1, 1)
            eps_target = torch.randn_like(z_0)
            z_t = sqrt_ab * z_0 + sqrt_1mab * eps_target
            if trajectory_loss_mask is not None:
                z_t = z_t * trajectory_loss_mask
                eps_target = eps_target * trajectory_loss_mask
            eps_pred = self.trajectory_denoiser(
                z_t,
                t,
                cond=cond,
                trajectory_mask=trajectory_loss_mask.squeeze(-1).bool() if trajectory_loss_mask is not None else None,
            )
            z_0_pred = (z_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-4)
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
        with torch.no_grad():
            extras = {
                "trajectory_mode": True,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(C),
                "z_norm": z_0.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z_t.detach().norm(dim=-1).mean().item(),
                "eps_norm": eps_norm,
                "eps_pred_norm": eps_pred.detach().norm(dim=-1).mean().item(),
                "state_norm": 0.0,
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": float(t.detach().mean().item()),
                "alpha_bar_mean": float(alpha_bar.detach().mean().item()),
            }
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()
        if trajectory_loss_mask is not None:
            extras["_trajectory_loss_mask"] = trajectory_loss_mask
        dummy_logits = torch.zeros(B * H_eff, C, self.vocab_size, device=device)
        return dummy_logits, eps_pred, eps_target, extras

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is not None:
            m = attention_mask.to(hidden.dtype).unsqueeze(-1)
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return hidden.mean(dim=1)

    def _encode_pooled(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kl_per_dim = None
        if self.encoder_type == "identity":
            pooled = pooled.to(self.latent_mu.dtype)
            z_0 = (pooled - self.latent_mu) / self.latent_sigma
            z_0 = z_0.to(next(self.alpha_heads.parameters()).dtype)
        elif self.encoder_type == "mlp":
            pooled = pooled.to(next(self.encoder.parameters()).dtype)
            z_0 = self.encoder(pooled)
        elif self.encoder_type == "variational":
            pooled = pooled.to(next(self.encoder_trunk.parameters()).dtype)
            if getattr(self, "s0_input_adapter", None) is not None:
                pooled = self.s0_input_adapter(pooled)
            h = self.encoder_trunk(pooled)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(min=-10.0, max=10.0)
            std = (0.5 * logvar).exp()
            z_0 = mu + std * torch.randn_like(std)
            kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - logvar - 1.0)
        else:
            raise RuntimeError(f"unreachable encoder_type {self.encoder_type}")
        return z_0, kl_per_dim

    def forward_prefix_suffix(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        split_idx: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, L = text_tokens.shape
        device = text_tokens.device
        if L < 3:
            raise ValueError("prefix/suffix S1 requires sequence length >= 3")
        split_idx = max(1, min(int(split_idx), L - 1))

        prefix_context = text_tokens[:, :split_idx]
        suffix_context = text_tokens[:, split_idx - 1:]
        suffix_tokens = text_tokens[:, split_idx:]
        prefix_mask = attention_mask[:, :split_idx] if attention_mask is not None else None
        suffix_mask = attention_mask[:, split_idx:] if attention_mask is not None else None

        with torch.no_grad():
            suffix_out = self.rwkv_model(
                input_ids=suffix_tokens,
                attention_mask=suffix_mask.bool() if suffix_mask is not None else None,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            pooled_suffix = self._pool_hidden(suffix_out.hidden_states[-1], suffix_mask)

            prefix_out = self.rwkv_model(
                input_ids=prefix_context,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values

        z_suffix, kl_per_dim = self._encode_pooled(pooled_suffix)
        predicted_states = self.predict_states(z_suffix)
        cache = self.inject_into_cache(cache, predicted_states)

        out_main = self.rwkv_model(
            input_ids=suffix_context,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )

        dummy_eps = torch.zeros(B, self.latent_dim, device=device, dtype=z_suffix.dtype)
        with torch.no_grad():
            extras = {
                "z_norm": z_suffix.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z_suffix.detach().norm(dim=-1).mean().item(),
                "eps_norm": 0.0,
                "eps_pred_norm": 0.0,
                "state_norm": torch.stack(
                    [s.detach().float().norm() for s in predicted_states]
                ).mean().item(),
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": 0.0,
                "alpha_bar_mean": 1.0,
                "split_idx": float(split_idx),
            }
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()

        return out_main.logits, dummy_eps, dummy_eps.clone(), extras

    def forward_prefix_suffix_diffusion(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        split_idx: int | torch.Tensor = 256,
        z_0_external: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, L = text_tokens.shape
        device = text_tokens.device
        split_is_tensor = isinstance(split_idx, torch.Tensor)
        if split_is_tensor:
            split_flat = split_idx.reshape(-1).to(device=device, dtype=torch.long)
            if split_flat.numel() != B:
                raise ValueError("split_idx tensor must contain one split per batch sample")
            if attention_mask is not None:
                real_lengths = attention_mask.long().sum(dim=1).clamp(min=2)
            else:
                real_lengths = torch.full((B,), L, device=device, dtype=torch.long)
            split_flat = torch.maximum(split_flat, torch.ones_like(split_flat))
            split_flat = torch.minimum(split_flat, (real_lengths - 1).clamp(min=1))
            max_prefix = int(split_flat.max().item())
            suffix_lengths = (real_lengths - split_flat).clamp(min=1)
            max_suffix = int(suffix_lengths.max().item())

            prefix_tokens = text_tokens.new_zeros((B, max_prefix))
            suffix_tokens = text_tokens.new_zeros((B, max_suffix))
            prefix_mask = torch.zeros((B, max_prefix), device=device, dtype=torch.bool)
            suffix_mask = torch.zeros((B, max_suffix), device=device, dtype=torch.bool)
            for b in range(B):
                p_len = int(split_flat[b].item())
                s_len = int(suffix_lengths[b].item())
                real_len = int(real_lengths[b].item())
                prefix_tokens[b, :p_len] = text_tokens[b, :p_len]
                suffix_tokens[b, :s_len] = text_tokens[b, p_len:real_len]
                if attention_mask is not None:
                    prefix_mask[b, :p_len] = attention_mask[b, :p_len].bool()
                    suffix_mask[b, :s_len] = attention_mask[b, p_len:real_len].bool()
                else:
                    prefix_mask[b, :p_len] = True
                    suffix_mask[b, :s_len] = True
            split_idx_value = float(split_flat.float().mean().item())
        else:
            split_idx = max(1, min(int(split_idx), L - 1))
            prefix_tokens = text_tokens[:, :split_idx]
            suffix_tokens = text_tokens[:, split_idx:]
            prefix_mask = attention_mask[:, :split_idx] if attention_mask is not None else None
            suffix_mask = attention_mask[:, split_idx:] if attention_mask is not None else None
            split_idx_value = float(split_idx)

        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            suffix_out = self.rwkv_model(
                input_ids=suffix_tokens,
                attention_mask=suffix_mask.bool() if suffix_mask is not None else None,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            pooled_prefix = self._pool_hidden(prefix_out.hidden_states[-1], prefix_mask)
            pooled_suffix = self._pool_hidden(suffix_out.hidden_states[-1], suffix_mask)

        z_prefix, _ = self._encode_pooled(pooled_prefix)
        z_suffix, kl_per_dim = self._encode_pooled(pooled_suffix)
        if z_0_external is not None:
            z_ext = z_0_external.to(device=device, dtype=z_suffix.dtype)
            if z_ext.dim() == 3:
                z_ext = z_ext.mean(dim=1)
            z_suffix = z_ext
            kl_per_dim = None
        cond = z_prefix.detach()
        drop_prob = float(getattr(self, "_cfg_drop_prob", 0.0))
        if self.training and drop_prob > 0.0:
            keep = torch.rand(B, 1, device=device, dtype=cond.dtype) >= drop_prob
            cond = cond * keep.to(cond.dtype)

        gen_type = getattr(self, '_gen_type', 'ddpm')
        alpha_bar = torch.ones(B, device=device, dtype=z_suffix.dtype)
        if gen_type == 'sphere_rf':
            radius = float(getattr(self, '_sphere_radius', math.sqrt(self.latent_dim)))
            x1 = sphere_flow.project_to_sphere(z_suffix, radius)
            x0 = sphere_flow.sample_sphere_uniform(z_suffix.shape, radius, device, z_suffix.dtype)
            t = torch.sigmoid(torch.randn(B, device=device, dtype=z_suffix.dtype) * 0.8 - 0.8)
            t_exp = t.unsqueeze(-1)
            z_t = sphere_flow.slerp(x0, x1, t_exp, radius)
            v_pred = self.latent_denoiser(z_t, t, cond=cond)
            v_pred = sphere_flow.proju(z_t, v_pred, radius)
            eps_target = sphere_flow.slerp_velocity(x0, x1, t_exp, radius)
            eps_pred = v_pred
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
        elif gen_type in ('flow', 'rf'):
            t = torch.sigmoid(torch.randn(B, device=device, dtype=z_suffix.dtype) * 0.8 - 0.8)
            t_exp = t.unsqueeze(-1)
            eps_noise = torch.randn_like(z_suffix)
            z_t = t_exp * z_suffix + (1.0 - t_exp) * eps_noise
            eps_pred = self.latent_denoiser(z_t, t, cond=cond)
            eps_target = z_suffix - eps_noise if gen_type == 'rf' else z_suffix
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
        else:
            t = torch.rand(B, device=device, dtype=z_suffix.dtype)
            alpha_bar = cosine_alpha_bar(t).to(z_suffix.dtype)
            sqrt_ab = alpha_bar.sqrt().unsqueeze(-1)
            sqrt_1mab = (1.0 - alpha_bar).sqrt().unsqueeze(-1)
            eps_target = torch.randn_like(z_suffix)
            z_t = sqrt_ab * z_suffix + sqrt_1mab * eps_target
            eps_pred = self.latent_denoiser(z_t, t, cond=cond)
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
            z_0_pred_singlez = (z_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-4)

        with torch.no_grad():
            extras = {
                "z_norm": z_suffix.detach().norm(dim=-1).mean().item(),
                "z_prefix_norm": z_prefix.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z_t.detach().norm(dim=-1).mean().item(),
                "eps_norm": eps_norm,
                "eps_pred_norm": eps_pred.detach().norm(dim=-1).mean().item(),
                "state_norm": 0.0,
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": float(t.detach().mean().item()),
                "alpha_bar_mean": float(alpha_bar.detach().mean().item()),
                "split_idx": split_idx_value,
            }
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()

        if bool(getattr(self, "_s2_coadapt_ce_enabled", False)):
            extras["_s2_coadapt_ce_loss_tensor"] = self._singlez_coadapt_ce_from_z(
                prefix_tokens, prefix_mask, suffix_tokens, suffix_mask, z_0_pred_singlez
            )

        dummy_logits = torch.zeros(B, suffix_tokens.shape[1], self.vocab_size, device=device)
        return dummy_logits, eps_pred, eps_target, extras

    def _singlez_coadapt_ce_from_z(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        suffix_tokens: torch.Tensor,
        suffix_mask: Optional[torch.Tensor],
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        states = self.predict_states(z_pred)
        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values
        cache = self.inject_into_cache(cache, states)
        out = self.rwkv_model(
            input_ids=suffix_tokens,
            attention_mask=suffix_mask.bool() if suffix_mask is not None else None,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, :-1, :]
        tgt = suffix_tokens[:, 1:]
        vocab = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, vocab).float(), tgt.reshape(-1), reduction="none")
        if suffix_mask is not None:
            m = suffix_mask[:, 1:].reshape(-1).bool()
            if m.sum().item() == 0:
                return ce.mean() * 0.0
            return (ce * m.to(ce.dtype)).sum() / m.sum().clamp(min=1)
        return ce.mean()

    def forward_prefix_suffix_trajectory_diffusion(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        split_idx: int | torch.Tensor = 256,
        z_0_external: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, L = text_tokens.shape
        device = text_tokens.device
        C = self.trajectory_chunk_size
        split_is_tensor = isinstance(split_idx, torch.Tensor)
        if split_is_tensor:
            split_flat = split_idx.reshape(-1).to(device=device, dtype=torch.long)
            if split_flat.numel() != B:
                raise ValueError("split_idx tensor must contain one split per batch sample")
            if attention_mask is not None:
                real_lengths = attention_mask.long().sum(dim=1).clamp(min=C + 1)
            else:
                real_lengths = torch.full((B,), L, device=device, dtype=torch.long)
            split_flat = torch.maximum(split_flat, torch.ones_like(split_flat))
            split_flat = torch.minimum(split_flat, (real_lengths - C).clamp(min=1))
            max_prefix = int(split_flat.max().item())
            suffix_lengths = (real_lengths - split_flat).clamp(min=C)
            max_suffix = int(suffix_lengths.max().item())

            prefix_tokens = text_tokens.new_zeros((B, max_prefix))
            suffix_tokens = text_tokens.new_zeros((B, max_suffix))
            prefix_mask = torch.zeros((B, max_prefix), device=device, dtype=torch.bool)
            suffix_mask = torch.zeros((B, max_suffix), device=device, dtype=torch.bool)
            for b in range(B):
                p_len = int(split_flat[b].item())
                s_len = int(suffix_lengths[b].item())
                real_len = int(real_lengths[b].item())
                prefix_tokens[b, :p_len] = text_tokens[b, :p_len]
                suffix_tokens[b, :s_len] = text_tokens[b, p_len:real_len]
                if attention_mask is not None:
                    prefix_mask[b, :p_len] = attention_mask[b, :p_len].bool()
                    suffix_mask[b, :s_len] = attention_mask[b, p_len:real_len].bool()
                else:
                    prefix_mask[b, :p_len] = True
                    suffix_mask[b, :s_len] = True
            split_idx_value = float(split_flat.float().mean().item())
        else:
            split_idx = max(1, min(int(split_idx), L - C))
            prefix_tokens = text_tokens[:, :split_idx]
            suffix_tokens = text_tokens[:, split_idx:]
            prefix_mask = attention_mask[:, :split_idx] if attention_mask is not None else None
            suffix_mask = attention_mask[:, split_idx:] if attention_mask is not None else None
            split_idx_value = float(split_idx)

        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            pooled_prefix = self._pool_hidden(prefix_out.hidden_states[-1], prefix_mask)

        z_prefix, _ = self._encode_pooled(pooled_prefix)
        if z_0_external is not None:
            z_0 = z_0_external.to(device=device, dtype=z_prefix.dtype)
            H_eff = int(z_0.shape[1])
            chunk_size = self.trajectory_chunk_size
            kl_per_dim = None
        else:
            z_0, kl_per_dim, H_eff, chunk_size = self._encode_trajectory_chunks(suffix_tokens, suffix_mask)
        trajectory_loss_mask = None
        if split_is_tensor:
            _, chunk_mask, _, _ = self._trajectory_view(suffix_tokens, suffix_mask)
            if chunk_mask is not None:
                trajectory_loss_mask = chunk_mask.any(dim=-1).to(z_0.dtype).unsqueeze(-1)
        cond = z_prefix.detach()
        drop_prob = float(getattr(self, "_cfg_drop_prob", 0.0))
        if self.training and drop_prob > 0.0:
            keep = torch.rand(B, 1, device=device, dtype=cond.dtype) >= drop_prob
            cond = cond * keep.to(cond.dtype)

        gen_type = getattr(self, '_gen_type', 'ddpm')
        alpha_bar = torch.ones(B, device=device, dtype=z_0.dtype)
        if gen_type in ('flow', 'rf'):
            t = torch.sigmoid(torch.randn(B, device=device, dtype=z_0.dtype) * 0.8 - 0.8)
            t_exp = t.view(B, 1, 1)
            eps_noise = torch.randn_like(z_0)
            z_t = t_exp * z_0 + (1.0 - t_exp) * eps_noise
            if trajectory_loss_mask is not None:
                z_t = z_t * trajectory_loss_mask
            eps_pred = self.trajectory_denoiser(
                z_t,
                t,
                cond=cond,
                trajectory_mask=trajectory_loss_mask.squeeze(-1).bool() if trajectory_loss_mask is not None else None,
            )
            eps_target = z_0 - eps_noise if gen_type == 'rf' else z_0
            if trajectory_loss_mask is not None:
                eps_target = eps_target * trajectory_loss_mask
            z_0_pred = z_t + (1.0 - t_exp) * eps_pred if gen_type == 'rf' else eps_pred
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()
        else:
            t = torch.rand(B, device=device, dtype=z_0.dtype)
            alpha_bar = cosine_alpha_bar(t).to(z_0.dtype)
            sqrt_ab = alpha_bar.sqrt().view(B, 1, 1)
            sqrt_1mab = (1.0 - alpha_bar).sqrt().view(B, 1, 1)
            eps_target = torch.randn_like(z_0)
            z_t = sqrt_ab * z_0 + sqrt_1mab * eps_target
            if trajectory_loss_mask is not None:
                z_t = z_t * trajectory_loss_mask
                eps_target = eps_target * trajectory_loss_mask
            eps_pred = self.trajectory_denoiser(
                z_t,
                t,
                cond=cond,
                trajectory_mask=trajectory_loss_mask.squeeze(-1).bool() if trajectory_loss_mask is not None else None,
            )
            z_0_pred = (z_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-4)
            eps_norm = eps_target.detach().norm(dim=-1).mean().item()

        with torch.no_grad():
            extras = {
                "prefix_suffix_trajectory": True,
                "trajectory_horizon": float(H_eff),
                "trajectory_chunk_size": float(chunk_size),
                "z_norm": z_0.detach().norm(dim=-1).mean().item(),
                "z_prefix_norm": z_prefix.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z_t.detach().norm(dim=-1).mean().item(),
                "eps_norm": eps_norm,
                "eps_pred_norm": eps_pred.detach().norm(dim=-1).mean().item(),
                "state_norm": 0.0,
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": float(t.detach().mean().item()),
                "alpha_bar_mean": float(alpha_bar.detach().mean().item()),
                "split_idx": split_idx_value,
            }
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()
        if trajectory_loss_mask is not None:
            extras["_trajectory_loss_mask"] = trajectory_loss_mask
        if bool(getattr(self, "_s2_align_enabled", False)):
            extras["_s2_align_loss_tensor"] = self._prefix_suffix_trajectory_alignment_kl(
                prefix_tokens,
                prefix_mask,
                suffix_tokens,
                suffix_mask,
                z_0,
                z_0_pred,
            )

        # ── Method A: co-adapt CE on SAMPLED-Z (z_0_pred), not the failed logit-KL. ──
        # Feeds the denoiser's predicted x0 (z_0_pred) into the state branch and
        # computes a real teacher-forced CE, so S1 learns to decode the *sampled*
        # latent distribution (exposure-bias fix, cf. CoLA-DLM co-adaptation).
        # Gated by _s2_coadapt_ce_enabled (default off => byte-identical old behavior).
        if bool(getattr(self, "_s2_coadapt_ce_enabled", False)):
            extras["_s2_coadapt_ce_loss_tensor"] = self._trajectory_coadapt_ce_from_z(
                prefix_tokens,
                prefix_mask,
                suffix_tokens,
                suffix_mask,
                z_0_pred,
                H_eff,
                chunk_size,
            )

        # ── LDLM recipe (arXiv:2605.07933): decoder-input noise + hidden-state MSE. ──
        # Perturb clean z_0 with a random cosine-schedule noise level, then MSE-match the
        # states it produces to the clean-z states. This makes S1 natively robust to the
        # noise S2 sampling injects (unlike failed post-hoc KL alignment). Gated (default off).
        if bool(getattr(self, "_s1_ldlm_mse_enabled", False)):
            extras["_s1_ldlm_mse_loss_tensor"] = self._trajectory_ldlm_mse_from_z(z_0)

        if bool(getattr(self, "_simcot_enabled", False)):
            extras["_simcot_step_loss_tensor"] = self._trajectory_simcot_step_from_z(z_0)

        dummy_logits = torch.zeros(B * H_eff, chunk_size, self.vocab_size, device=device)
        return dummy_logits, eps_pred, eps_target, extras

    def _trajectory_simcot_step_from_z(self, z_0: torch.Tensor) -> torch.Tensor:
        pooled = getattr(self, "_simcot_pooled_cache", None)
        if pooled is None:
            return z_0.new_zeros(())
        B, H, _ = z_0.shape
        pred = self.simcot_step_decoder(z_0.reshape(B * H, -1))
        target = pooled.reshape(B * H, -1).to(pred.dtype)
        return F.mse_loss(pred, target)

    def _trajectory_ldlm_mse_from_z(self, z_0: torch.Tensor) -> torch.Tensor:
        max_sigma = float(getattr(self, "_s1_ldlm_max_sigma", 1.0))
        B = z_0.shape[0]
        t = torch.rand(B, device=z_0.device, dtype=z_0.dtype)
        ab = cosine_alpha_bar(t).to(z_0.dtype)
        view = (B,) + (1,) * (z_0.dim() - 1)
        z_noisy = ab.sqrt().view(view) * z_0 + (1.0 - ab).sqrt().view(view) * max_sigma * torch.randn_like(z_0)
        with torch.no_grad():
            clean_states = self.predict_trajectory_states(z_0)
        noisy_states = self.predict_trajectory_states(z_noisy)
        mse = torch.stack([
            F.mse_loss(ns.float(), cs.float()) for ns, cs in zip(noisy_states, clean_states)
        ]).mean()
        return mse

    def _trajectory_coadapt_ce_from_z_forward(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        suffix_tokens: torch.Tensor,
        suffix_mask: Optional[torch.Tensor],
        z_pred: torch.Tensor,
        H_eff: int,
        chunk_size: int,
    ) -> torch.Tensor:
        B = z_pred.shape[0]
        chunks, chunk_mask, _, _ = self._trajectory_view(suffix_tokens, suffix_mask)
        layer_states = self.predict_trajectory_states(z_pred)
        with torch.no_grad():
            prefix_out = self.rwkv_model(
                input_ids=prefix_tokens,
                attention_mask=prefix_mask.bool() if prefix_mask is not None else None,
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values
        blend = self.trajectory_state_blend
        logits_by_chunk = []
        for h in range(H_eff):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            cache = self.blend_into_cache(cache, states_h, blend)
            mask_h = chunk_mask[:, h].bool() if chunk_mask is not None else None
            out_h = self.rwkv_model(
                input_ids=chunks[:, h],
                attention_mask=mask_h,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out_h.past_key_values
            logits_by_chunk.append(out_h.logits)
        vocab = logits_by_chunk[0].shape[-1]
        logits = torch.stack(logits_by_chunk, dim=1).reshape(B * H_eff, chunk_size, vocab)
        tgt = chunks.reshape(B * H_eff, chunk_size)
        logits_shift = logits[:, :-1, :].reshape(-1, vocab)
        tgt_shift = tgt[:, 1:].reshape(-1)
        if chunk_mask is not None:
            m = chunk_mask.reshape(B * H_eff, chunk_size)[:, 1:].reshape(-1).bool()
        else:
            m = torch.ones_like(tgt_shift, dtype=torch.bool)
        ce = F.cross_entropy(logits_shift.float(), tgt_shift, reduction="none")
        if m.sum().item() == 0:
            return ce.mean() * 0.0
        return (ce * m.to(ce.dtype)).sum() / m.sum().clamp(min=1)

    def _trajectory_coadapt_ce_from_z(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        suffix_tokens: torch.Tensor,
        suffix_mask: Optional[torch.Tensor],
        z_pred: torch.Tensor,
        H_eff: int,
        chunk_size: int,
    ) -> torch.Tensor:
        # Gradient-checkpoint the RWKV rollout to avoid a long BPTT graph over
        # H_eff chunks, which OOMs at long context (e.g., 4096 tokens).
        use_checkpoint = bool(getattr(self, "_s2_coadapt_ce_checkpoint", True))
        if not use_checkpoint:
            return self._trajectory_coadapt_ce_from_z_forward(
                prefix_tokens, prefix_mask, suffix_tokens, suffix_mask,
                z_pred, H_eff, chunk_size,
            )

        def _fn(_z_pred: torch.Tensor) -> torch.Tensor:
            return self._trajectory_coadapt_ce_from_z_forward(
                prefix_tokens, prefix_mask, suffix_tokens, suffix_mask,
                _z_pred, H_eff, chunk_size,
            )

        return cast(torch.Tensor, checkpoint.checkpoint(_fn, z_pred, use_reentrant=False))

    # ── Forward ──
    def forward(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z_0_external: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            text_logits: [B, N, V]
            eps_pred:    [B, D]
            eps_target:  [B, D]
            extras:      dict for diagnostics

        z_0_external: optional [B, H, latent_dim] external source latents (e.g. a
            13.3B/0.4B S0 dump) for cross-model latent communication. When provided
            in prefix/suffix trajectory S2, the diffusion forward uses these external
            latents as the clean Z_0 instead of the target model's own S0 encoding.
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        if getattr(self, "_prefix_suffix_trajectory_s1", False) and getattr(self, "_training_stage", 1) == 1:
            return self.forward_prefix_suffix_trajectory_state_rollout(
                text_tokens,
                attention_mask=attention_mask,
                split_idx=getattr(self, "_prefix_suffix_split_idx", 256),
            )
        if getattr(self, "_prefix_suffix_trajectory_s2", False) and getattr(self, "_training_stage", 1) == 2:
            return self.forward_prefix_suffix_trajectory_diffusion(
                text_tokens,
                attention_mask=attention_mask,
                split_idx=getattr(self, "_prefix_suffix_split_idx", 256),
                z_0_external=z_0_external,
            )
        if self.trajectory_enabled and getattr(self, "_trajectory_mode", True):
            training_stage = getattr(self, '_training_stage', 1)
            if training_stage == 3:
                return self._trajectory_vae_forward(text_tokens, attention_mask)
            if training_stage == 1:
                return self.forward_trajectory_state(text_tokens, attention_mask)
            if training_stage == 2:
                return self.forward_trajectory_diffusion(text_tokens, attention_mask)
        if getattr(self, "_prefix_suffix_s1", False) and getattr(self, "_training_stage", 1) == 1:
            return self.forward_prefix_suffix(
                text_tokens,
                attention_mask=attention_mask,
                split_idx=getattr(self, "_prefix_suffix_split_idx", 256),
            )
        if getattr(self, "_prefix_suffix_s2", False) and getattr(self, "_training_stage", 1) == 2:
            return self.forward_prefix_suffix_diffusion(
                text_tokens,
                attention_mask=attention_mask,
                split_idx=getattr(self, "_prefix_suffix_split_idx", 256),
                z_0_external=z_0_external,
            )

        # ── Step 1 (no_grad): pooled hidden + bootstrap Cache ──
        with torch.no_grad():
            out_pool = self.rwkv_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            h_last = out_pool.hidden_states[-1]
            pooled = self._pool_hidden(h_last, attention_mask)
        cache = out_pool.past_key_values

        # ── Step 2: encode pooled → z_0 (encoder mode dependent) ──
        z_0, kl_per_dim = self._encode_pooled(pooled)

        # ── Step 3: Denoiser latent diffusion ──
        # Initialize all variables up-front so diagnostics never crash regardless
        # of stage/gen_type. Stage 1 (Denoiser frozen) keeps these no-op defaults.
        gen_type = getattr(self, '_gen_type', 'ddpm')              # 'ddpm' | 'flow'
        training_stage = getattr(self, '_training_stage', 1)       # 0=blend, 1=warmup, 2=diffusion-only, 3=VAE-only
        if training_stage == 3:
            return self._vae_forward(text_tokens, attention_mask)
        skip_denoiser = training_stage == 1

        t_user = t  # preserve sampler-supplied t (DDIM/flow sampler passes explicit t)
        t = torch.zeros(B, device=device, dtype=z_0.dtype) if t_user is None else t_user
        eps = torch.zeros_like(z_0)
        eps_pred = torch.zeros_like(z_0)
        z_t = z_0.clone()
        z_0_pred = z_0
        alpha_bar = torch.ones(B, device=device, dtype=z_0.dtype)

        if not skip_denoiser:
            if gen_type in ('flow', 'rf'):
                # ELF-style rectified flow:
                #   forward: z_t = t·z_0 + (1-t)·eps   (t=1 clean, t=0 noise)
                #   Denoiser directly outputs z_0_pred (NOT ε)
                #   loss:    MSE(z_0_pred, z_0)
                # We repurpose the (eps_pred, eps) return slots to carry
                # (z_0_pred, z_0) so the trainer's existing
                #   `diff_loss = MSE(eps_pred, eps_target)`
                # works as the flow loss without modification. Diagnostics:
                #   extras["eps_pred_norm"] = ‖z_0_pred‖
                #   extras["eps_norm"]      = ‖z_0‖   (MSE target, not noise)
                if t_user is None:
                    t = torch.sigmoid(torch.randn(B, device=device, dtype=z_0.dtype) * 0.8 - 0.8)
                t_exp = t.unsqueeze(-1)
                eps_noise = torch.randn_like(z_0)
                z_t = t_exp * z_0 + (1.0 - t_exp) * eps_noise
                eps_pred = self.latent_denoiser(z_t, t)
                if gen_type == 'rf':
                    eps = z_0 - eps_noise
                    z_0_pred = z_t + (1.0 - t_exp) * eps_pred
                else:
                    z_0_pred = eps_pred
                    eps = z_0
            else:
                # DDPM (cosine schedule, ε prediction)
                if t_user is None:
                    t = torch.rand(B, device=device, dtype=z_0.dtype)
                alpha_bar = cosine_alpha_bar(t).to(z_0.dtype)
                sqrt_ab = alpha_bar.sqrt().unsqueeze(-1)
                sqrt_1mab = (1.0 - alpha_bar).sqrt().unsqueeze(-1)
                eps = torch.randn_like(z_0)
                z_t = sqrt_ab * z_0 + sqrt_1mab * eps
                eps_pred = self.latent_denoiser(z_t, t)
                # z_0_pred derived in Step 3b for DDPM

        # ── Step 3b: denoised ẑ_0, blend with clean z_0 ──
        teacher_force_ratio = getattr(self, '_teacher_force_ratio', 1.0)
        if training_stage == 1:
            z_blend = z_0
        else:
            if gen_type == 'ddpm':
                # ε-parameterization reverse: ẑ_0 = (z_t - √(1-ᾱ)·ε̂) / √ᾱ
                ab = alpha_bar.clamp(min=1e-8)
                z_0_pred = (z_t - (1 - ab).sqrt().unsqueeze(-1) * eps_pred) / ab.sqrt().unsqueeze(-1)
            # else: flow path already set z_0_pred directly from Denoiser output above
            if training_stage == 2:
                # S2: only train Denoiser, skip state injection and CE
                # Return early with dummy logits — trainer only uses MSE loss
                with torch.no_grad():
                    extras = {
                        "z_norm": z_0.detach().norm(dim=-1).mean().item(),
                        "z_t_norm": z_t.detach().norm(dim=-1).mean().item(),
                        "eps_norm": eps.detach().norm(dim=-1).mean().item(),
                        "eps_pred_norm": eps_pred.detach().norm(dim=-1).mean().item(),
                        "state_norm": 0.0,
                        "state_scale": float(self.state_scale.detach().item()),
                        "t_mean": float(t.detach().mean().item()),
                        "alpha_bar_mean": float(alpha_bar.detach().mean().item()),
                    }
                dummy_logits = torch.zeros(B, text_tokens.shape[1], self.vocab_size, device=device)
                return dummy_logits, eps_pred, eps, extras
            else:  # stage 0 — cosine blend between clean and denoised
                z_blend = teacher_force_ratio * z_0 + (1.0 - teacher_force_ratio) * z_0_pred.detach()

        # ── Step 4: state branch with blended z ──
        predicted_states = self.predict_states(z_blend)

        # ── Step 5: inject into cache, reset counters ──
        for l, st in enumerate(predicted_states):
            layer = cache.layers[l]
            if layer.state is None:
                layer.state = {
                    "recurrent_state": None, "attn_state": None,
                    "conv_state": None, "ffn_state": None,
                }
            layer.state["recurrent_state"] = st.to(torch.float32)
            for sub_key in ("conv_state", "ffn_state"):
                cs = layer.state.get(sub_key)
                if isinstance(cs, torch.Tensor):
                    layer.state[sub_key] = torch.zeros_like(cs)
            layer._seen_tokens = 0
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = 0

        # ── Step 6: real RWKV forward with injected state ──
        out_main = self.rwkv_model(
            input_ids=text_tokens,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        text_logits = out_main.logits

        # ── Diagnostics ──
        with torch.no_grad():
            extras = {
                "z_norm": z_0.detach().norm(dim=-1).mean().item(),
                "z_t_norm": z_t.detach().norm(dim=-1).mean().item(),
                "eps_norm": eps.detach().norm(dim=-1).mean().item(),
                "eps_pred_norm": eps_pred.detach().norm(dim=-1).mean().item(),
                "state_norm": torch.stack(
                    [s.detach().float().norm() for s in predicted_states]
                ).mean().item(),
                "state_scale": float(self.state_scale.detach().item()),
                "t_mean": float(t.detach().mean().item()),
                "alpha_bar_mean": float(alpha_bar.detach().mean().item()),
            }
        # KL term (variational mode only — kept in graph for backward)
        if kl_per_dim is not None:
            extras["kl_per_dim"] = kl_per_dim
            extras["kl_mean"] = kl_per_dim.detach().sum(-1).mean().item()

        return text_logits, eps_pred, eps, extras

    # ── DDIM sampler (inference: z_T ~ N(0,I) → ẑ_0) ──
    def inject_into_cache(self, cache, predicted_states):
        for l, st in enumerate(predicted_states):
            layer = cache.layers[l]
            if layer.state is None:
                layer.state = {
                    "recurrent_state": None, "attn_state": None,
                    "conv_state": None, "ffn_state": None,
                }
            layer.state["recurrent_state"] = st.to(torch.float32)
            for sub_key in ("conv_state", "ffn_state"):
                cs = layer.state.get(sub_key)
                if isinstance(cs, torch.Tensor):
                    layer.state[sub_key] = torch.zeros_like(cs)
            layer._seen_tokens = 0
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = 0
        return cache

    @torch.no_grad()
    def ddim_sample(
        self, num_samples: int, num_steps: int = 50, device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate z_0 from noise."""
        gen_type = getattr(self, '_gen_type', 'ddpm')
        if gen_type == 'flow':
            return self._flow_sample(num_samples, num_steps, device, dtype)
        if gen_type == 'rf':
            return self._rf_sample(num_samples, num_steps, device, dtype)
        return self._ddim_sample(num_samples, num_steps, device, dtype)

    @torch.no_grad()
    def ddpm_sample(
        self, num_samples: int, num_steps: int = 1000, device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate z_0 from noise using standard DDPM (with noise at each step)."""
        gen_type = getattr(self, '_gen_type', 'ddpm')
        if gen_type == 'flow':
            return self._flow_sample(num_samples, num_steps, device, dtype)
        if gen_type == 'rf':
            return self._rf_sample(num_samples, num_steps, device, dtype)
        return self._ddpm_sample(num_samples, num_steps, device, dtype)

    def _flow_sample(self, num_samples, num_steps, device, dtype):
        """Rectified flow sampling: integrate from noise (t=0) to clean (t=1)."""
        device = device or next(self.latent_denoiser.parameters()).device
        dtype = dtype or next(self.latent_denoiser.parameters()).dtype
        z = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device, dtype=dtype)
            z_0_pred = self.latent_denoiser(z, t)
            # Velocity: v = (z_0 - z_t) / (1 - t), but z_t = t*z_0 + (1-t)*noise
            # So v = z_0 - noise = (z_0_pred - z) / (1 - t)
            # Clamp (1-t) to avoid division by zero near t=1
            one_minus_t = (1.0 - t.unsqueeze(-1)).clamp(min=0.1)
            v_pred = (z_0_pred - z) / one_minus_t
            z = z + dt * v_pred
        return z

    def _rf_sample(self, num_samples, num_steps, device, dtype):
        device = device or next(self.latent_denoiser.parameters()).device
        dtype = dtype or next(self.latent_denoiser.parameters()).dtype
        z = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device, dtype=dtype)
            v_pred = self.latent_denoiser(z, t)
            z = z + dt * v_pred
        return z

    def trajectory_sample(self, num_samples: int, num_steps: int = 1000, device=None, dtype=None, sampler: Optional[str] = None):
        gen_type = getattr(self, '_gen_type', 'ddpm')
        if gen_type == 'flow':
            return self._trajectory_flow_sample(num_samples, num_steps, device, dtype)
        if gen_type == 'rf':
            if sampler == 'rf_heun':
                return self._trajectory_rf_heun_sample(num_samples, num_steps, device, dtype)
            return self._trajectory_rf_sample(num_samples, num_steps, device, dtype)
        return self._trajectory_ddpm_sample(num_samples, num_steps, device, dtype)

    def _trajectory_flow_sample(self, num_samples, num_steps, device, dtype):
        device = device or next(self.trajectory_denoiser.parameters()).device
        dtype = dtype or next(self.trajectory_denoiser.parameters()).dtype
        z = torch.randn(
            num_samples, self.trajectory_horizon, self.latent_dim,
            device=device, dtype=dtype,
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device, dtype=dtype)
            z_0_pred = self.trajectory_denoiser(z, t)
            one_minus_t = (1.0 - t.view(num_samples, 1, 1)).clamp(min=0.1)
            z = z + dt * ((z_0_pred - z) / one_minus_t)
        return z

    def _trajectory_rf_sample(self, num_samples, num_steps, device, dtype):
        device = device or next(self.trajectory_denoiser.parameters()).device
        dtype = dtype or next(self.trajectory_denoiser.parameters()).dtype
        z = torch.randn(
            num_samples, self.trajectory_horizon, self.latent_dim,
            device=device, dtype=dtype,
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device, dtype=dtype)
            v_pred = self.trajectory_denoiser(z, t)
            z = z + dt * v_pred
        return z

    def _trajectory_rf_heun_sample(self, num_samples, num_steps, device, dtype):
        device = device or next(self.trajectory_denoiser.parameters()).device
        dtype = dtype or next(self.trajectory_denoiser.parameters()).dtype
        z = torch.randn(
            num_samples, self.trajectory_horizon, self.latent_dim,
            device=device, dtype=dtype,
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device, dtype=dtype)
            t_next = torch.full((num_samples,), min((i + 1) * dt, 1.0), device=device, dtype=dtype)
            v_1 = self.trajectory_denoiser(z, t)
            z_euler = z + dt * v_1
            v_2 = self.trajectory_denoiser(z_euler, t_next)
            z = z + 0.5 * dt * (v_1 + v_2)
        return z

    def _trajectory_ddpm_sample(self, num_samples, num_steps, device, dtype):
        device = device or next(self.trajectory_denoiser.parameters()).device
        dtype = dtype or next(self.trajectory_denoiser.parameters()).dtype
        z = torch.randn(
            num_samples, self.trajectory_horizon, self.latent_dim,
            device=device, dtype=dtype,
        )
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for i in range(num_steps):
            t_cur, t_nxt = ts[i], ts[i + 1]
            ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4, max=1.0)
            ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4, max=1.0)
            alpha_cur = (ab_cur / ab_nxt).clamp(min=1e-4, max=1.0) if i < num_steps - 1 else ab_cur
            eps_pred = self.trajectory_denoiser(z, t_cur.expand(num_samples))
            one_minus_ab_cur = (1 - ab_cur).clamp(min=1e-4)
            mu = (1.0 / alpha_cur.sqrt()) * (
                z - (1 - alpha_cur) / one_minus_ab_cur.sqrt() * eps_pred
            )
            if i < num_steps - 1:
                sigma_sq = ((1 - alpha_cur) * (1 - ab_nxt).clamp(min=1e-4) / one_minus_ab_cur).clamp(min=0.0)
                noise = torch.randn_like(z)
                z = mu + sigma_sq.sqrt() * noise
            else:
                z = mu
        return z

    def _ddim_sample(self, num_samples, num_steps, device, dtype):
        """DDIM deterministic sampling (η=0)."""
        device = device or next(self.latent_denoiser.parameters()).device
        dtype = dtype or next(self.latent_denoiser.parameters()).dtype
        z = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for i in range(num_steps):
            t_cur, t_nxt = ts[i], ts[i + 1]
            ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
            ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
            eps = self.latent_denoiser(z, t_cur.expand(num_samples))
            z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
        return z

    def _ddpm_sample(self, num_samples, num_steps, device, dtype):
        """Standard DDPM sampling with noise at each step (η=1)."""
        device = device or next(self.latent_denoiser.parameters()).device
        dtype = dtype or next(self.latent_denoiser.parameters()).dtype
        z = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for i in range(num_steps):
            t_cur, t_nxt = ts[i], ts[i + 1]
            ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
            ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
            alpha_cur = (ab_cur / ab_nxt).clamp(min=1e-4, max=1.0) if i < num_steps - 1 else ab_cur.clamp(min=1e-4)
            eps_pred = self.latent_denoiser(z, t_cur.expand(num_samples))
            
            ab_cur = ab_cur.clamp(min=1e-4, max=1.0)
            one_minus_ab_cur = (1 - ab_cur).clamp(min=1e-4)
            
            z0_pred = (z - one_minus_ab_cur.sqrt() * eps_pred) / ab_cur.sqrt()
            mu = (1.0 / alpha_cur.sqrt()) * (z - (1 - alpha_cur) / one_minus_ab_cur.sqrt() * eps_pred)
            
            if i < num_steps - 1:
                sigma_sq = ((1 - alpha_cur) * (1 - ab_nxt).clamp(min=1e-4) / one_minus_ab_cur).clamp(min=0.0)
                sigma = sigma_sq.sqrt()
                noise = torch.randn_like(z)
            else:
                sigma = torch.zeros_like(ab_cur)
                noise = torch.zeros_like(z)
            
            z = mu + sigma * noise
        return z
