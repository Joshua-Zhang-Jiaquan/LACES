"""BiRWKV-7 token-level masked-diffusion denoiser.

The BiRWKV IS the denoiser: corrupted tokens go in, clean-token logits come out
of this model's own lm_head. There is no frozen renderer and no post-diffusion
causal generator (code/plan section 1.1 / 4.6 contract).

Design
------
Each layer holds TWO complete fla ``RWKV7Attention`` modules (forward + reverse)
plus the pretrained ``RWKV7FeedForward`` and norms. The reverse module runs on
the flipped sequence. Outputs fuse through a learned sigmoid gate initialised
strongly toward the forward direction, so at init (or with ``force_forward``)
the network is numerically ~identical to the pretrained causal RWKV-7 — which
is what makes the HF warm-start parity test possible.

Warm-start: both attention copies, the FFN, all norms, the embedding and the
lm_head load from the HF ``RWKV7ForCausalLM`` safetensors (fla-format keys,
e.g. ``model.layers.{i}.attn.r_proj.weight``). The reverse copy is a clone of
the forward weights (code/plan section 4.2 recipe).

Requires the ``fla`` package with CUDA (relay2:v2 image). Cannot import on a
CPU-only box: fla's triton kernels need an active GPU driver.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import torch
from fla.layers.rwkv7 import RWKV7Attention
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward
from torch import nn
from torch.utils import checkpoint as _torch_checkpoint
from typing_extensions import override

from models.torch_types import TypedTorchModule

MASK_TOKEN_ID = 65535  # unused slot in the RWKV World vocab (EOS=65530, max data id 65530)
PAD_TOKEN_ID = 0
_MIN_KENDALL_POINTS = 2

_F = TypeVar("_F", bound=Callable[..., object])


def _no_grad(func: _F) -> _F:
    """Type-preserving ``torch.no_grad`` decorator (torch's own TypeVar is Any-bound)."""
    return cast(_F, torch.no_grad()(func))


class BiRWKV7Block(TypedTorchModule):
    """RWKV7Block with dual-direction attention and gated fusion.

    Key names mirror fla's RWKV7Block (attn_norm / ffn / ffn_norm / pre_norm)
    so pretrained weights load with a pure key remap; the two attention copies
    live at ``attn_fwd`` / ``attn_bwd``.
    """

    if TYPE_CHECKING:
        __call__: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, bool],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]

    def __init__(self, config: RWKV7Config, layer_idx: int, gate_bias_init: float = 4.0) -> None:
        """Build one dual-direction block from the fla config."""
        super().__init__()
        self.config: RWKV7Config = config
        self.layer_idx: int = layer_idx
        hidden = config.hidden_size

        if config.norm_first and layer_idx == 0:
            self.pre_norm: nn.LayerNorm = nn.LayerNorm(
                hidden, bias=config.norm_bias, eps=config.norm_eps
            )
        self.attn_norm: nn.LayerNorm = nn.LayerNorm(
            hidden, bias=config.norm_bias, eps=config.norm_eps
        )

        def _make_attn() -> RWKV7Attention:
            return RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=hidden,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers,
            )

        self.attn_fwd: RWKV7Attention = _make_attn()
        self.attn_bwd: RWKV7Attention = _make_attn()

        # sigmoid fusion gate: alpha = sigmoid(W x + b); W zero-init, b = +gate_bias_init
        # -> alpha ~= sigmoid(4.0) ~= 0.982 forward at init (near-causal start).
        self.fuse_proj: nn.Linear = nn.Linear(hidden, hidden, bias=False)
        _ = nn.init.zeros_(self.fuse_proj.weight)
        self.fuse_bias: nn.Parameter = nn.Parameter(torch.full((hidden,), float(gate_bias_init)))

        self.ffn_norm: nn.LayerNorm = nn.LayerNorm(
            hidden, bias=config.norm_bias, eps=config.norm_eps
        )
        self.ffn: RWKV7FeedForward = RWKV7FeedForward(
            hidden_size=hidden,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers,
        )

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first_fwd: torch.Tensor,
        v_first_bwd: torch.Tensor,
        force_forward: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse forward and reverse attention streams around the pretrained FFN."""
        residual: torch.Tensor = (
            self.pre_norm(hidden_states) if hasattr(self, "pre_norm") else hidden_states
        )
        x: torch.Tensor = self.attn_norm(residual)

        o_fwd, _, _, v_first_fwd = cast(
            "tuple[torch.Tensor, None, object | None, torch.Tensor]",
            self.attn_fwd(hidden_states=x, v_first=v_first_fwd),
        )
        if force_forward:
            o = o_fwd
        else:
            # reverse stream operates entirely in flipped coordinates; its
            # v_first stays flipped across layers.
            x_rev = torch.flip(x, dims=(1,))
            o_bwd, _, _, v_first_bwd = cast(
                "tuple[torch.Tensor, None, object | None, torch.Tensor]",
                self.attn_bwd(hidden_states=x_rev, v_first=v_first_bwd),
            )
            o_bwd = torch.flip(o_bwd, dims=(1,))
            gate_pre: torch.Tensor = self.fuse_proj(x)
            alpha = torch.sigmoid(gate_pre + self.fuse_bias.to(x.dtype))
            o = alpha * o_fwd + (1.0 - alpha) * o_bwd

        hidden_states = residual + o
        residual = hidden_states
        ffn_in: torch.Tensor = self.ffn_norm(hidden_states)
        ffn_out, _ = cast("tuple[torch.Tensor, object | None]", self.ffn(ffn_in))
        hidden_states = residual + ffn_out
        return hidden_states, v_first_fwd, v_first_bwd

    def mean_forward_alpha(self) -> float:
        """Diagnostic: gate bias midpoint (input-independent part only)."""
        return float(torch.sigmoid(self.fuse_bias.detach().float()).mean())


class _GradCheckpointer(Protocol):
    """Typed surface of ``torch.utils.checkpoint.checkpoint`` for block calls."""

    def __call__(  # noqa: PLR0913
        self,
        block: BiRWKV7Block,
        hidden_states: torch.Tensor,
        v_first_fwd: torch.Tensor,
        v_first_bwd: torch.Tensor,
        force_forward: bool,  # noqa: FBT001
        *,
        use_reentrant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


_grad_checkpoint = cast(
    _GradCheckpointer,
    getattr(_torch_checkpoint, "checkpoint"),  # noqa: B009
)


class BiRWKV7ForMaskedDiffusion(TypedTorchModule):
    """Token denoiser: embeddings -> BiRWKV7 blocks -> norm -> lm_head."""

    if TYPE_CHECKING:
        __call__: Callable[[torch.Tensor, bool], torch.Tensor]

    def __init__(
        self,
        config: RWKV7Config,
        gate_bias_init: float = 4.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Assemble embeddings, dual-direction blocks, final norm, and lm_head."""
        super().__init__()
        self.config: RWKV7Config = config
        self.mask_token_id: int = MASK_TOKEN_ID
        self.pad_token_id: int = PAD_TOKEN_ID
        self.gradient_checkpointing: bool = gradient_checkpointing

        self.embeddings: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        blocks = tuple(
            BiRWKV7Block(config, i, gate_bias_init) for i in range(config.num_hidden_layers)
        )
        self.layers: nn.ModuleList = nn.ModuleList(blocks)
        self._blocks: tuple[BiRWKV7Block, ...] = blocks
        self.norm: nn.LayerNorm = nn.LayerNorm(
            config.hidden_size, bias=config.norm_bias, eps=config.norm_eps
        )
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        force_forward: bool = False,
    ) -> torch.Tensor:
        """Return [B, T, vocab] logits.

        Padding must be handled by the caller's loss mask; RWKV state simply
        flows through pad positions.
        """
        h: torch.Tensor = self.embeddings(input_ids)
        v_first_fwd = torch.zeros_like(h)
        v_first_bwd = torch.zeros_like(h)
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                h, v_first_fwd, v_first_bwd = _grad_checkpoint(
                    block,
                    h,
                    v_first_fwd,
                    v_first_bwd,
                    force_forward,
                    use_reentrant=False,
                )
            else:
                h, v_first_fwd, v_first_bwd = block(h, v_first_fwd, v_first_bwd, force_forward)
        normed_h: torch.Tensor = self.norm(h)
        logits: torch.Tensor = self.lm_head(normed_h)
        return logits

    def mean_forward_alpha(self) -> float:
        """Average the per-block forward-gate midpoint across all layers."""
        return sum(b.mean_forward_alpha() for b in self.layers) / len(self.layers)

    # ------------------------------------------------------------------
    # HF warm-start
    # ------------------------------------------------------------------
    @staticmethod
    def remap_hf_key(key: str) -> list[str]:
        """Map one HF RWKV7ForCausalLM key to our key(s).

        Attention weights map to BOTH direction copies (reverse = clone of
        forward).
        """
        if key.startswith("model.layers.") and ".attn." in key and ".attn_norm" not in key:
            head, tail = key.split(".attn.", 1)
            idx = head.removeprefix("model.layers.")
            return [f"layers.{idx}.attn_fwd.{tail}", f"layers.{idx}.attn_bwd.{tail}"]
        if key.startswith("model."):
            return [key.removeprefix("model.")]
        return [key]  # lm_head.weight

    @classmethod
    def from_hf_pretrained(
        cls,
        model_dir: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        gate_bias_init: float = 4.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
    ) -> BiRWKV7ForMaskedDiffusion:
        """Warm-start from an HF RWKV7ForCausalLM directory via key remap."""
        from safetensors import torch as _safetensors_torch

        load_file = cast(
            "Callable[[str], dict[str, torch.Tensor]]",
            getattr(_safetensors_torch, "load_file"),  # noqa: B009
        )

        model_dir = Path(model_dir)
        cfg_json = cast("dict[str, object]", json.loads((model_dir / "config.json").read_text()))
        _ = cfg_json.pop("auto_map", None)
        _ = cfg_json.pop("architectures", None)
        # Build via kwargs: __init__ derives num_heads and the per-layer
        # value_dim list; post-hoc setattr would leave them stale (GroupNorm
        # divisibility crash at 2.9B geometry).
        probe = RWKV7Config()
        filtered = {k: v for k, v in cfg_json.items() if hasattr(probe, k)}
        config = RWKV7Config(**filtered)

        model = cls(config, gate_bias_init, gradient_checkpointing)

        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            weight_map = cast(
                "dict[str, str]", json.loads(index_file.read_text())["weight_map"]
            )
            shard_files = sorted(set(weight_map.values()))
        else:
            shard_files = ["model.safetensors"]

        remapped: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            for hf_key, tensor in load_file(str(model_dir / shard)).items():
                for our_key in cls.remap_hf_key(hf_key):
                    remapped[our_key] = tensor

        incompatible = model.load_state_dict(remapped, strict=False)
        missing = cast("list[str]", getattr(incompatible, "missing_keys"))  # noqa: B009
        unexpected = cast("list[str]", getattr(incompatible, "unexpected_keys"))  # noqa: B009
        # only the fusion gates may be missing; anything else is a mapping bug.
        bad_missing = [k for k in missing if "fuse_proj" not in k and "fuse_bias" not in k]
        if bad_missing or unexpected:
            msg = (
                f"HF warm-start mapping incomplete: missing={bad_missing[:8]} "
                f"unexpected={list(unexpected)[:8]}"
            )
            raise RuntimeError(msg)
        return model.to(dtype)


# ----------------------------------------------------------------------
# Iterative denoising sampler (port of state_prefill_rwkv denoise_block,
# global-sequence variant: the masked positions of the WHOLE input are
# iteratively committed by confidence).
# ----------------------------------------------------------------------
@_no_grad
def iterative_denoise(  # noqa: PLR0913
    model: BiRWKV7ForMaskedDiffusion,
    corrupted: torch.Tensor,
    masked: torch.Tensor,
    steps: int = 16,
    temperature: float = 0.0,
    self_correction: bool = False,  # noqa: FBT001, FBT002
    remask_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-denoise ``corrupted`` (mask ids at ``masked`` positions).

    Linear commit schedule: each step commits the highest-confidence masked
    positions so that ~(step/steps) of the originally-masked set is clean.
    Returns (denoised ids, commit_step [B,T]; 0 for never-masked positions).
    """
    cur = corrupted.clone()
    still = masked.clone()
    commit_step = torch.zeros_like(cur)
    total = masked.sum(dim=1)  # [B]

    for step in range(1, steps + 1):
        if not still.any():
            break
        logits = model(cur, False).float()  # noqa: FBT003
        logits[..., model.mask_token_id] = float("-inf")
        logits[..., model.pad_token_id] = float("-inf")
        probs = logits.softmax(dim=-1)
        if temperature > 0.0:
            shaped = (logits / max(temperature, 1e-6)).softmax(dim=-1)
            pred = torch.multinomial(shaped.reshape(-1, shaped.shape[-1]), 1).view_as(cur)
        else:
            pred = probs.argmax(dim=-1)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        update = still.clone()
        if self_correction and step < steps:
            cur_conf = probs.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
            reopen = (~still) & masked & (cur_conf < remask_threshold)
            update |= reopen
        conf = conf.masked_fill(~update, float("-inf"))

        if step == steps:
            commit = update
        else:
            commit = torch.zeros_like(update)
            target_clean = (total.float() * step / steps).long()
            already_clean = (masked & ~update).sum(dim=1)
            need = (target_clean - already_clean).clamp_min(0)
            for row in range(cur.shape[0]):
                k = min(int(need[row]), int(update[row].sum()))
                if k > 0:
                    _, idx = torch.topk(conf[row], k=k)
                    commit[row, idx] = True

        cur = torch.where(commit, pred, cur)
        uncommitted = commit & (commit_step == 0)
        commit_step = torch.where(uncommitted, torch.full_like(commit_step, step), commit_step)
        reopened = update & ~commit & ~still
        cur = torch.where(reopened, torch.full_like(cur, model.mask_token_id), cur)
        still = update & ~commit

    return cur, commit_step


def kendall_tau_commit_order(commit_step: torch.Tensor, masked: torch.Tensor) -> float:
    """Measure Kendall tau-b of commit order against left-to-right position.

    Computed over masked positions only (any-order-ness diagnostic; tau=1
    means strictly L2R).
    """
    taus: list[float] = []
    for row in range(commit_step.shape[0]):
        steps = commit_step[row][masked[row]].float()
        n = steps.numel()
        if n < _MIN_KENDALL_POINTS:
            continue
        pos = torch.arange(n, dtype=torch.float32, device=steps.device)
        d_steps = steps.unsqueeze(0) - steps.unsqueeze(1)
        d_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        iu = torch.triu_indices(n, n, offset=1)
        s = torch.sign(d_steps[iu[0], iu[1]]) * torch.sign(d_pos[iu[0], iu[1]])
        concordant = (s > 0).sum().item()
        discordant = (s < 0).sum().item()
        ties = (s == 0).sum().item()
        denom = math.sqrt((concordant + discordant + ties) ** 2)  # tau-a with tie damping
        if concordant + discordant > 0:
            taus.append((concordant - discordant) / denom)
    return sum(taus) / len(taus) if taus else 0.0
