"""Stub for the RELAY (StateInjectionDiT) eval path — NOT part of this package.

The BiRWKV masked-diffusion architecture (``birwkv_diffusion`` and ``hf_causal``
model kinds) does not need the RELAY state-injection model. The ``run_eval``
harness references a few RELAY symbols so that its ``relay`` model kind can
dispatch to it when the full DiffRWKV-RELAY repo is present. In this standalone
"laces" package the RELAY architecture is intentionally not vendored; this stub
keeps the harness importable and turns any RELAY invocation into a clear error
instead of a silent import failure.
"""

from __future__ import annotations

_RELAY_ERROR = (
    "the 'relay' model_kind requires the full DiffRWKV-RELAY repo "
    "(StateInjectionDiTRELAY + relay_model.py); this laces package ships only "
    "birwkv_diffusion and hf_causal. Use --model_kind birwkv_diffusion (or "
    "hf_causal for the Qwen baseline)."
)


class LoadedRelay:  # pragma: no cover - annotation-only placeholder
    """Placeholder so ``multichoice.eval_multichoice`` type hints resolve."""


class RelayGenerator:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_RELAY_ERROR)


def load_relay(*args, **kwargs):  # pragma: no cover - stub
    raise NotImplementedError(_RELAY_ERROR)


def sample_states(*args, **kwargs):  # pragma: no cover - stub
    raise NotImplementedError(_RELAY_ERROR)


def encode_oracle_states(*args, **kwargs):  # pragma: no cover - stub
    raise NotImplementedError(_RELAY_ERROR)
