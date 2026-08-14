"""Capability eval: BiRWKV diffusion + generic HF causal baseline arms.

``run_eval.py`` dispatches on ``--model_kind``:

* ``birwkv_diffusion`` — BiRWKV7ForMaskedDiffusion flat state-dict (needs ``--model_dir``).
* ``hf_causal`` — any HF ``AutoModelForCausalLM`` (the Goal-2 Qwen baseline; needs
  ``--model_dir``).
* ``relay`` — StateInjectionDiTRELAY; NOT vendored here (see the fail-closed
  ``relay_model.py`` stub).
"""

__all__ = []
