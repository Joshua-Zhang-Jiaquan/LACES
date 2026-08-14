"""Convert a BiRWKV flat-state-dict checkpoint (model.pt) to bf16 safetensors.

The trainer writes a flat fp32 state dict (with the FSDP FULL_STATE_DICT format).
This halves the release size (~16.4GB -> ~8.2GB) and makes the checkpoint
framework-agnostic (no torch.pickle). Usage:

    python scripts/convert_ckpt_to_safetensors.py \
        --ckpt /path/to/step_00004000/model.pt \
        --out /path/to/staging/model.safetensors

Loading back is `model.load_state_dict(safetensors.torch.load_file(path), strict=False)`.
"""
from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        # FSDP FULL_STATE_DICT checkpoints are a flat dict already; guard anyway.
        raise TypeError(f"expected a state dict, got {type(state)}")
    bf16 = {k: v.to(torch.bfloat16) for k, v in state.items()}
    save_file(bf16, args.out)
    n = len(bf16)
    gb = sum(v.numel() * v.element_size() for v in bf16.values()) / 1e9
    print(f"wrote {args.out}: {n} tensors, {gb:.2f} GB (bf16)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
