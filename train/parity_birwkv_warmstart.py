"""Warm-start parity test: BiRWKV7ForMaskedDiffusion(force_forward) vs HF causal RWKV-7.

Loads the same HF checkpoint into (a) the reference fla RWKV7ForCausalLM and
(b) our BiRWKV7ForMaskedDiffusion, runs a fixed token batch through both with
the reverse stream disabled, and requires logits to agree within bf16
tolerance. Passing this proves the HF->BiRWKV key remap is exact BEFORE any
training spend (plan Step 1.2). Requires CUDA (fla kernels).

Usage:
  python scale/train/parity_birwkv_warmstart.py --model-dir <hf-dir> [--seq-len 512]
Exit 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCALE_DIR = Path(__file__).resolve().parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

_MIN_ARGMAX_MATCH = 0.999


def main() -> int:
    """Run the warm-start parity check; exit 0 on PASS, 1 on FAIL."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--max-abs-tol", type=float, default=0.15,
                   help="bf16 logit tolerance (max |delta|)")
    p.add_argument("--mean-abs-tol", type=float, default=0.02)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    device = torch.device("cuda")
    torch.manual_seed(7)
    # avoid ids >= 65530 (EOS/reserved); realistic mid-range vocab draw
    ids = torch.randint(100, 60000, (args.batch, args.seq_len), device=device)

    print(f"[parity] loading HF causal reference from {args.model_dir}")  # noqa: T201
    ref = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    with torch.no_grad():
        ref_logits = ref(input_ids=ids).logits.float()
    del ref
    torch.cuda.empty_cache()

    print("[parity] loading BiRWKV7ForMaskedDiffusion (same weights)")  # noqa: T201
    ours = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
        args.model_dir, dtype=torch.bfloat16
    ).to(device).eval()
    with torch.no_grad():
        our_logits = ours(ids, force_forward=True).float()

    delta = (ref_logits - our_logits).abs()
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    ref_pred = ref_logits.argmax(-1)
    our_pred = our_logits.argmax(-1)
    argmax_match = float((ref_pred == our_pred).float().mean())

    print(f"[parity] max|delta|={max_abs:.5f} mean|delta|={mean_abs:.6f} "  # noqa: T201
          f"argmax_match={argmax_match:.4f}")

    ok = (
        max_abs <= args.max_abs_tol
        and mean_abs <= args.mean_abs_tol
        and argmax_match >= _MIN_ARGMAX_MATCH
    )
    print(f"[parity] {'PASS' if ok else 'FAIL'}")  # noqa: T201
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
