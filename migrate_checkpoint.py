from __future__ import annotations

# Migrate an old State-Hijacking RELAY checkpoint (attribute names contained "dit")
# to the renamed LACES layout (dit -> denoiser).
#
# Old checkpoints saved weight keys under the wrapper attribute names:
#     latent_dit.*        (single-z latent denoiser)
#     trajectory_dit.*     (trajectory denoiser; for the champion this holds the BiRWKV weights)
# The renamed model uses:
#     latent_denoiser.*
#     trajectory_denoiser.*
# All other prefixes (state_basis, state_scale, encoder_trunk, mu_head, logvar_head,
# aux_decoder, alpha_heads) are unchanged.
#
# The checkpoint is a dict: {trainable_state: {name: tensor}, state_basis, state_scale, step, config}.
# Only keys in `trainable_state` need remapping; `config` is a yaml string (may still say
# trajectory_denoiser_type: birwkv etc., which the new model reads unchanged).
#
# Usage:
#   python migrate_checkpoint.py --in  outputs_relay/<old>/step_00026000/model.pt \
#                                --out outputs_relay/<old>/step_00026000/model_migrated.pt
#   # or in place (overwrites): add --inplace

import argparse
from pathlib import Path

import torch

KEY_PREFIX_RENAMES = (
    ("trajectory_dit.", "trajectory_denoiser."),
    ("latent_dit.", "latent_denoiser."),
)


def remap_key(key: str) -> str:
    for old, new in KEY_PREFIX_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def migrate_state(trainable_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    migrated: dict[str, torch.Tensor] = {}
    n_renamed = 0
    for key, value in trainable_state.items():
        new_key = remap_key(key)
        if new_key != key:
            n_renamed += 1
        migrated[new_key] = value
    return migrated, n_renamed


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate old dit-keyed RELAY checkpoint to LACES denoiser keys")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to old model.pt")
    parser.add_argument("--out", dest="out_path", default=None, help="Output path (default: <in>.migrated.pt)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file in place")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    obj = torch.load(in_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "trainable_state" not in obj:
        raise ValueError("Checkpoint is not the expected {trainable_state, ...} dict format")

    ts = obj["trainable_state"]
    already = any(k.startswith(("trajectory_denoiser.", "latent_denoiser.")) for k in ts)
    has_old = any(k.startswith(("trajectory_dit.", "latent_dit.")) for k in ts)
    if already and not has_old:
        print("Checkpoint already uses denoiser.* keys; nothing to migrate.")
        return

    migrated, n_renamed = migrate_state(ts)
    obj["trainable_state"] = migrated

    if args.inplace:
        out_path = in_path
    elif args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = in_path.with_suffix(".migrated.pt")

    torch.save(obj, out_path)
    print(f"Renamed {n_renamed} keys (trajectory_dit.*/latent_dit.* -> *_denoiser.*)")
    print(f"Wrote migrated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
