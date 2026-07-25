# LACES

Self-contained copy of the **champion** State-Hijacking RELAY line: frozen RWKV-7 + a
compact latent diffusion policy that injects directly into the WKV recurrent state. Includes
the joint-scratch co-adapt champion and its variants (condboundary, SFT), across single-z and
trajectory, at 0.4B / 1.5B / 2.9B / 13.3B.

> **Naming note.** This folder has **no `dit` in any Python name**. The old line was in
> `models/state_hijacking_dit.py` with classes/attributes named `...DiT...` /
> `trajectory_dit` / `latent_dit`. That was a historical misnomer — the champion denoiser is
> **BiRWKV, not a Transformer DiT**. Here everything is renamed to `denoiser`. Old checkpoints
> (whose weight keys are `trajectory_dit.*` / `latent_dit.*`) are made loadable via
> `migrate_checkpoint.py`. The original repo files are untouched.

## What the method is

Frozen RWKV-7 backbone is used as a text executor; only small policy modules are trained:

- **S0 encoder** — text → latent `z` (single-z: one `z ∈ R^32`; trajectory: `Z ∈ R^[H,32]`).
- **S1 bridge** — `z → RWKV recurrent state` (the champion uses the **independent linear**
  single-z projection, frozen).
- **S2 denoiser** — a diffusion model over the latent, conditioned on the prefix. The champion
  denoiser is **BiRWKV** (bidirectional RWKV scan), with **condboundary** conditioning
  (boundary token + bidirectional scan). A Transformer denoiser (`trajectory_denoiser_type=transformer`,
  formerly `dit`) exists as a **baseline** only.

## Champion recipe (joint-scratch co-adapt + condboundary)

512-token trajectory, 2.9B. From-scratch **joint** training of S1+S2 with co-adapt CE fed the
sampled Z, plus condboundary conditioning:

```
trajectory_s1_mode   = independent   (linear single-z bridge S1)
trajectory_denoiser_type = birwkv    (BiRWKV S2 denoiser)
use_cond_boundary    = true          (boundary token + BiRWKV bidirectional scan)
s2_unfreeze_s1       = true          (joint co-adapt)
s2_coadapt_ce_loss_weight = 0.5      (CE on z_0_pred flows back into S1)
gen_type             = ddpm
trajectory_state_blend = 0.7,  cfg_drop_prob = 0.1
denoiser_hidden = 768, denoiser_depth = 8, n_basis = 16, traj 16x32 = 512
```

Real-sampling score (generate-then-match, cfg=3, blend=0.7, 100/task): **avg ~57.67–58.33**
(MMLU 50 / OBQA 65 / RACE 58), which first matched the single-z SFT ceiling (58.67).

## Champion + variants (checkpoints, all in root `outputs_relay/`)

| Line | Variant | Checkpoint | Score |
|---|---|---|---|
| trajectory 2.9B | **champion joint-scratch** | `traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000` | **57.67** |
| trajectory 2.9B | condboundary bare | `traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000` | ~53 |
| trajectory 2.9B | condboundary + SFT-maximal | `...-s2-birwkv-ddpm-condboundary-sft-maximal/step_00050000` | ~56 |
| trajectory 2.9B | condboundary + SFT-stem | `...-s2-birwkv-ddpm-condboundary-sft-stem/step_00050000` | ~56 |
| single-z 2.9B | SFT-maximal (global ref) | `test-v6-2.9B-s2-prefix-suffix-cfg` @150k | 58.67 |
| trajectory 4096 | co-adapt (H200) | `C-4096-coadapt` (traj64x64) | in progress |

Variants that were tried and **rejected** (kept only as baselines): `condadaln` (adaLN-Zero
global modulation — collapses), decoded-logit KL alignment, the old Transformer/RWKV
`TrajectoryStateDecoder` S1, and the DiT (Transformer) denoiser. See root `README.md` §C for
the full ablation history.

HuggingFace: `SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained` (champion, 57.67).

## Files

| File | Role |
|---|---|
| `model.py` | `StateHijackingRELAY` wrapper + `LatentDenoiser` / `TrajectoryLatentRWKV` (BiRWKV) / `TrajectoryLatentTransformer` (baseline) / encoders / helpers. No `dit` names. |
| `train.py` | Training + sampling entrypoint (was `train_state_hijacking_dit.py`). Imports local `model`, reuses root `data_simple` / `utils`. |
| `albatross_rwkv7.py`, `sphere_flow.py` | Model dependencies (native Albatross RWKV7 block; spherical flow). |
| `data_simple.py`, `utils.py` | Dataset loader + lr/dtype helpers (copied in — no repo-root dependency). |
| `configs/` | 32 champion-line configs, 0.4B/1.5B/2.9B/13.3B, single-z + prefix/suffix + trajectory, **both 512 (traj32x16) and 4096 (traj64x64) context lengths**. `denoiser_*` keys (old `dit_*` keys still accepted). |
| `migrate_checkpoint.py` | Remap old `trajectory_dit.*`/`latent_dit.*` checkpoint keys → `*_denoiser.*`. |
| `eval/` | Self-contained eval chain: `run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py` (+`_cfg.py`), samplers (`sample_prefix_suffix_cfg.py`, `sample_prefix_suffix_trajectory_cfg.py`, `sample_trajectory_ladire.py`), `relay_utils.py` (loads + migrates dit→denoiser keys), scorer `acc_calc.py`, and `task_data/*.jsonl` (mmlu/obqa/race/babilong). All imports are local. |

## Self-containment

The folder is **code-self-contained**: `model.py`, `train.py`, samplers, `eval/`, `data_simple.py`,
`utils.py` all import only from within this folder (verified: no `models.*` / `scripts.*` repo
imports remain). `train.py` puts the project dir first on `sys.path`, so the local copies win.

Only these are external **by size** (never copied into a code folder — reference by path):

- backbone weights: `models/RWKV7-Goose-World3-{0.4B,2.9B,13.3B}-HF` (set in each config's `rwkv_local_path`)
- training data: `preprocessed_data/...` (config `data.token_dir`)
- trained checkpoints: `outputs_relay/...` (loaded via `--ckpt_dir` / `training.resume`)

### Rename map (old → new)

| Old (repo) | New (here) |
|---|---|
| module `models/state_hijacking_dit.py` | `model.py` |
| class `StateInjectionDiTRELAY` | `StateHijackingRELAY` |
| class `LatentDiT` | `LatentDenoiser` |
| class `TrajectoryLatentDiT` | `TrajectoryLatentTransformer` |
| class `DiTBlock` | `DenoiserBlock` |
| attr `self.latent_dit` | `self.latent_denoiser` |
| attr `self.trajectory_dit` | `self.trajectory_denoiser` |
| kwargs/config `dit_hidden/dit_depth/dit_num_heads/dit_num_tokens` | `denoiser_*` (old keys still read) |
| config value `trajectory_denoiser_type: dit` | `transformer` (`dit` still accepted) |

Data input / preprocessing is unchanged and reuses the repo root (`data_simple.py`,
`preprocessed_data/...`).

## Load an existing (old) champion checkpoint

Old checkpoints store weights under `trajectory_dit.*` / `latent_dit.*`. Migrate first:

```bash
python projects/LACES/migrate_checkpoint.py \
  --in  outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000/model.pt \
  --out /tmp/champion_migrated.pt
# renames 570 keys -> *_denoiser.*, then StateHijackingRELAY.load_state_dict loads it clean.
```

The migrated checkpoint loads into the renamed model with **0 unexpected keys** (verified).

## Train (champion recipe, 2.9B, single GPU)

```bash
python projects/LACES/train.py \
  --config-name rwkv_relay_2.9B_state_relay_vae32_traj32x16 \
  training.stage=2 training.gen_type=ddpm \
  model.n_basis=16 model.trajectory_chunk_size=32 model.trajectory_horizon=16 \
  model.trajectory_s1_mode=independent model.trajectory_denoiser_type=birwkv \
  +model.use_cond_boundary=true +training.s2_unfreeze_s1=true \
  +loss.s2_coadapt_ce_loss_weight=0.5 +training.cfg_drop_prob=0.1 \
  model.denoiser_hidden=768 model.denoiser_depth=8 model.trajectory_state_blend=0.7 \
  data.token_dir=preprocessed_data/owt_rwkv_tokens/train data.max_length=512 \
  training.train_batch_size=6 training.save_every_n_steps=2000 training.num_train_steps=16000 \
  logging.run_name=laces-2.9B-champion
```

Configs cover 0.4B / 1.5B / 2.9B / 13.3B; single-z (`..._vae32`), single-z prefix/suffix
(`..._prefix_suffix_s1/s2`), and trajectory. Both context lengths are included:

- **512**: `..._traj32x16` (chunk 32 × horizon 16, n_basis 16), single-z `..._vae32` — data `owt_rwkv_tokens/train`.
- **4096**: `..._traj64x64` (chunk 64 × horizon 64, n_basis 32) — data `fineweb_4096_packed_full`; single-z `..._vae32_4096` / `..._prefix_suffix_s1/s2_4096` — data `fineweb_4096_full`.

The 4096 champion just swaps the trajectory config for `..._traj64x64` and keeps every other
flag identical to the 512 recipe.

## Eval (champion, generate-then-match)

Uses the root eval runner (data/preprocessing unchanged):

```bash
python scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py \
  --ckpt_dir <migrated-ckpt-dir> \
  --task_data_dir baseline/Cola-DLM/generate_task_data \
  --tasks mmlu --trajectory_s1_mode independent \
  --trajectory_state_blend 0.7 --cfg_scale 3 \
  --max_samples 100 --steps 100 --max_new_tokens 32 --temperature 0.0
```
