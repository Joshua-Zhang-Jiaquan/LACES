# LACES: Linear-Attention Continuous-state DiffuSion

**Continuous Diffusion as Recurrent-State Memory Planning for Language Models**

[![arXiv](https://img.shields.io/badge/technical%20report-PDF-blue)](../technical_report/main.pdf)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-orange)](https://huggingface.co/SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained)

---

LACES trains a compact diffusion planner that writes directly into the **recurrent state**
of a frozen pretrained RWKV-7 language model — controlling what it generates without touching
a single backbone weight. The diffusion model never predicts tokens; it samples memory plans,
maps them through a learned coefficient-controlled state basis, and injects the resulting
recurrent memory directly into the frozen renderer.

Trained only on OpenWebText, the same frozen 2.9B renderer equipped with LACES:

| Capability | Benchmark | Raw → LACES | Gain |
|---|---|---|---|
| Language modeling | PG19 PPL | 4.88 → 2.23 | **−54%** |
| Language modeling | WikiText-2 PPL | 16.01 → 10.43 | **−35%** |
| Instruction following | IFEval inst-strict | 0.204 → 0.301 | **+48%** |
| Function calling | BFCL V4 parse rate | 0.45 → 0.92 | **+104%** |
| Function calling | BFCL V4 accuracy | 0.17 → 0.41 | **+142%** |
| Long-passage reading | RACE | 38.7 → 58.6 | **+19.9 pts** |
| Long-context recall | BaBILong QA1 (4k ctx) | 37.0 → 48.0 | **+11.0** |

The chunked variant samples all 16 per-chunk memory plans in a single non-autoregressive pass,
enabling **14.5× parallel decoding** with identical generation quality to the sequential baseline.

> **Paper**: `technical_report/main.pdf` — *LACES: Continuous Diffusion as Recurrent-State Memory Planning for Language Models* (Jiaquan Zhang, Yunbo Long)

---

## Table of Contents

- [How it works](#how-it-works)
- [Key results](#key-results)
- [Architecture](#architecture)
- [Champion recipe](#champion-recipe)
- [Checkpoints](#checkpoints)
- [Quick start](#quick-start)
- [Repo structure](#repo-structure)
- [Evaluation](#evaluation)
- [Baselines and rejected variants](#baselines-and-rejected-variants)
- [Citation](#citation)

---

## How it works

The frozen RWKV-7 renderer is treated as a **text executor** whose memory can be externally
written. Three small trainable modules form the planner:

```
frozen recurrent hidden states
    → compact latent memory plan z ∈ R³² (S0 encoder)
    → recurrent latent denoising prior (S2: BiRWKV diffusion)
    → coefficient-controlled recurrent-state basis (S1 bridge)
    → direct state injection into frozen renderer
    → autoregressive token rendering
```

**Key insight**: the diffusion target is the renderer's internal recurrent memory, not tokens,
embeddings, or text latents. This bypasses the input-side bottleneck that all existing diffusion
language models (discrete or continuous-latent) share.

### Global vs. chunked

- **Global**: one latent plan for the whole sequence. Strong diagnostic ceiling.
- **Chunked**: the document is split into *H* chunks; a separate memory plan is learned per chunk
  and sampled jointly by a bidirectional (BiRWKV) denoiser. Memory is refreshed at each chunk
  boundary — the main mechanism for long-context behavior.

### Joint-scratch co-adaptation

The staged pipeline (train bridge on clean latents → freeze → train prior) leaves a persistent
gap: the bridge only ever sees oracle states during training but must consume sampled states at
inference. LACES trains the bridge and prior **jointly from scratch** with a co-adaptation
cross-entropy term that feeds the denoiser's own predicted latent back through the bridge.
This is the single training decision that lets the chunked variant match the global ceiling
(58.3 vs. 58.7 avg).

---

## Key results

### Language modeling perplexity (teacher-forced, 200 samples)

| Corpus | Raw RWKV-7 2.9B | LACES global | Gain |
|---|---|---|---|
| PG19 | 4.88 | **2.23** | −54% |
| LAMBADA | 45.65 | **32.70** | −28% |
| WikiText-2 | 16.01 | **10.43** | −35% |

### Downstream accuracy (generate-then-match, full 2500-question sets)

| Method | MMLU | OBQA | RACE | Avg |
|---|---|---|---|---|
| Raw RWKV-7 | 48.7 | 61.0 | 38.7 | 49.5 |
| LACES global | 46.2 | 65.8 | 45.2 | 52.4 |
| **LACES chunked** | **47.0** | **65.4** | **58.6** | **57.0** |
| Δ chunked vs. raw | −1.7 | +4.4 | **+19.9** | +7.5 |

### Central 512-token champion result

| Method | Training | MMLU | OBQA | RACE | Avg |
|---|---|---|---|---|---|
| Raw RWKV-7 | — | 53.0 | 62.0 | 44.0 | 52.7 |
| **Chunked (joint scratch, co-adapt)** | **ours** | **52.0** | 65.0 | 58.0 | **58.3** |
| *Global SFT (reference ceiling)* | *SFT* | *44.0* | *65.0* | *67.0* | *58.7* |

Joint-scratch co-adaptation is the first chunked configuration to match the global SFT ceiling.
Its MMLU (52.0) exceeds that ceiling.

### Parallel decoding speedup

At blend λ = 1, the 16 chunk states are independent — all chunks decode concurrently as one
batch, reducing autoregressive steps from 16×32 to 32:

| Mode | Throughput | Latency (512 tok) | Speedup |
|---|---|---|---|
| Sequential | 21 tok/s | 24,195 ms | 1× |
| **Parallel (λ=1)** | **307 tok/s** | **1,667 ms** | **14.5×** |

Generation quality is identical at both blend settings (55.93 avg).

---

## Architecture

### S0: Latent encoder
Text → pooled hidden states → VAE encoder → `z ∈ R³²` (or `Z ∈ R^[H,32]` for chunked).
Trains with reconstruction + KL to build a Gaussian-compatible latent space.

### S1: State bridge
Learns layer-wise **coefficient-controlled recurrent-state bases** `B_l^(k)` (K=16 for 512-tok,
K=32 for 4096-tok). A per-layer linear head maps `z` → coefficients `α_l(z) ∈ R^K`, composing:
```
Ŝ_l = γ_l Σ_k α_l^(k)(z) · B_l^(k)
```
The learned scale γ_l starts near zero — injected states are negligible at training start
and grow only where they help next-token prediction. The champion bridge is a **linear
independent** projection reused from the global model, frozen during S2 training.

### S2: Diffusion prior
Bi-directional RWKV (BiRWKV) denoiser operating on the latent sequence `Z_{1:H}`.
The bidirectional scan coordinates all H chunk latents jointly. **Boundary-token conditioning**
(prepend/append prefix summary → BiRWKV forward+backward scan → slice) is the only
conditioning mechanism that makes classifier-free guidance effective in state space.

### Injection
Before chunk h, the planned state `Ŝ_{:,h}` is blended into the running cache:
```
S̃ = (1−λ)·S_real + λ·Ŝ
```
The champion uses λ = 0.7; at λ = 1 (hard write), generation quality is identical, enabling
the 14.5× parallel decoding.

---

## Champion recipe

512-token trajectory, 2.9B renderer, joint-scratch co-adaptation + boundary conditioning:

```yaml
trajectory_s1_mode:         independent    # linear single-z bridge S1
trajectory_denoiser_type:   birwkv         # BiRWKV S2 denoiser
use_cond_boundary:          true           # boundary token + bidirectional scan
s2_unfreeze_s1:             true           # joint co-adapt
s2_coadapt_ce_loss_weight:  0.5            # CE on z_0_pred flows into S1
gen_type:                   ddpm
trajectory_state_blend:     0.7
cfg_drop_prob:              0.1
denoiser_hidden:            768
denoiser_depth:             8
n_basis:                    16
chunk_size × horizon:       32 × 16 = 512
```

**Score** (generate-then-match, CFG=3, blend=0.7, 100/task): avg **57.67–58.33**
(MMLU 50 / OBQA 65 / RACE 58).

---

## Checkpoints

| Line | Variant | Checkpoint | Score |
|---|---|---|---|
| trajectory 2.9B | **champion joint-scratch** | `traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000` | **57.67** |
| trajectory 2.9B | condboundary bare | `traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000` | ~53 |
| trajectory 2.9B | condboundary + SFT-maximal | `...-s2-birwkv-ddpm-condboundary-sft-maximal/step_00050000` | ~56 |
| trajectory 2.9B | condboundary + SFT-stem | `...-s2-birwkv-ddpm-condboundary-sft-stem/step_00050000` | ~56 |
| single-z 2.9B | SFT-maximal (global ref) | `test-v6-2.9B-s2-prefix-suffix-cfg` @150k | 58.67 |
| trajectory 4096 | co-adapt (H200) | `C-4096-coadapt` (traj64x64) | in progress |

All checkpoints under `outputs_relay/` in the parent repo.
HuggingFace release: [`SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained`](https://huggingface.co/SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained) (champion, 57.67).

---

## Quick start

### Migrate an old checkpoint (dit → denoiser keys)

Old checkpoints use `trajectory_dit.*` / `latent_dit.*` weight keys. Migrate first:

```bash
python projects/LACES/migrate_checkpoint.py \
  --in  outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000/model.pt \
  --out /tmp/champion_migrated.pt
# renames 570 keys → *_denoiser.*, 0 unexpected keys (verified)
```

### Train (champion recipe, 2.9B, single GPU)

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

**Supported backbones**: 0.4B / 1.5B / 2.9B / 13.3B. **Context lengths**:
- **512**: `traj32x16` (chunk 32 × horizon 16, n_basis 16) — data `owt_rwkv_tokens/train`
- **4096**: `traj64x64` (chunk 64 × horizon 64, n_basis 32) — data `fineweb_4096_packed_full`

The 4096 champion just swaps the config and keeps all other flags identical.

### Eval (generate-then-match)

```bash
python scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py \
  --ckpt_dir <migrated-ckpt-dir> \
  --task_data_dir baseline/Cola-DLM/generate_task_data \
  --tasks mmlu --trajectory_s1_mode independent \
  --trajectory_state_blend 0.7 --cfg_scale 3 \
  --max_samples 100 --steps 100 --max_new_tokens 32 --temperature 0.0
```

---

## Repo structure

| File | Role |
|---|---|
| `model.py` | `StateHijackingRELAY` wrapper + `LatentDenoiser` / `TrajectoryLatentRWKV` (BiRWKV) / `TrajectoryLatentTransformer` (baseline) / encoders / helpers |
| `train.py` | Training + sampling entrypoint. Imports local `model`, reuses repo `data_simple` / `utils` |
| `albatross_rwkv7.py` | Native Albatross RWKV-7 block (renderer backbone) |
| `sphere_flow.py` | Spherical geodesic flow matching (experimental) |
| `data_simple.py` | Dataset loader (copied in — no repo-root dependency) |
| `utils.py` | LR schedule, dtype helpers |
| `configs/` | 32 configs covering all backbones, context lengths, and mode variants |
| `migrate_checkpoint.py` | Remap old `*_dit.*` → `*_denoiser.*` checkpoint keys |
| `eval/` | Self-contained eval chain: samplers, runner, scorer, task data (MMLU/OBQA/RACE/BaBILong) |

**Self-containment**: all imports are local. External dependencies by reference only:
- Backbone weights: `models/RWKV7-Goose-World3-{0.4B,2.9B,13.3B}-HF`
- Training data: `preprocessed_data/...`
- Trained checkpoints: `outputs_relay/...`

---

## Baselines and rejected variants

**Effective** (used in the champion line):
- `condboundary` — boundary-token conditioning via bidirectional scan
- `maximal SFT` / `STEM SFT` — supervised finetuning on the S2 prior only

**Rejected** (kept as baselines only):
- `condadaln` — adaLN-Zero global modulation; collapses during training
- Decoded-logit KL alignment — post-hoc loss that degrades sampled quality
- Old `TrajectoryStateDecoder` (RWKV/BiRWKV rollout S1) — 14 pts weaker than linear bridge
- DiT (Transformer) denoiser — functional but no conditioning advantage
- SIM-CoT step supervision — worsens latent diversity (23% vs. champion 77% effective rank)

See the [technical report](../technical_report/main.pdf) Appendix for full ablation tables.

---

## Citation

```bibtex
@techreport{zhang2026laces,
  title        = {{LACES}: Continuous Diffusion as Recurrent-State Memory Planning for Language Models},
  author       = {Jiaquan Zhang and Yunbo Long},
  year         = {2026},
  note         = {Technical report}
}
```

## Rename map (old → new)

| Old (repo) | New (here) |
|---|---|
| module `models/state_hijacking_dit.py` | `model.py` |
| class `StateInjectionDiTRELAY` | `StateHijackingRELAY` |
| class `LatentDiT` | `LatentDenoiser` |
| class `TrajectoryLatentDiT` | `TrajectoryLatentTransformer` |
| class `DiTBlock` | `DenoiserBlock` |
| attr `self.latent_dit` | `self.latent_denoiser` |
| attr `self.trajectory_dit` | `self.trajectory_denoiser` |
| kwargs `dit_hidden/dit_depth/...` | `denoiser_*` (old keys still accepted) |
| config `trajectory_denoiser_type: dit` | `transformer` (`dit` still accepted) |
