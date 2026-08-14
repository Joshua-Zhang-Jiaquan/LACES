# LACES-BiRWKV-DLM — A Bidirectional RWKV-7 Token Masked-Diffusion Language Model

**Jiaquan Zhang (SII-Jiaquan)** · 2026-08-14 · status: working release, Goal-2 evals in flight

---

## 1. Motivation & Goals

This project is the "authentic token-level masked-diffusion LM" arm of the DiffRWKV 2.0
program. The 2.0 plan (`DAN/DiffRWKV_2_0_完整新版计划.md` §0) explicitly retired the
prior LACES design — *continuous latent diffusion → recurrent-state bridge → frozen
causal RWKV autoregressive renderer* — as the **main probability path**, because that
design diffuses a latent/state that is then read out **causally**, so the diffusion is
structurally unable to produce parallel or any-order token emission. The 2.0 plan
replaces it with:

```
absorbing-mask token diffusion
  → bidirectional BGDA-RWKV denoiser operating directly on the token canvas
  → Joint-Commit RWKV modeling the joint probability of a committed token group
  → commit / token-to-mask revision
```

The final model must satisfy nine requirements (§0). The current release implements and
**measures** requirements 1–4 and 6–9:

| # | Requirement | Status |
|---|---|---|
| 1 | diffusion acts directly on tokens (not latent/state) | ✅ by construction |
| 2 | condition on an arbitrary visible token subset | ✅ masked-canvas forward |
| 3 | any reveal order defines a normalized AR factorization | ✅ Kendall τ ≈ 0 measured |
| 4 | exact mode commits one token at a time | ✅ sampler supports |
| 5 | fast mode = Joint-Commit group model (not independent marginals) | ⏳ **not built** (计划二 §4.x) |
| 6 | canonical denoiser output independent of canvas history | ✅ by construction |
| 7 | no frozen left-to-right renderer | ✅ own `lm_head`, no renderer |
| 8 | linear complexity per denoise step | ✅ dual RWKV-7 scan, O(L) |
| 9 | honest memory-complexity claims | ✅ |

Two campaign goals frame the evidence:

- **Goal 1 — authentic diffusion LM (ACHIEVED):** all diffusion-gate properties measured
  with bootstrap confidence intervals.
- **Goal 2 — beat Qwen3.5-4B on code (IN PROGRESS):** infill is strong, but free
  program *synthesis* was near-zero; a free-generation corruption lane is under
  evaluation.

---

## 2. Architecture

`models/birwkv7_diffusion.py` — **BiRWKV7ForMaskedDiffusion** (~3.9–4.1B params at the
2.9B-base geometry; the forward RWKV-7 backbone is 2.948B, the cloned reverse attention
branch adds ~934M; FFN / embeddings / head are shared, not duplicated).

```
embeddings → N × BiRWKV7Block → final norm → lm_head → [B, T, vocab] logits
```

Each `BiRWKV7Block` holds **two complete `fla` RWKV7Attention modules** — a forward scan
and a reverse scan (on the flipped sequence) — plus the pretrained `RWKV7FeedForward` and
norms. The two streams fuse through a learned sigmoid gate:

```
alpha = σ(fuse_proj(x) + fuse_bias)          # fuse_proj zero-init, fuse_bias init +4.0
o = alpha · o_fwd + (1 − alpha) · o_bwd
```

The `+4.0` bias makes `alpha ≈ 0.982` at init, so the untrained network is numerically
≈ the pretrained causal RWKV-7. This is what makes the HF warm-start **parity test**
meaningful: with `force_forward=True` (or the reverse branch weight-cloned at warm-start)
the BiRWKV must reproduce the frozen causal model exactly. Warm-start clones the forward
attention weights into the reverse copy (`remap_hf_key` fans one `model.layers.{i}.attn.*`
key out to both directions), and `from_hf_pretrained` **fails unless the only missing keys
are `fuse_proj`/`fuse_bias`** — a strict parity gate.

- `MASK_TOKEN_ID = 65535` (unused slot in the RWKV World vocab; `EOS = 65530`),
  `PAD_TOKEN_ID = 0`.
- **No frozen renderer, no post-diffusion causal generator.** The denoiser's own `lm_head`
  produces clean-token logits directly from corrupted input.

### Sampling — `iterative_denoise`

A global-sequence confidence-commit sampler: corrupted tokens go in, and at each of `steps`
rounds a linear fraction (`≈ step/steps`) of the still-masked positions is committed to the
argmax (or temperature-multinomial) token; committed tokens are never re-masked unless
`self_correction=True`. `kendall_tau_commit_order` measures any-order-ness of the commit
order vs. left-to-right.

---

## 3. Training Recipe

Masked-diffusion objective: per-block noise buckets corrupt a fraction of each block with
`[MASK]`, and the model is trained with a token-normalized masked cross-entropy, plus a
small causal-replay auxiliary loss (`λ_c ≈ 0.1`) to retain the warm-started language
knowledge. Noise level is **inferred from the mask count** (MD4-style: no explicit `t`).

Three stages (all on 4×8 H200, FSDP FULL_SHARD):

1. **0.4B pilot (2000 steps / 2.1B tokens, fineweb).** The bidirectional-necessity
   ablation: identical seed/data/objective for a `bidir` arm vs a `--force-forward`
   control arm.
2. **2.9B conversion (4000 steps / 4.2B tokens, fineweb + capability blend).** Staged
   unfreeze (`--stage-a-frac 0.05 --stage-b-frac 0.15`). Two trainer bugs were found and
   fixed: `RWKV7Config` must be constructed via kwargs (setattr-after-init leaves
   `num_heads`/`value_dim` stale → GroupNorm divisibility crash at hidden 2560), and
   staged-unfreeze grads must be **zeroed, not None'd** (AdamW lazy state +
   `FSDP.optim_state_dict` at the first checkpoint requires state for every param).
3. **code-CPT (4000 steps / 4.2B tokens, code_mixture + fineweb), then gen-lane**
   (`--gen-prob 0.3`, warm-start from code-CPT, `--lr 2e-5 --warmup-steps 200`). The
   gen-lane corruption gives 30% of rows a **clean prefix (uniform 5–60% of the sequence)
   followed by an all-masked contiguous canvas to the end** — the corruption shape of
   prompt→completion generation, which block-local bucket masking never produces. This is
   the designed fix for the Stage-C finding "strong infill, no synthesis."

---

## 4. Results

### Goal 1 — diffusion gates (codecpt-2p9b / step 4000, unless noted)

| Gate | Result | Verdict |
|---|---|---|
| Bidirectional necessity | 0.4B pilot: bidir `val_mask_ce 4.861` vs fwdonly `5.828` (**−0.97 nats**), widening monotonically to step 2000 | ✅ PASS |
| Iterative beats single-step | sampler grid (512 seqs, sc OFF): r30 0.739→**0.764**@16, r50 0.607→**0.646**, r70 0.399→**0.447**@32 — disjoint bootstrap CIs | ✅ PASS |
| Any-order decoding | Kendall τ ≈ −0.05 … 0.07 (task-adaptive, far from the L→R order) | ✅ PASS |
| Direct token provenance | own `lm_head`, no frozen renderer (by construction) | ✅ PASS |
| Noise conditioning | bucket-ordered val CE low 1.10 / med 1.81 / high 3.49 | ✅ PASS |
| Warm-start parity | strict key-mismatch gate on `from_hf_pretrained` | ✅ PASS |

**Self-correction remasking is destructive at eval** (`remask_threshold=0.25`): it reopens
most correct commits because true-token probabilities sit below 0.25 at this checkpoint's
entropy, collapsing exact-match by 4–6× with steps. Eval generators therefore default
`self_correction=False`.

### Final checkpoint (this release) — `codecpt2-genlane / step_00004000`

- `val_mask_ce = 2.5725` (buckets: low 1.0975 / med 1.8136 / high 3.4880)
- sampler: `em@1 = 0.5878`, `em@8 = 0.6251`, `em@16 = 0.6386`, `em@32 = 0.6305`,
  `τ@16 = 0.0672`, `mask_residue = 0.0`

### Goal 2 — code synthesis (the open problem)

- **Infill strong:** reconstruction exact-match ≈ 0.76 at 30% mask (Stage C).
- **Synthesis was ~0:** HumanEval pass@1 = 0.0, MBPP iter32 = 0.0023 (1/427) on
  `codecpt-2p9b`. Failure taxonomy: 98% `syntax_error`. Completion dumps show the model
  filling the masked canvas with degenerate chat-marker tokens (`</istant>…`) — the
  training corpora are SFT/agent-formatted with plain-text `Assistant:` markers, so
  bare-code prompts are out-of-distribution. A chat-frame adapter (wrap prompt in
  `User: … Assistant:` + open a ```python fence, close-at-fence extraction) was added, but
  the **free all-masked canvas was simply untrained** — the model learned reconstruction,
  not synthesis.
- The gen-lane retrain (`--gen-prob 0.3`) targeted exactly this gap, but **HumanEval3
  merged to iter16=0.0 / iter32=0.0 (n=164) and MBPP3 merged to iter16=0.0 / iter32=0.0
  (n=427)** — an unchanged (or slightly worse) failure taxonomy (98–100% `syntax_error`).
  So 4000 steps at `--gen-prob 0.3` did **not** move free synthesis off zero: the
  corruption shape was not the binding constraint at this scale.

---

## 5. In-Flight & Next

- **Round-3 eval result (2026-08-14):** HumanEval3 = **0.0** and MBPP3 = **0.0** on the
  gen-lane checkpoint — the `--gen-prob 0.3` lane did not move synthesis off zero (see §4).
- **Qwen3.5-4B baseline (measured 2026-08-14): HumanEval pass@1 = 0.616** (CI95
  0.537–0.689, n=164) and **MBPP pass@1 = 0.759** (CI95 0.717–0.801, n=427). This is the
  Goal-2 bar; the BiRWKV arm is at 0.0/0.0, a **qualitative** gap (syntax_error vs
  valid-code).
- **Joint-Commit fast mode** (requirement 5): a small group-model for joint conditional
  probability of a committed group, rather than per-position marginals — 计划二 §4.x,
  the main unbuilt piece.
- **Remask/revision sampler** (ReMDM/RemeDi) with a *recalibrated* threshold, to restore
  iterative-refinement without the destructive 0.25 collapse.
- **Open synthesis question:** the sampler degenerates specifically on the *all-masked
  canvas* while infill is strong — next-round options include a dedicated synthesis-only
  phase (higher gen-prob / longer masked tail), a bare-code data lane (strip chat frames),
  or revisiting the sampler (commit schedule / temperature) rather than the corruption.

---

## 6. How to Use

```python
from models.birwkv7_diffusion import (
    BiRWKV7ForMaskedDiffusion, iterative_denoise, MASK_TOKEN_ID,
)

model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
    "RWKV/RWKV7-Goose-World3-2.9B-HF", dtype=torch.bfloat16
)
model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=False)

ids = torch.tensor([[MASK_TOKEN_ID] * 128])      # all-masked canvas
denoised, commit_step = iterative_denoise(model, ids, torch.ones_like(ids, dtype=torch.bool),
                                          steps=16, temperature=0.0, self_correction=False)
```

Training: `bash qz/launch_birwkv_diffusion.sh` with `MODE=train`, or invoke
`train/train_birwkv_diffusion.py` directly. Full checkpoint: see the Hugging Face model
card (`SII-Jiaquan/LACES-BiRWKV-DLM-2.9B-genlane`).

---

## 7. Limitations (honest)

- Free-generation code synthesis is **not yet demonstrated**; the in-flight eval may
  resolve this, or document a negative result that drives the next corruption curriculum.
- The reverse attention branch is a weight-cloned *parallel* stream, not an independently
  pretrained bidirectional mixer — bidirectional benefit is measured, not architecturally
  "for free".
- Requires the `fla` (flash-linear-attention) CUDA kernels; the model cannot import on a
  CPU-only host.
- The Joint-Commit fast mode and remasking revision are unbuilt.
