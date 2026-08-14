# DiffRWKV 2.0 — Goals Reference (what this repo implements)

This is a condensed pointer to the source plan
`DAN/DiffRWKV_2_0_完整新版计划.md` (§0, §4.6) that LACES-BiRWKV-DLM implements.

## §0 — The pivot

The 2.0 plan retires this as the *main* probability path:

```
continuous latent diffusion → recurrent-state bridge → frozen RWKV autoregressive renderer
```

and replaces it with:

```
absorbing-mask token diffusion
  → bidirectional BGDA-RWKV denoiser on the token canvas
  → Joint-Commit RWKV for a committed token group's joint probability
  → commit / token-to-mask revision
```

## §0 — Nine final-model requirements (with this repo's status)

1. diffusion acts directly on tokens → ✅
2. condition on arbitrary visible token subset → ✅
3. any reveal order defines a normalized AR factorization → ✅ (τ≈0)
4. exact mode commits one token at a time → ✅
5. fast mode = Joint-Commit group model (no per-position-marginal product) → ⏳ unbuilt (计划二)
6. canonical denoiser output independent of canvas history → ✅ by construction
7. no frozen left-to-right renderer → ✅
8. linear complexity per denoise step → ✅ (dual RWKV-7 scan)
9. honest memory-complexity claims → ✅

## §4.6 — Acceptance gates (计划二 Joint-Consistent BGDA)

The §4.6 gates are phrased for the *Joint-Commit* stage (not yet built); the §4.6-*style*
diffusion gates that **have** been measured on this architecture are:

- mode-mixing / joint-consistency error — not measured (needs Joint-Commit)
- **iterative > single-step** — ✅ (disjoint CIs)
- **bidirectional necessity** — ✅ (−0.97 nats, 0.4B pilot)
- **any-order τ** — ✅
- **noise conditioning** — ✅ (bucket-ordered CE)
- vector-erase / local-attention kernel gates — n/a (kernel plan, not this repo)

See `docs/tech_report.md` §4 for the full numbers.
