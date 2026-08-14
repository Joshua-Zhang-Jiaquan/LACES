# LACES-BiRWKV-DLM

A bidirectional RWKV-7 **token-level masked-diffusion language model**. The BiRWKV is the
denoiser: corrupted tokens go in, clean-token logits come out of the model's own `lm_head`.
No frozen renderer, no post-diffusion causal generator — diffusion acts directly on tokens.

This is the "authentic token masked-diffusion LM" arm of the DiffRWKV 2.0 program, which
retired the prior latent/state-diffusion LACES design as the main probability path.

## Results at a glance

| Metric | Value |
|---|---|
| Params (2.9B-base geometry) | ~3.9–4.1B (2.948B forward + 934M cloned reverse) |
| Final val mask CE (`codecpt2-genlane`) | **2.573** (low 1.10 / med 1.81 / high 3.49) |
| Sampler exact-match (gen-lane) | em@1 0.588 → em@16 **0.639** (iterative > single) |
| Bidirectional necessity (0.4B pilot) | **−0.97 nats** vs `--force-forward` control |
| Any-order decoding | Kendall τ ≈ 0.07 |
| Warm-start parity | strict key-mismatch gate on HF load |

## Layout

```
models/   birwkv7_diffusion.py (BiRWKV7ForMaskedDiffusion, iterative_denoise) + torch_types.py
train/    train_birwkv_diffusion.py (FSDP masked-diffusion trainer) + parity_birwkv_warmstart.py
data/     fineweb4096 packed-pickle data pipeline
eval/     capability/ — birwkv_diffusion_model, hf_causal_model, run_eval, code, math, sandbox, merge
qz/       launch scripts + job specs (the actual H200/H100 run provenance)
docs/     tech_report.md, plan_v2_reference.md
```

## Quickstart

```bash
pip install -e .
python -c "import models.birwkv7_diffusion"   # requires fla + CUDA
```

Load the released checkpoint (Hugging Face `SII-Jiaquan/LACES-BiRWKV-DLM-2.9B-genlane`):

```python
import torch
from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion, iterative_denoise, MASK_TOKEN_ID

model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
    "RWKV/RWKV7-Goose-World3-2.9B-HF", dtype=torch.bfloat16)
model.load_state_dict(torch.load("model.safetensors", map_location="cpu"), strict=False)
```

See [docs/tech_report.md](docs/tech_report.md) for the full technical report
(architecture, training recipe, all gate numbers with CIs, and the Goal-2 code-synthesis
status), and [docs/plan_v2_reference.md](docs/plan_v2_reference.md) for the DiffRWKV 2.0
goals this implements.

## Status

Goal 1 (authentic diffusion LM) gates **all pass**. Goal 2 (beat Qwen3.5-4B on code) is in
progress — infill is strong, free synthesis is under active evaluation via a
free-generation corruption lane.

MIT licensed. Author: Jiaquan Zhang (SII-Jiaquan).
