"""Eval adapter for the BiRWKV token-diffusion denoiser.

Loads a flat-state-dict checkpoint (written by train_birwkv_diffusion.py) into
BiRWKV7ForMaskedDiffusion and exposes the duck-typed ``generate(prompt, **kw)
-> str`` interface that eval_code/eval_math call — the denoiser's own lm_head
produces the tokens via the iterative confidence-commit sampler. No frozen
renderer anywhere in the path.

Requires CUDA (fla kernels). ``model_dir`` supplies the HF geometry + tokenizer
(config.json / tokenizer files); ``ckpt_dir`` supplies model.pt + meta.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LoadedBiRWKVDiffusion:
    model: object  # BiRWKV7ForMaskedDiffusion
    tokenizer: object
    step: int
    ckpt_dir: str
    device: str = "cuda"
    dtype: object = torch.bfloat16


def load_birwkv_diffusion(
    ckpt_dir: str,
    model_dir: str,
    device: str = "cuda",
) -> LoadedBiRWKVDiffusion:
    """Build geometry from the HF dir, then load the flat training state dict."""
    from transformers import AutoTokenizer

    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    ckpt_path = Path(ckpt_dir)
    model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(model_dir, dtype=torch.bfloat16)
    state = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
    state = {k: v.to(torch.bfloat16) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if "fuse_" not in k] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch: {bad[:8]}")
    model = model.to(device).eval()

    step = 0
    meta = ckpt_path / "meta.json"
    if meta.exists():
        step = int(json.loads(meta.read_text()).get("step", 0))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return LoadedBiRWKVDiffusion(model=model, tokenizer=tokenizer, step=step,
                                 ckpt_dir=str(ckpt_path.resolve()), device=device)


class BiRWKVDiffusionGenerator:
    """Duck-typed generator: prompt -> masked continuation -> iterative denoise."""

    # The masked-diffusion training corpora (capability mixture + code mixture)
    # are SFT/agent-trajectory formatted with plain-text "Assistant:" turn
    # markers. Bare-code prompts are out-of-distribution and the sampler
    # commits degenerate chat-marker tokens ("</istant>...") first — verified
    # by completion dumps 2026-08-12. Wrapping the prompt in the training
    # frame + opening a fenced code block puts generation in-distribution;
    # extract_code() prefers fenced blocks, and "```" is the stop.
    CHAT_PREFIX = "User: {prompt}\n\nAssistant:\n```python\n"

    def __init__(
        self,
        loaded: LoadedBiRWKVDiffusion,
        steps: int = 16,
        seed: int = 42,
        self_correction: bool = False,
        block_round: int = 32,
        chat_frame: bool = True,
    ):
        # self_correction defaults OFF: the offline sampler grid (2026-08-11)
        # showed remask_threshold=0.25 reopens most CORRECT commits (typical
        # true-token prob << 0.25 at this checkpoint's entropy), collapsing em
        # by 4-6x vs single-shot. Re-enable only with a recalibrated threshold.
        self.loaded = loaded
        self.steps = steps
        self.seed = seed
        self.self_correction = self_correction
        self.block_round = block_round
        self.chat_frame = chat_frame

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_strings: tuple[str, ...] | None = None,
    ) -> str:
        # top_k/top_p/repetition_penalty accepted for interface parity; the
        # confidence-commit sampler uses temperature only.
        del top_k, top_p, repetition_penalty
        from models.birwkv7_diffusion import MASK_TOKEN_ID, iterative_denoise

        model = self.loaded.model
        device = next(iter(model.parameters())).device
        torch.manual_seed(self.seed)

        framed = self.CHAT_PREFIX.format(prompt=prompt) if self.chat_frame else prompt
        prompt_ids = self.loaded.tokenizer(framed, return_tensors=None)["input_ids"]
        gen_len = min(int(max_new_tokens), 512)
        gen_len = ((gen_len + self.block_round - 1) // self.block_round) * self.block_round

        ids = torch.tensor([prompt_ids + [MASK_TOKEN_ID] * gen_len], device=device)
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[:, len(prompt_ids):] = True

        denoised, _ = iterative_denoise(
            model, ids, masked,
            steps=self.steps,
            temperature=temperature,
            self_correction=self.self_correction,
        )
        out_ids = denoised[0, len(prompt_ids):].tolist()
        # The denoiser fills the whole masked canvas; treat the first EOS
        # (65530, decodes to "\n\n") as end-of-completion so trailing canvas
        # noise doesn't poison parsing.
        eos_id = 65530
        if eos_id in out_ids:
            out_ids = out_ids[: out_ids.index(eos_id)]
        text = self.loaded.tokenizer.decode(out_ids)
        if self.chat_frame:
            # generation opened inside a ```python fence: close at the fence
            # and re-wrap so extract_code()'s fenced-block path fires.
            fence = text.find("```")
            body = text[:fence] if fence != -1 else text
            text = "```python\n" + body + "\n```"
        if stop_strings:
            cut = len(text)
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]
        return text
