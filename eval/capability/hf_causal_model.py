"""Eval adapter for a vanilla HuggingFace causal LM (the Goal-2 baseline arm).

Loads any ``AutoModelForCausalLM``-compatible HF checkpoint (used for the
Qwen3.5-4B baseline: Goal-2 = beat the locally-present Qwen3.5-4B on the
project's own code/math/multichoice harness) and exposes the same duck-typed
``generate(prompt, **kw) -> str`` interface that ``eval_code`` / ``eval_math``
call, so the scorer paths run unchanged.

Differences vs the BiRWKV diffusion arm:
* No diffusion / iterative denoise — generation is a plain autoregressive
  ``model.generate``.
* The baseline is an *instruct* model, so the prompt is wrapped in the model's
  own chat template (``apply_chat_template``) before generation. This is the
  fair, strong baseline protocol — an instruct model used without its chat
  frame is unfairly hobbled. If the template fails to render (e.g. a multimodal
  template with vision macros on a text-only message), a minimal
  ``<|im_start|>...`` fallback is used.
* ``greedy=True`` (default) gives a deterministic pass@1, the standard baseline
  reporting convention; ``greedy=False`` reproduces the sampling contract
  (temperature / top_k / top_p / repetition_penalty) used by the diffusion arm.

Requires CUDA for real eval (the pod). Local CPU loads are a smoke probe only.
``model_dir`` supplies config + weights + tokenizer; ``ckpt_dir`` is accepted
for interface parity but ignored (the HF dir IS the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoadedHFCausal:
    model: object  # PreTrainedModel
    tokenizer: object
    model_dir: str
    step: int  # always 0 for a from-pretrained HF baseline (no training step)
    device: str = "cuda"
    dtype: object = torch.bfloat16


def load_hf_causal(
    model_dir: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedHFCausal:
    """Load an HF causal LM + tokenizer from ``model_dir``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=True,
    ).to(device).eval()

    return LoadedHFCausal(
        model=model,
        tokenizer=tokenizer,
        model_dir=str(model_dir),
        step=0,
        device=device,
        dtype=dtype,
    )


class HFCausalGenerator:
    """Duck-typed generator: chat-framed prompt -> autoregressive decode."""

    def __init__(
        self,
        loaded: LoadedHFCausal,
        *,
        steps: int = 0,  # unused (diffusion parity); kept for call-site compatibility
        seed: int = 42,
        greedy: bool = True,
        chat_frame: bool = True,
        enable_thinking: bool | None = None,
    ) -> None:
        self.loaded = loaded
        self.steps = steps
        self.seed = seed
        self.greedy = greedy
        self.chat_frame = chat_frame
        # Qwen3.5 (and other thinking models) inject a ``<think>`` block via the
        # chat template. For pass@1 code/math we want the answer directly, not a
        # reasoning trace that may itself contain fenced code (which would fool
        # extract_code's first-fence heuristic). Default: thinking OFF. Override
        # with env HF_ENABLE_THINKING=1 for the thinking-on arm.
        if enable_thinking is None:
            import os
            enable_thinking = os.environ.get("HF_ENABLE_THINKING", "0") == "1"
        self.enable_thinking = enable_thinking

    def _frame(self, prompt: str) -> str:
        """Wrap ``prompt`` as a single user turn via the model's chat template.

        For thinking-capable templates (Qwen3.5), ``enable_thinking=False``
        pre-emits an empty ``</think>`` so generation goes straight to the
        answer — the thinking trace can itself contain fenced code that would
        fool extract_code's first-fence heuristic. Falls back to a manual
        ``<|im_start|>`` frame if the template rejects a text-only message.
        """
        tok = self.loaded.tokenizer
        if self.chat_frame:
            kw = dict(
                add_generation_prompt=True,
                tokenize=False,
            )
            if self.enable_thinking is not None:
                kw["enable_thinking"] = self.enable_thinking
            try:
                framed = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **kw,
                )
                if framed:
                    return framed
            except Exception:
                pass  # fall through to manual frame
            # Manual fallback using Qwen-style markers (matches Qwen3.5 + most
            # instruct models). Harmless if the model uses different markers —
            # generation still proceeds from the raw prompt.
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

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
        tok = self.loaded.tokenizer
        model = self.loaded.model
        device = next(model.parameters()).device
        torch.manual_seed(self.seed)

        framed = self._frame(prompt)
        enc = tok(framed, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None

        eos = tok.eos_token_id
        pad = tok.pad_token_id
        if pad is None:
            pad = eos  # greedy batch-of-1 is fine with pad=eos

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=int(max_new_tokens),
            do_sample=not self.greedy,
            pad_token_id=pad,
            eos_token_id=eos,
        )
        if not self.greedy:
            gen_kwargs.update(
                temperature=max(float(temperature), 1e-6),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )

        out = model.generate(**gen_kwargs)
        new_ids = out[0, input_ids.shape[1]:]
        text = tok.decode(new_ids, skip_special_tokens=True)

        if stop_strings:
            cut = len(text)
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]
        return text
