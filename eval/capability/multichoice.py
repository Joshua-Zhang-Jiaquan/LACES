"""Multichoice knowledge eval (MMLU / ARC-Challenge / HellaSwag).

Scores a RELAY checkpoint on pre-tokenized ``.npz`` benchmarks where each file
is one example with ``N`` choices:

* ``input_ids``    — ``[N, L]`` token ids; row ``j`` is ``prompt + choice_j``
* ``attention_mask`` — ``[N, L]`` bool
* ``choice_start`` — ``[N]`` int: token index where the choice span begins
* ``label``        — ``[1]`` int: the correct choice index

For each choice the scorer computes the length-normalized NLL of the choice
span under the model's next-token distribution, and the prediction is
``argmin(NLL)``. Conditions:

* ``raw``   — the frozen backbone with no diffusion injection (control arm).
* ``ddpm<N>`` / ``ddim<N>`` / ``flow<N>`` — sample latent states with ``<N>``
  diffusion steps and inject them into the KV cache before scoring the span.

This is a faithful port of
``research/DiffRwkv/scripts/eval/eval_state_hijack_multichoice_shared.py`` so the
accuracy numbers are directly comparable to the prior MMLU smoke run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence

from eval.capability.relay_model import LoadedRelay, encode_oracle_states, sample_states

_RESULT_SCHEMA: Final = "qz_capability_multichoice_v1"


@dataclass(frozen=True, slots=True)
class MultichoiceRecord:
    """One scored example."""

    document_id: str
    arm: str
    seed: int
    correct: int  # 0 or 1
    nll: float  # length-normalized NLL of the chosen-best... not used for acc; raw span NLL of label
    label: int
    pred: int

    @property
    def metrics(self) -> tuple[tuple[str, float], ...]:
        return (("correct", float(self.correct)), ("label_nll", float(self.nll)))


@dataclass(frozen=True, slots=True)
class Condition:
    """A scoring condition parsed from a name like ``ddpm100``."""

    name: str
    method: str | None  # None => raw; else ddpm/ddim/flow
    steps: int | None

    @property
    def arm(self) -> str:
        return self.name


def parse_condition(name: str) -> Condition:
    """Parse ``raw`` / ``ddpm<N>`` / ``ddim<N>`` / ``flow<N>`` / ``maskce`` /
    ``oracle`` / ``shuffled``."""
    cleaned = name.strip().lower()
    if cleaned == "raw":
        return Condition(cleaned, None, None)
    if cleaned == "maskce":
        # masked-diffusion answer-span scoring (BiRWKV denoiser models only):
        # MASK the choice span, one forward, position-aligned NLL (no shift).
        return Condition(cleaned, "maskce", None)
    if cleaned in ("oracle", "shuffled"):
        # Conditioning diagnostic arms (no sampler): inject the encoder MEAN of
        # real tokens. ``oracle`` encodes the example's OWN prompt (the upper
        # bound a prompt-conditional sampler could reach); ``shuffled`` encodes
        # a DIFFERENT example's prompt (on-manifold, wrong-prompt control that
        # separates "any real-data z" from "the right prompt's z").
        return Condition(cleaned, cleaned, 0)
    if cleaned.startswith("pfx"):
        # PREFIX-PRIMED family (_score_prefix): prime on the question only,
        # inject, score the choice as a continuation — the leak-free protocol.
        #   pfxraw            prefix control, no injection
        #   pfxddpm<N>        classic destructive inject of ddpm<N>-sampled z
        #   pfxoracle         classic destructive inject of own-prompt oracle z
        #   pfxsoft<B>        ddpm100 z, preserve-transients blend at B%
        #   pfxsoftoracle<B>  oracle z, preserve-transients blend at B%
        rest = cleaned[len("pfx"):]
        if rest == "raw":
            return Condition(cleaned, "pfxraw", None)
        if rest == "oracle":
            return Condition(cleaned, "pfxoracle", 0)
        if rest.startswith("ddpm"):
            return Condition(cleaned, "pfxddpm", int(rest[len("ddpm"):]))
        if rest.startswith("softoracle"):
            pct = int(rest[len("softoracle"):]) if rest != "softoracle" else 100
            return Condition(cleaned, "pfxsoftoracle", pct)
        if rest.startswith("soft"):
            pct = int(rest[len("soft"):]) if rest != "soft" else 100
            return Condition(cleaned, "pfxsoft", pct)
        raise ValueError(f"unknown pfx condition: {name!r}")
    if cleaned in ("soft", "softoracle"):
        # Preserve-transients arms: same injected z as ddpm100/oracle, but the
        # recurrent state is swapped WITHOUT zeroing conv/ffn shift registers or
        # seen_tokens. ``soft`` = ddpm100 z; ``softoracle`` = own-prompt
        # encoder-mean z. steps=100 = full replacement (the historical arm).
        return Condition(cleaned, cleaned, 100)
    if cleaned.startswith("softoracle"):
        # softoracle<N>: oracle z convex-blended into the model's OWN recurrent
        # state at N% (own·(1−N/100) + oracle·N/100), transients preserved.
        # The preserve diagnostic showed FULL replacement (softoracle == N=100)
        # loses the prompt's recurrent memory (−12 pts); small N keeps that
        # memory while adding conditioned signal — the last untested corner.
        pct = int(cleaned[len("softoracle"):])
        if not 0 <= pct <= 100:
            raise ValueError(f"softoracle percent out of range: {pct}")
        return Condition(cleaned, "softoracle", pct)
    if cleaned.startswith("blend"):
        # Injection-STRENGTH diagnostic: ddpm100-sample z, then convex-blend the
        # predicted state into the primed cache at <N>% (blend_into_cache).
        # blend100 == full replacement (ddpm100); blend0 keeps the recurrent
        # state but still zeroes transient conv/ffn states + seen_tokens — i.e.
        # it isolates the cache-surgery overhead from the content replacement.
        tail = cleaned[len("blend"):]
        if not tail:
            raise ValueError(f"condition {cleaned!r} has no blend percent")
        pct = int(tail)
        if not 0 <= pct <= 100:
            raise ValueError(f"blend percent out of range: {pct}")
        return Condition(cleaned, "blend", pct)
    for method in ("ddpm", "ddim", "flow"):
        if cleaned.startswith(method):
            tail = cleaned[len(method):]
            if not tail:
                raise ValueError(f"condition {cleaned!r} has no step count")
            return Condition(cleaned, method, int(tail))
    raise ValueError(f"unknown condition: {name!r}")


def list_npz_files(task_dir: str | Path, max_samples: int | None = None) -> list[Path]:
    """Sorted ``.npz`` example files in ``task_dir``."""
    files = sorted(Path(task_dir).glob("*.npz"))
    if max_samples is not None:
        files = files[: max(0, int(max_samples))]
    return files


def shard_files(files: Sequence[Path], shard_index: int, num_shards: int) -> list[Path]:
    """Round-robin slice of ``files`` for ``shard_index/num_shards``."""
    if num_shards <= 1:
        return list(files)
    return [f for i, f in enumerate(files) if i % num_shards == shard_index]


@torch.no_grad()
def _span_nll(
    logits: torch.Tensor,
    ids: torch.Tensor,
    mask: torch.Tensor,
    choice_start: int,
    length_normalize: bool,
) -> float:
    """Length-normalized NLL of the choice span ``[choice_start, L)``."""
    if choice_start <= 0:
        choice_start = 1
    pred_logits = logits[0, choice_start - 1: ids.shape[1] - 1].float()
    targets = ids[0, choice_start: ids.shape[1]]
    valid = mask[0, choice_start: ids.shape[1]].bool()
    if valid.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[valid]
    total = float(nll.sum().item())
    return total / max(1, int(nll.numel())) if length_normalize else total


@torch.no_grad()
def _score_raw(model, ids, mask, choice_start, length_normalize) -> float:
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    return _span_nll(out.logits, ids, mask, choice_start, length_normalize)


@torch.no_grad()
def _score_maskce(model, ids, mask, choice_start, length_normalize) -> float:
    """Masked-diffusion span scoring: MASK the choice span, position-aligned NLL.

    ``model`` is a BiRWKV7ForMaskedDiffusion (logits[t] predicts token t, no
    shift). The span is fully masked so the score is the denoiser's one-step
    reconstruction NLL of the answer given the bidirectional context.
    """
    from models.birwkv7_diffusion import MASK_TOKEN_ID

    if choice_start <= 0:
        choice_start = 1
    corrupted = ids.clone()
    valid = mask[0].bool()
    span = torch.zeros_like(valid)
    span[choice_start:] = True
    span &= valid
    if span.sum() == 0:
        return float("inf")
    corrupted[0, span] = MASK_TOKEN_ID
    logits = model(corrupted, False).float()
    log_probs = F.log_softmax(logits[0, span], dim=-1)
    targets = ids[0, span]
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(nll.sum().item())
    return total / max(1, int(nll.numel())) if length_normalize else total


@torch.no_grad()
def _inject_preserving_transients(cache, states, blend=1.0):
    """Inject recurrent states WITHOUT the destructive cache surgery.

    ``inject_predicted_states`` (models/state_hijacking_cache.py) zeroes the
    RWKV token-shift registers (conv_state/ffn_state) and resets seen_tokens —
    and the strength sweep proved THAT costs ~8 pts on its own. The preserve
    diagnostic then proved full recurrent REPLACEMENT costs ~12 pts even with
    transients kept (prompt working memory lives in the recurrent state).
    ``blend`` < 1 therefore convex-mixes the injected state into the model's
    OWN state (own·(1−b) + injected·b), keeping prompt memory while adding
    conditioned signal. Transients and counters stay intact in all cases.
    """
    for layer_index, predicted_state in enumerate(states):
        layer = cache.layers[layer_index]
        if layer.state is None:
            layer.state = {}
        planned = predicted_state.to(torch.float32)
        current = layer.state.get("recurrent_state")
        if isinstance(current, torch.Tensor) and 0.0 < blend < 1.0:
            layer.state["recurrent_state"] = current.to(torch.float32) * (1.0 - blend) + planned * blend
        else:
            layer.state["recurrent_state"] = planned
    return cache


@torch.no_grad()
def _score_injected(model, ids, mask, choice_start, states, length_normalize, blend=None) -> float:
    out_pool = model.rwkv_model(
        input_ids=ids,
        attention_mask=mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    if blend is None:
        cache = model.inject_into_cache(out_pool.past_key_values, states)
    elif isinstance(blend, str) and blend.startswith("preserve"):
        frac = float(blend.split(":")[1]) if ":" in blend else 1.0
        cache = _inject_preserving_transients(out_pool.past_key_values, states, blend=frac)
    else:
        cache = model.blend_into_cache(out_pool.past_key_values, states, blend)
    out = model.rwkv_model(
        input_ids=ids,
        attention_mask=mask.bool(),
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    return _span_nll(out.logits, ids, mask, choice_start, length_normalize)


@torch.no_grad()
def _score_prefix(
    model, ids, mask, choice_start, length_normalize,
    states=None, mode=None, blend_frac=1.0,
) -> float:
    """Prefix-primed scoring: no token is ever scored twice.

    The classic ``_score_injected`` primes the cache on the FULL sequence and
    then re-scores that same sequence from the primed cache — any arm that
    retains the model's own state therefore re-scores seen text (label_nll
    collapses ~2.86→0.83 and argmin discrimination decays toward chance; the
    softblend sweep exposed this). Here the cache is primed on the question
    prefix ``[0, choice_start)`` ONLY, the injection variant is applied, and
    the choice span is scored as a continuation. The first choice token is
    predicted from the prefix's last (pre-injection) logits — exactly the
    semantics of ``RelayGenerator.generate``, which samples its first token
    from the primed logits before the injected cache takes over.
    """
    if choice_start <= 0:
        choice_start = 1
    prefix_ids = ids[:, :choice_start]
    prefix_mask = mask[:, :choice_start]
    out = model.rwkv_model(
        input_ids=prefix_ids,
        attention_mask=prefix_mask.bool(),
        use_cache=True,
        return_dict=True,
    )
    cache = out.past_key_values
    if states is not None:
        if mode == "classic":
            cache = model.inject_into_cache(cache, states)
        elif mode == "preserve":
            cache = _inject_preserving_transients(cache, states, blend=blend_frac)
        else:
            raise ValueError(f"unknown prefix inject mode: {mode!r}")
    first_logits = out.logits[:, -1:]
    cont_ids = ids[:, choice_start:]
    cont_mask = mask[:, choice_start:]
    if cont_ids.shape[1] == 0:
        return float("inf")
    out2 = model.rwkv_model(
        input_ids=cont_ids,
        attention_mask=cont_mask.bool(),
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    pred_logits = torch.cat([first_logits, out2.logits[:, :-1]], dim=1).float()
    log_probs = F.log_softmax(pred_logits[0], dim=-1)
    nll = -log_probs.gather(1, cont_ids[0].unsqueeze(-1)).squeeze(-1)
    nll = nll[cont_mask[0].bool()]
    if nll.numel() == 0:
        return float("inf")
    total = float(nll.sum().item())
    return total / max(1, int(nll.numel())) if length_normalize else total


def eval_multichoice(
    loaded: LoadedRelay,
    *,
    task_dir: str | Path,
    condition: Condition,
    shard_index: int = 0,
    num_shards: int = 1,
    max_samples: int | None = None,
    length_normalize: bool = True,
    seed: int = 42,
    progress_every: int = 50,
) -> tuple[list[MultichoiceRecord], dict]:
    """Score every sharded example in ``task_dir`` under ``condition``.

    Returns ``(records, header)`` where ``header`` carries task/condition/shard
    provenance for the per-shard JSON.
    """
    model = loaded.model
    device = str(loaded.device)
    dtype = loaded.dtype
    task_name = Path(task_dir).name
    all_files = list_npz_files(task_dir, max_samples)
    files = shard_files(all_files, shard_index, num_shards)

    records: list[MultichoiceRecord] = []
    correct = 0
    for file_i, fp in enumerate(files):
        data = np.load(fp)
        input_ids = data["input_ids"]
        attn_mask = data["attention_mask"]
        choice_start = data["choice_start"]
        label = int(data["label"][0])
        states = None
        if condition.method == "maskce":
            pass  # no state sampling; scored directly by the denoiser
        elif condition.method in ("oracle", "shuffled", "softoracle", "pfxoracle", "pfxsoftoracle"):
            # Encode real tokens through the model's own encoder (mean, no
            # noise). oracle/softoracle: this example's first choice-sequence
            # (question text is shared across choices; the pooled mean is
            # dominated by it). shuffled: a different example's tokens — same
            # manifold, wrong prompt. Deterministic peer keeps shards reproducible.
            src = data
            if condition.method == "shuffled":
                peer = files[(file_i + len(files) // 2) % len(files)]
                if peer == fp:  # single-file shard: fall back to self (logged n excludes nothing)
                    peer = files[(file_i + 1) % len(files)]
                src = np.load(peer)
            src_ids = torch.from_numpy(src["input_ids"][0:1].astype(np.int64)).to(device)
            src_mask = torch.from_numpy(src["attention_mask"][0:1].astype(np.int64)).to(device)
            if condition.method in ("pfxoracle", "pfxsoftoracle"):
                # Leak-free arms must not let the choice text into z either:
                # encode ONLY the question prefix (what generation would have).
                pcs = max(1, int(src["choice_start"][0]))
                src_ids = src_ids[:, :pcs]
                src_mask = src_mask[:, :pcs]
            states = encode_oracle_states(model, src_ids, src_mask)
        elif condition.method in ("soft", "pfxddpm"):
            steps = condition.steps if condition.method == "pfxddpm" else 100
            states = sample_states(
                model,
                "ddpm",
                steps,
                device,
                dtype,
                seed + 100_000 * (hash(condition.name) % 997) + file_i,
            )
        elif condition.method == "pfxsoft":
            states = sample_states(
                model,
                "ddpm",
                100,
                device,
                dtype,
                seed + 100_000 * (hash(condition.name) % 997) + file_i,
            )
        elif condition.method == "pfxraw":
            pass  # prefix control: no states
        elif condition.method == "blend":
            # strength arm: ddpm100-quality states, blended at condition.steps %
            states = sample_states(
                model,
                "ddpm",
                100,
                device,
                dtype,
                seed + 100_000 * (hash(condition.name) % 997) + file_i,
            )
        elif condition.method is not None:
            assert condition.steps is not None
            states = sample_states(
                model,
                condition.method,
                condition.steps,
                device,
                dtype,
                seed + 100_000 * (hash(condition.name) % 997) + file_i,
            )
        scores: list[float] = []
        label_nll = float("nan")
        for j in range(input_ids.shape[0]):
            ids = torch.from_numpy(input_ids[j:j + 1].astype(np.int64)).to(device)
            mask = torch.from_numpy(attn_mask[j:j + 1].astype(np.int64)).to(device)
            cs = int(choice_start[j])
            if condition.method == "maskce":
                score = _score_maskce(model, ids, mask, cs, length_normalize)
            elif condition.method is not None and condition.method.startswith("pfx"):
                if condition.method == "pfxraw":
                    score = _score_prefix(model, ids, mask, cs, length_normalize)
                elif condition.method in ("pfxddpm", "pfxoracle"):
                    score = _score_prefix(
                        model, ids, mask, cs, length_normalize,
                        states=states, mode="classic",
                    )
                else:  # pfxsoft / pfxsoftoracle
                    score = _score_prefix(
                        model, ids, mask, cs, length_normalize,
                        states=states, mode="preserve",
                        blend_frac=condition.steps / 100.0,
                    )
            elif states is None:
                score = _score_raw(model, ids, mask, cs, length_normalize)
            elif condition.method == "blend":
                score = _score_injected(
                    model, ids, mask, cs, states, length_normalize,
                    blend=condition.steps / 100.0,
                )
            elif condition.method in ("soft", "softoracle"):
                score = _score_injected(
                    model, ids, mask, cs, states, length_normalize,
                    blend=f"preserve:{condition.steps / 100.0}",
                )
            else:
                score = _score_injected(model, ids, mask, cs, states, length_normalize)
            scores.append(score)
            if j == label:
                label_nll = score
        pred = int(np.argmin(scores))
        is_correct = int(pred == label)
        correct += is_correct
        records.append(
            MultichoiceRecord(
                document_id=fp.stem,
                arm=condition.arm,
                seed=seed,
                correct=is_correct,
                nll=label_nll,
                label=label,
                pred=pred,
            )
        )
        if progress_every and ((file_i + 1) % progress_every == 0 or file_i + 1 == len(files)):
            n = file_i + 1
            print(
                f"{condition.arm}/{task_name}: files={n}/{len(files)} "
                f"acc={correct / max(1, n):.4f}",
                flush=True,
            )
    header = {
        "schema": _RESULT_SCHEMA,
        "task": task_name,
        "task_dir": str(task_dir),
        "arm": condition.arm,
        "condition_method": condition.method,
        "condition_steps": condition.steps,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_scored": len(files),
        "n_total_in_task": len(all_files),
        "length_normalize": length_normalize,
        "seed": seed,
        "checkpoint_step": loaded.step,
    }
    return records, header
