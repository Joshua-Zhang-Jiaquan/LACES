"""Math capability eval (GSM8K / MATH-500) via prompted generation + exact match.

Unlike the multichoice family, math is generative: the model is given a
problem, generates a solution (optionally CoT), and the final answer is
extracted and compared to the gold. Scoring is exact-match on a normalized
numeric / LaTeX answer.

Answer extraction follows the standard GSM8K/MATH convention:
* prefer the last ``\\boxed{...}`` content;
* else the last number in the generation (GSM8K convention);
* normalization strips whitespace, LaTeX ``$``, surrounding ``\\text{}``,
  percent, and trailing ``.0``.

The gold answers come from a small vendored task file (``math_tasks.jsonl``)
with ``{problem, answer, task_id}``. The file is downloaded/curated once and
pinned by SHA; it lives under ``scale/eval/capability/data/``. (A TODO marker
in the launcher notes the data-acquisition step; if absent the runner reports
``tasks_unavailable`` rather than silently skipping.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

from eval.capability.relay_model import RelayGenerator

_RESULT_SCHEMA: Final = "qz_capability_math_v1"

# ── answer extraction & normalization ────────────────────────────────────────
# NOTE: a regex cannot match arbitrarily-nested braces. The previous pattern
# allowed only ONE level (`\{[^{}]*\}`), so a real MATH-500 gold answer like
# `\frac{3\sqrt{3}}{4}` (two levels deep) failed to match `\boxed{...}` and fell
# through to the bare-number branch, extracting `4`. That silently undercounts:
# 5/500 MATH-500 answers could not even score themselves. `_find_boxed` scans
# for balanced braces instead, which handles any depth.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_TOKEN = "\\boxed"


def _find_boxed(text: str) -> list[str]:
    """Return the contents of every ``\\boxed{...}``, brace-balanced, in order."""
    out: list[str] = []
    start = text.find(_BOXED_TOKEN)
    while start != -1:
        i = start + len(_BOXED_TOKEN)
        while i < len(text) and text[i] != "{":
            if not text[i].isspace():  # e.g. `\boxedx` — not a boxed answer
                i = -1
                break
            i += 1
        if i != -1 and i < len(text):
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[i + 1 : j])
                        break
        start = text.find(_BOXED_TOKEN, start + len(_BOXED_TOKEN))
    return out


def extract_answer(text: str) -> str:
    """Extract the final answer from a generated solution."""
    boxed = _find_boxed(text)
    if boxed:
        return _normalize(boxed[-1])
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _normalize(numbers[-1])
    return _normalize(text.strip().splitlines()[-1]) if text.strip() else ""


def _normalize(answer: str) -> str:
    """Normalize a math answer for exact-match comparison."""
    s = answer.strip()
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\text{", "").replace("}", "", 1) if s.startswith("\\text{") else s
    s = s.replace("%", "")
    s = s.replace("\\,", "").strip()
    # remove trailing .0 for integer-valued floats (5.0 -> 5)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    # collapse whitespace
    s = re.sub(r"\s+", "", s)
    return s


def is_correct(pred: str, gold: str) -> bool:
    """Exact-match after normalization, with a numeric tolerance fallback."""
    if pred == _normalize(gold):
        return True
    try:
        return abs(float(pred) - float(_normalize(gold))) <= 1e-3
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class MathRecord:
    document_id: str
    arm: str
    seed: int
    correct: int
    gold: str
    pred: str

    @property
    def metrics(self) -> dict[str, float]:
        return {"correct": float(self.correct)}


def load_math_tasks(path: str | Path, *, shard_index: int = 0, num_shards: int = 1) -> list[dict]:
    """Load vendored math tasks, sharded by rank."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if num_shards <= 1:
        return rows
    return [r for i, r in enumerate(rows) if i % num_shards == shard_index]


_MATH_PROMPT = (
    # The literal {} of \boxed{} must be doubled: .format() reads a bare {} as
    # a positional placeholder and raises IndexError (killed the first gsm8k/
    # math500 probes, job-fe1ef829/job-fc4a9ccc).
    "Solve the following math problem. Think step by step, then put your "
    "final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:")


def eval_math(
    generator: RelayGenerator,
    *,
    tasks: Sequence[dict],
    arm: str,
    seed: int,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stop_strings: Sequence[str] | None = None,
    progress_every: int = 25,
) -> tuple[list[MathRecord], dict]:
    """Generate + score each math task; return records + header."""
    records: list[MathRecord] = []
    correct = 0
    for i, task in enumerate(tasks):
        problem = task["problem"]
        gold = str(task["answer"])
        task_id = str(task.get("task_id", i))
        prompt = _MATH_PROMPT.format(problem=problem)
        text = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
        )
        pred = extract_answer(text)
        ok = int(is_correct(pred, gold))
        correct += ok
        records.append(MathRecord(task_id, arm, seed, ok, gold, pred))
        if progress_every and (i + 1) % progress_every == 0:
            n = i + 1
            print(f"math/{arm}: tasks={n}/{len(tasks)} acc={correct / n:.4f}", flush=True)
    header = {
        "schema": _RESULT_SCHEMA,
        "task": "math",
        "arm": arm,
        "n_scored": len(tasks),
        "seed": seed,
    }
    return records, header
