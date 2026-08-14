"""Code capability eval (HumanEval / MBPP) via prompted generation + sandboxed exec.

Generative: the model is given a function signature + docstring (HumanEval) or
an instruct-style task + test list (MBPP) and must complete the body. The
completion is fenced-code-stripped, truncated at the suite's stop strings,
assembled with the prompt per suite rules, and executed against the task's
unit tests in a hardened subprocess sandbox (``eval.capability.sandbox``).
Pass@1 is the fraction of tasks whose completion passes *all* tests.

Suite assembly (load-bearing, ported from ``RL/eval/run_eval.py``):

* ``humaneval`` — completion-style: the candidate program is ``prompt +
  completion`` (the prompt already contains the signature + docstring). The
  test is the raw ``test`` field plus ``check(<entry_point>)``. If the model
  re-emits the signature the concatenation still parses; if it only emits a
  body, the assembly still yields a runnable ``def``. Stop strings:
  ``("\\ndef ", "\\nclass ", "\\nif __name__")``.
* ``mbpp`` — full-program-style: the candidate program is the completion
  *alone* (the prompt is an instruct wrapper, not code). Tests are each
  ``test_list`` entry prefixed by ``test_imports``. Stop strings:
  ``("\\n# Test", "\\ndef check")``.

Execution uses ``eval.capability.sandbox.run_tests``: ``python -I``, scrubbed
env, network namespace or socket stub, ``RLIMIT_AS/CPU/FSIZE/NPROC``, per-test
fresh subprocess, wall-clock timeout, plus an AST/entry-point pre-check for
taxonomy. A crash or timeout is a fail, never the scorer.

Tasks come from the canonical pinned JSONL files under ``RL/data/benchmarks/``
(``humaneval.jsonl`` 164 rows, ``mbpp.jsonl`` 427 rows, pinned by SHA256 in
``RL/eval/run_eval.py``); the qz launcher points ``--tasks_file`` at one of
them and ``--suite`` selects the assembly rule.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

from eval.capability.relay_model import RelayGenerator
from eval.capability.sandbox import ExecResult, run_tests

_RESULT_SCHEMA: Final = "qz_capability_code_v1"
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

HUMANEVAL_STOPS: Final = ("\ndef ", "\nclass ", "\nif __name__")
MBPP_STOPS: Final = ("\n# Test", "\ndef check")


@dataclass(frozen=True, slots=True)
class CodeRecord:
    document_id: str
    arm: str
    seed: int
    pass_at_1: int  # 0 or 1
    task_id: str
    isolation_mode: str
    failure: str  # "" on pass; else taxonomy tag (syntax_error / entry_point_missing / timed_out / tests_failed)

    @property
    def metrics(self) -> dict[str, float]:
        return {"pass_at_1": float(self.pass_at_1)}


def load_code_tasks(path: str | Path, *, shard_index: int = 0, num_shards: int = 1) -> list[dict]:
    """Load a HumanEval/MBPP-style JSONL, sharded round-robin by rank."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if num_shards <= 1:
        return rows
    return [r for i, r in enumerate(rows) if i % num_shards == shard_index]


def extract_code(generation: str, *, entry_hint: str = "") -> str:
    """Extract a code block from a generation.

    Prefer the first ``\\`\\`\\`python`` fenced block; else fall back to the
    raw text after stripping leading prose lines that don't look like code.
    (The fenced path is NEW vs the RL/ scorer, which assumes raw completions;
    DiffRWKV chat-style generations frequently wrap code in fences.)
    """
    m = _FENCE_RE.search(generation)
    if m:
        return m.group(1)
    lines = generation.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(def |class |import |from |[a-zA-Z_]\w*\s*=)", line):
            start = i
            break
    return "\n".join(lines[start:])


def truncate_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    """Cut ``text`` at the earliest stop-string occurrence (ported from run_eval)."""
    cut = len(text)
    for stop in stop_strings:
        pos = text.find(stop)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut]


def normalize_problem(suite: str, raw: dict) -> dict:
    """Format a raw benchmark row into ``{task_id, prompt, test_cases, stop_strings, entry_point}``.

    Ported from ``RL/eval/run_eval.py:normalize_problem`` — the HumanEval-concat
    vs MBPP-standalone asymmetry is load-bearing for correct scoring.
    """
    if suite == "humaneval":
        entry = str(raw.get("entry_point", ""))
        return {
            "task_id": str(raw["task_id"]),
            "prompt": str(raw["prompt"]),
            "test_cases": [str(raw["test"]) + (f"\ncheck({entry})" if entry else "")],
            "stop_strings": HUMANEVAL_STOPS,
            "entry_point": entry,
        }
    if suite == "mbpp":
        imports = "\n".join(raw.get("test_imports") or [])
        tests = [
            f"{imports}\n{case}" if imports else str(case)
            for case in raw["test_list"]
        ]
        prompt = (
            "You are an expert Python programmer. Write only the Python code for this task.\n"
            f"Task: {raw['prompt']}\n"
            "Your code should pass these tests:\n"
            + "\n".join(raw["test_list"])
            + "\n\nPython code:\n"
        )
        return {
            "task_id": str(raw["task_id"]),
            "prompt": prompt,
            "test_cases": tests,
            "stop_strings": MBPP_STOPS,
            "entry_point": "",  # MBPP has no single entry point; tests call the functions directly
        }
    raise ValueError(f"unknown suite: {suite}")


def build_candidate_code(suite: str, prompt: str, completion: str) -> str:
    """Assemble the executable program from prompt + completion per suite.

    HumanEval: ``prompt + completion`` (completion-style). MBPP: ``completion``
    alone (full-program-style, since the prompt is an instruct wrapper).
    """
    if suite == "humaneval":
        return prompt + completion
    return completion


def _failure_tag(res: ExecResult) -> str:
    if res.passed:
        return ""
    if res.syntax_error:
        return "syntax_error"
    if res.entry_point_missing:
        return "entry_point_missing"
    if res.timed_out:
        return "timed_out"
    return "tests_failed"


_CODE_PROMPT_SUFFIX = "\n\nComplete the function body. Only output code.\n"


def eval_code(
    generator: RelayGenerator,
    *,
    tasks: Sequence[dict],
    suite: str,
    arm: str,
    seed: int,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    timeout_s: float = 5.0,
    mem_mb: int = 256,
    progress_every: int = 10,
) -> tuple[list[CodeRecord], dict]:
    """Generate + exec each code task; return records + header."""
    records: list[CodeRecord] = []
    passed = 0
    for i, raw in enumerate(tasks):
        prob = normalize_problem(suite, raw)
        task_id = prob["task_id"]
        prompt = prob["prompt"] + _CODE_PROMPT_SUFFIX
        text = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        completion = truncate_at_stop(extract_code(text), prob["stop_strings"])
        program = textwrap.dedent(build_candidate_code(suite, prob["prompt"], completion))
        res = run_tests(
            program,
            prob["test_cases"],
            timeout_s=timeout_s,
            mem_mb=mem_mb,
            entry_point=prob["entry_point"] or None,
        )
        ok = int(res.passed)
        passed += ok
        records.append(
            CodeRecord(task_id, arm, seed, ok, task_id, res.isolation_mode, _failure_tag(res))
        )
        if progress_every and (i + 1) % progress_every == 0:
            n = i + 1
            print(f"code/{arm}: tasks={n}/{len(tasks)} pass@1={passed / n:.4f}", flush=True)
    header = {
        "schema": _RESULT_SCHEMA,
        "task": f"code-{suite}",
        "arm": arm,
        "n_scored": len(tasks),
        "seed": seed,
        "suite": suite,
    }
    return records, header
