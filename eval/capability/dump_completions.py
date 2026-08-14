"""Dump raw BiRWKV-diffusion completions for a few code tasks (diagnostic).

The pass@1=0.0 evals report syntax_error but discard the generated text; this
prints prompt + raw completion + extracted code for N tasks so the failure
mode (format mismatch? canvas noise? chat-style output?) is directly visible.

Usage: python -m eval.capability.dump_completions --ckpt_dir ... --model_dir ...
       --tasks_file .../humaneval.jsonl --n 6 --steps 16 --out /path/dump.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCALE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tasks_file", required=True)
    p.add_argument("--suite", default="humaneval")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from eval.capability.birwkv_diffusion_model import (
        BiRWKVDiffusionGenerator,
        load_birwkv_diffusion,
    )
    from eval.capability.code import (
        _CODE_PROMPT_SUFFIX,
        extract_code,
        load_code_tasks,
        normalize_problem,
    )

    loaded = load_birwkv_diffusion(args.ckpt_dir, args.model_dir)
    gen = BiRWKVDiffusionGenerator(loaded, steps=args.steps)
    raw_tasks = load_code_tasks(args.tasks_file, shard_index=0, num_shards=1)[: args.n]

    lines = []
    for raw in raw_tasks:
        prob = normalize_problem(args.suite, raw)
        prompt = prob["prompt"] + _CODE_PROMPT_SUFFIX
        completion = gen.generate(prompt, max_new_tokens=args.max_new_tokens,
                                  temperature=0.2,
                                  stop_strings=tuple(prob["stop_strings"]))
        code = extract_code(completion, entry_hint=prob.get("entry_point", ""))
        lines += [
            "=" * 80,
            f"TASK {prob['task_id']}",
            "-" * 40 + " PROMPT (tail 400 chars)",
            prompt[-400:],
            "-" * 40 + " RAW COMPLETION",
            completion[:1500],
            "-" * 40 + " EXTRACTED CODE",
            (code or "<none>")[:800],
        ]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(raw_tasks)} tasks)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
