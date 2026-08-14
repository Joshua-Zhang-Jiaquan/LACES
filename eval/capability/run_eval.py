"""Capability eval runner CLI — the qz-job entrypoint.

One invocation scores one (task, condition, shard) triple and writes a per-shard
JSON that ``merge_shards.py`` folds into a bootstrap-CI summary. Dispatch:

* ``multichoice`` — MMLU / ARC-Challenge / HellaSwag (pre-tokenized ``.npz``).
* ``math``       — GSM8K / MATH-500 (vendored JSONL, generative exact-match).
* ``code``       — HumanEval / MBPP (vendored JSONL, generative + sandboxed exec).

Local work is CPU-only (repo rule); this script requires CUDA and is meant to
run inside an approved qz job (see ``scale/qz/launch_capability_eval.sh``).

Example (multichoice, MMLU, raw + ddpm100, shard 0/8)::

    python -m eval.capability.run_eval \\
        --ckpt_dir /path/to/step_00100000 \\
        --task multichoice --task_dir /path/to/mmlu/validation \\
        --conditions raw,ddpm100 --shard 0 --num-shards 8 \\
        --output results/mmlu_raw_ddpm100_s0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Bootstrap: make ``eval.capability.*`` importable AND keep our package dir OFF
# sys.path so the ``code.py`` / ``math.py`` submodules don't shadow stdlib
# ``code`` / ``math`` (torch → pdb → ``import code``). Prefer
# ``python -m eval.capability.run_eval``; this guard makes script-mode safe too.
_HERE = Path(__file__).resolve().parent
_SCALE_ROOT = _HERE.parents[1]  # scale/
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]
if str(_SCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCALE_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiffRWKV capability eval runner")
    p.add_argument("--ckpt_dir", required=True, help="checkpoint dir with model.pt")
    p.add_argument(
        "--model_kind",
        default="relay",
        choices=("relay", "birwkv_diffusion", "hf_causal"),
        help="relay = frozen-renderer RELAY ckpt; birwkv_diffusion = flat "
             "BiRWKV7ForMaskedDiffusion state dict (needs --model_dir); "
             "hf_causal = vanilla HF AutoModelForCausalLM baseline (needs "
             "--model_dir = HF dir; --ckpt_dir is provenance-only and should "
             "point at the same HF dir).",
    )
    p.add_argument(
        "--model_dir",
        default=None,
        help="(birwkv_diffusion) HF dir supplying geometry + tokenizer",
    )
    p.add_argument("--task", required=True, choices=("multichoice", "math", "code"))
    p.add_argument("--task_dir", help="(multichoice) dir of .npz examples")
    p.add_argument("--tasks_file", help="(math/code) vendored JSONL task file")
    p.add_argument(
        "--suite",
        default="humaneval",
        choices=("humaneval", "mbpp"),
        help="(code) assembly rule: humaneval=concat, mbpp=standalone",
    )
    p.add_argument("--task_name", default="task", help="label for the task in output JSON")
    p.add_argument("--conditions", default="raw", help="comma-sep conditions (multichoice)")
    p.add_argument("--arm", default=None, help="arm label (math/code); defaults to condition")
    p.add_argument(
        "--sample_steps", type=int, default=100, help="diffusion steps for injected conditions"
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--max_new_tokens", type=int, default=512, help="gen budget for math/code"
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--progress_every", type=int, default=50)
    p.add_argument("--output", required=True)
    # provenance stubs the qz launcher fills in (required for merge consistency)
    p.add_argument("--registry_hash", default=os.environ.get("CAPABILITY_REGISTRY_HASH", "unpinned"))
    p.add_argument("--profile_sha256", default=os.environ.get("CAPABILITY_PROFILE_SHA256", "unpinned"))
    p.add_argument(
        "--condition_profile_hash",
        default=os.environ.get("CAPABILITY_CONDITION_PROFILE_HASH", "unpinned"),
    )
    return p.parse_args(argv)


def _write(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def parse_generative_condition(name: str, default_steps: int) -> tuple[str, str, int]:
    """Parse a math/code condition into ``(arm, method, steps)``.

    Same grammar as the multichoice ``parse_condition`` (``raw`` /
    ``ddpm<N>`` / ``ddim<N>`` / ``flow<N>``); ``raw`` maps to the
    no-injection control handled by ``RelayGenerator(method="raw")``.
    """
    cleaned = name.strip().lower()
    if cleaned == "raw":
        return cleaned, "raw", default_steps
    if cleaned == "greedy":
        # Deterministic / no-injection baseline arm. For hf_causal this is the
        # standard pass@1 convention (greedy decode); for relay it behaves like
        # ``raw`` (no diffusion injection). Distinct arm label keeps the merged
        # JSON honest about what was run.
        return cleaned, "raw", default_steps
    for method in ("ddpm", "ddim", "flow", "iter"):
        if cleaned.startswith(method):
            tail = cleaned[len(method):]
            if not tail:
                raise ValueError(f"condition {cleaned!r} has no step count")
            return cleaned, method, int(tail)
    raise ValueError(f"unknown condition: {name!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.device == "cpu":
        print("WARNING: CUDA unavailable; real capability eval needs a qz GPU job.", flush=True)

    from eval.capability.relay_model import load_relay

    if args.model_kind == "birwkv_diffusion":
        if not args.model_dir:
            raise SystemExit("--model_dir is required for --model_kind birwkv_diffusion")
        from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion

        loaded = load_birwkv_diffusion(args.ckpt_dir, args.model_dir, device=args.device)
    elif args.model_kind == "hf_causal":
        if not args.model_dir:
            raise SystemExit("--model_dir is required for --model_kind hf_causal")
        from eval.capability.hf_causal_model import load_hf_causal

        loaded = load_hf_causal(args.model_dir, device=args.device)
    else:
        loaded = load_relay(args.ckpt_dir, device=args.device)
    checkpoint = str(Path(args.ckpt_dir).resolve())
    seeds = (args.seed,)

    records_payload: list[dict] = []
    metric_schema: tuple[str, ...]
    header_extra: dict

    if args.task == "multichoice":
        metric_schema = ("correct", "label_nll")
        from eval.capability.multichoice import (  # type: ignore[import]
            eval_multichoice,
            parse_condition,
        )

        cond_names = [c for c in args.conditions.split(",") if c.strip()]
        if not args.task_dir:
            raise SystemExit("--task_dir is required for multichoice")
        for cond_name in cond_names:
            cond = parse_condition(cond_name)
            records, header = eval_multichoice(
                loaded,
                task_dir=args.task_dir,
                condition=cond,
                shard_index=args.shard,
                num_shards=args.num_shards,
                max_samples=args.max_samples,
                seed=args.seed,
                progress_every=args.progress_every,
            )
            for r in records:
                records_payload.append(
                    {
                        "document_id": r.document_id,
                        "arm": r.arm,
                        "seed": r.seed,
                        "metrics": dict(r.metrics),
                    }
                )
            print(f"multichoice {cond_name}: {header['n_scored']} scored", flush=True)
        header_extra = {"task_dir": args.task_dir, "conditions": args.conditions}

    elif args.task == "math":
        metric_schema = ("correct",)
        from eval.capability.math import eval_math, load_math_tasks
        from eval.capability.relay_model import RelayGenerator

        if not args.tasks_file:
            raise SystemExit("--tasks_file is required for math")
        tasks = load_math_tasks(
            args.tasks_file, shard_index=args.shard, num_shards=args.num_shards
        )
        if args.max_samples:
            tasks = tasks[: args.max_samples]
        # Mirror the multichoice contract: every generative shard scores each
        # condition in --conditions (raw = no-injection control, ddpm<N> =
        # injected). --arm overrides to a single explicitly-labeled arm, which
        # is what the probe uses.
        if args.arm:
            probe_method = "iter" if args.model_kind == "birwkv_diffusion" else "ddpm"
            conditions = [(args.arm, probe_method, args.sample_steps)]
        else:
            conditions = []
            for cond_name in (c.strip() for c in args.conditions.split(",")):
                if not cond_name:
                    continue
                cond = parse_generative_condition(cond_name, args.sample_steps)
                conditions.append(cond)
        for arm, method, steps in conditions:
            if args.model_kind == "birwkv_diffusion":
                if method != "iter":
                    raise SystemExit(
                        f"model_kind birwkv_diffusion supports iter<N> conditions only, got {arm!r}"
                    )
                from eval.capability.birwkv_diffusion_model import BiRWKVDiffusionGenerator

                generator = BiRWKVDiffusionGenerator(loaded, steps=steps, seed=args.seed)
            elif args.model_kind == "hf_causal":
                from eval.capability.hf_causal_model import HFCausalGenerator

                # hf_causal has no diffusion steps; the condition arm label is
                # carried through verbatim for merge grouping. greedy=True gives
                # the standard deterministic pass@1 baseline.
                generator = HFCausalGenerator(loaded, steps=steps, seed=args.seed)
            else:
                generator = RelayGenerator(
                    loaded,
                    method=method,
                    sample_steps=steps,
                    seed=args.seed,
                )
            records, header = eval_math(
                generator,
                tasks=tasks,
                arm=arm,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                progress_every=args.progress_every,
            )
            for r in records:
                records_payload.append(
                    {
                        "document_id": r.document_id,
                        "arm": r.arm,
                        "seed": r.seed,
                        "metrics": dict(r.metrics),
                    }
                )
            print(f"math {arm}: {header['n_scored']} scored", flush=True)
        header_extra = {"tasks_file": args.tasks_file, "conditions": args.conditions}

    else:  # code
        metric_schema = ("pass_at_1",)
        from eval.capability.code import eval_code, load_code_tasks
        from eval.capability.relay_model import RelayGenerator

        if not args.tasks_file:
            raise SystemExit("--tasks_file is required for code")
        tasks = load_code_tasks(
            args.tasks_file, shard_index=args.shard, num_shards=args.num_shards
        )
        if args.max_samples:
            tasks = tasks[: args.max_samples]
        if args.arm:
            probe_method = "iter" if args.model_kind == "birwkv_diffusion" else "ddpm"
            conditions = [(args.arm, probe_method, args.sample_steps)]
        else:
            conditions = []
            for cond_name in (c.strip() for c in args.conditions.split(",")):
                if not cond_name:
                    continue
                cond = parse_generative_condition(cond_name, args.sample_steps)
                conditions.append(cond)
        for arm, method, steps in conditions:
            if args.model_kind == "birwkv_diffusion":
                if method != "iter":
                    raise SystemExit(
                        f"model_kind birwkv_diffusion supports iter<N> conditions only, got {arm!r}"
                    )
                from eval.capability.birwkv_diffusion_model import BiRWKVDiffusionGenerator

                generator = BiRWKVDiffusionGenerator(loaded, steps=steps, seed=args.seed)
            elif args.model_kind == "hf_causal":
                from eval.capability.hf_causal_model import HFCausalGenerator

                # hf_causal has no diffusion steps; the condition arm label is
                # carried through verbatim for merge grouping. greedy=True gives
                # the standard deterministic pass@1 baseline.
                generator = HFCausalGenerator(loaded, steps=steps, seed=args.seed)
            else:
                generator = RelayGenerator(
                    loaded,
                    method=method,
                    sample_steps=steps,
                    seed=args.seed,
                )
            records, header = eval_code(
                generator,
                tasks=tasks,
                suite=args.suite,
                arm=arm,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                progress_every=args.progress_every,
            )
            for r in records:
                records_payload.append(
                    {
                        "document_id": r.document_id,
                        "arm": r.arm,
                        "seed": r.seed,
                        "metrics": dict(r.metrics),
                        "isolation_mode": r.isolation_mode,
                        "failure": r.failure,
                    }
                )
            print(f"code {arm}: {header['n_scored']} scored", flush=True)
        header_extra = {
            "tasks_file": args.tasks_file,
            "suite": args.suite,
            "conditions": args.conditions,
        }

    payload = {
        "schema": f"qz_capability_{args.task}_shard_v1",
        "task": args.task_name,
        "task_kind": args.task,
        "checkpoint": checkpoint,
        "checkpoint_step": loaded.step,
        "registry_hash": args.registry_hash,
        "profile_sha256": args.profile_sha256,
        "condition_profile_hash": args.condition_profile_hash,
        "seeds": list(seeds),
        "metric_schema": list(metric_schema),
        "shard_index": args.shard,
        "num_shards": args.num_shards,
        "n_records": len(records_payload),
        "records": records_payload,
        **header_extra,
    }
    _write(args.output, payload)
    print(f"wrote {args.output} ({len(records_payload)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
