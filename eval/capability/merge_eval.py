"""CLI: fold per-shard capability-eval JSONs into a bootstrap-CI summary.

Each qz eval pod writes one JSON per ``(task, condition, shard)`` triple to a
shared GPFS output dir (see ``launch_capability_eval.sh``). This command globs
those shard files, groups them by ``(task, condition)`` (the merge key), and
folds each group via :func:`eval.capability.merge_shards.merge_shards` into a
single ``qz_capability_eval_merged_v1`` summary with per-arm mean + 95% CI.

Local CPU-only (no CUDA, no model); safe to run on the orchestrator box.

Example::

    python -m eval.capability.merge_eval \\
        --shards-glob "/gpfs/.../cap_eval/**/*.json" \\
        --output-dir /gpfs/.../cap_eval/merged
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Keep our package dir off sys.path so stdlib code/math aren't shadowed.
_HERE = Path(__file__).resolve().parent
_SCALE_ROOT = _HERE.parents[1]
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]
if str(_SCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCALE_ROOT))

from eval.capability.merge_shards import (  # noqa: E402
    ShardResult,
    load_shard,
    merge_shards,
    write_merged,
)


def _is_shard_file(path: Path) -> bool:
    """A per-shard JSON has the shard schema and a ``shard_index`` field."""
    if path.suffix != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("schema", "").startswith(
        "qz_capability_"
    ) and "shard_index" in data and "records" in data


def _merge_key(shard: ShardResult) -> tuple[str, str]:
    """Group shards by (task, arm-bucket) so conditions stay separate.

    The per-shard JSON carries ``task`` and ``task_kind``; multichoice shards
    for different conditions carry distinct ``arm`` labels inside their records,
    so we group by ``task`` only for multichoice (one merged file per task) and
    by ``task`` for math/code (one merged file per task/suite). The merged
    summary itself separates arms internally via bootstrap.
    """
    return (shard.checkpoint, _task_label(shard))


def _task_label(shard: ShardResult) -> str:
    # Reconstruct a stable label from the shard's records' arms is fragile;
    # instead we key on checkpoint+metric_schema which is constant per run.
    return ":".join(shard.metric_schema)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge capability-eval shard JSONs")
    p.add_argument(
        "--shards-glob",
        required=True,
        help="glob (quote it) matching per-shard JSON files, e.g. '/dir/**/*.json'",
    )
    p.add_argument("--output-dir", required=True, help="dir for merged JSONs")
    p.add_argument(
        "--bootstrap-samples", type=int, default=1000, help="bootstrap resamples per metric"
    )
    args = p.parse_args(argv)

    shard_paths = sorted(
        fp for fp in glob.glob(args.shards_glob, recursive=True) if _is_shard_file(Path(fp))
    )
    if not shard_paths:
        print(f"merge_eval: no shard JSONs matched {args.shards_glob!r}", file=sys.stderr)
        return 1

    shards: list[ShardResult] = []
    for fp in shard_paths:
        try:
            shards.append(load_shard(fp))
        except Exception as exc:  # noqa: BLE001
            print(f"merge_eval: skip unreadable shard {fp}: {exc!r}", file=sys.stderr)

    if not shards:
        print("merge_eval: no loadable shards", file=sys.stderr)
        return 1

    # Group by checkpoint + metric_schema (one merged file per group).
    groups: dict[tuple[str, str], list[ShardResult]] = {}
    for sh in shards:
        groups.setdefault(_merge_key(sh), []).append(sh)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for key, group in groups.items():
        merged = merge_shards(group, bootstrap_samples=args.bootstrap_samples)
        ckpt_short = Path(group[0].checkpoint).name
        schema_tag = "-".join(group[0].metric_schema)
        out_path = out_dir / f"merged_{ckpt_short}_{schema_tag}.json"
        write_merged(out_path, merged)
        print(f"merge_eval: wrote {out_path} ({len(merged.records)} records)", flush=True)
        # also print a one-line summary per arm/metric for quick eyeballing
        for m in merged.metrics:
            print(
                f"  {m.arm}/{m.metric}: mean={m.mean:.4f} "
                f"ci95=[{m.ci95_low:.4f}, {m.ci95_high:.4f}] n={m.n_documents}",
                flush=True,
            )
        wrote += 1
    print(f"merge_eval: merged {len(shards)} shards into {wrote} summary file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
