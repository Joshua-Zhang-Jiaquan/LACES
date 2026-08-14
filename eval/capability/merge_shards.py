"""Shard merge for capability eval — folds per-shard JSONs into a CI summary.

Reuses the proven packing/merge contract from ``scale/qz/run_eval_shards.py`` /
``merge_eval_shards.py`` (document-level records, bootstrap CIs) but consumes
the capability-eval record schema from ``multichoice.py`` /
``math.py`` / ``code.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_SCHEMA: Final = "qz_capability_eval_merged_v1"


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """One scored example (multichoice, math, or code)."""

    document_id: str
    arm: str
    seed: int
    metrics: dict[str, float]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.document_id, self.arm, self.seed


@dataclass(frozen=True, slots=True)
class ShardResult:
    shard_index: int
    registry_hash: str
    profile_sha256: str
    condition_profile_hash: str
    checkpoint: str
    seeds: tuple[int, ...]
    metric_schema: tuple[str, ...]
    records: tuple[CapabilityRecord, ...]


@dataclass(frozen=True, slots=True)
class MergedMetric:
    arm: str
    metric: str
    mean: float
    ci95_low: float
    ci95_high: float
    n_documents: int


@dataclass(frozen=True, slots=True)
class MergedEvaluation:
    schema: str
    task_id: str
    registry_hash: str
    profile_sha256: str
    condition_profile_hash: str
    checkpoint: str
    seeds: tuple[int, ...]
    metric_schema: tuple[str, ...]
    records: tuple[CapabilityRecord, ...]
    metrics: tuple[MergedMetric, ...]
    bootstrap_samples: int


def load_shard(path: str | Path) -> ShardResult:
    """Load a per-shard JSON written by a capability eval run."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        CapabilityRecord(
            document_id=r["document_id"],
            arm=r["arm"],
            seed=int(r["seed"]),
            metrics=dict(r["metrics"]),
        )
        for r in data.get("records", [])
    )
    return ShardResult(
        shard_index=int(data["shard_index"]),
        registry_hash=data["registry_hash"],
        profile_sha256=data["profile_sha256"],
        condition_profile_hash=data["condition_profile_hash"],
        checkpoint=data["checkpoint"],
        seeds=tuple(data["seeds"]),
        metric_schema=tuple(data["metric_schema"]),
        records=records,
    )


def _bootstrap_ci(values: Sequence[float], samples: int, salt: str) -> tuple[float, float]:
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v
    count = len(values)
    means = []
    for sample in range(samples):
        total = 0.0
        for position in range(count):
            idx = int.from_bytes(
                hashlib.sha256(f"{salt}:{sample}:{position}".encode()).digest()[:8], "big"
            ) % count
            total += values[idx]
        means.append(total / count)
    ordered = sorted(means)
    lo = _percentile(ordered, 0.025)
    hi = _percentile(ordered, 0.975)
    return lo, hi


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def merge_shards(
    shards: Sequence[ShardResult], *, bootstrap_samples: int = 1_000
) -> MergedEvaluation:
    """Merge per-shard capability records into a bootstrap-CI summary."""
    if not shards:
        raise ValueError("no shard results to merge")
    registry_hash = shards[0].registry_hash
    profile_sha256 = shards[0].profile_sha256
    condition_profile_hash = shards[0].condition_profile_hash
    checkpoint = shards[0].checkpoint
    seeds = shards[0].seeds
    metric_schema = shards[0].metric_schema
    task_id = "capability-eval"
    all_records: list[CapabilityRecord] = []
    seen: set[tuple[str, str, int]] = set()
    for shard in shards:
        if shard.registry_hash != registry_hash:
            raise ValueError("registry_hash mismatch across shards")
        for rec in shard.records:
            if rec.key in seen:
                continue
            seen.add(rec.key)
            all_records.append(rec)
    summaries: list[MergedMetric] = []
    arms = sorted({r.arm for r in all_records})
    for arm in arms:
        arm_records = [r for r in all_records if r.arm == arm]
        doc_ids = sorted({r.document_id for r in arm_records})
        for metric in metric_schema:
            values = []
            for did in doc_ids:
                docs = [r for r in arm_records if r.document_id == did]
                total = math.fsum(r.metrics.get(metric, 0.0) for r in docs) / max(1, len(seeds))
                values.append(total)
            lo, hi = _bootstrap_ci(values, bootstrap_samples, f"{arm}:{metric}")
            mean = math.fsum(values) / max(1, len(values))
            summaries.append(MergedMetric(arm, metric, mean, lo, hi, len(doc_ids)))
    return MergedEvaluation(
        schema=_SCHEMA,
        task_id=task_id,
        registry_hash=registry_hash,
        profile_sha256=profile_sha256,
        condition_profile_hash=condition_profile_hash,
        checkpoint=checkpoint,
        seeds=seeds,
        metric_schema=metric_schema,
        records=tuple(all_records),
        metrics=tuple(summaries),
        bootstrap_samples=bootstrap_samples,
    )


def write_merged(path: str | Path, merged: MergedEvaluation) -> None:
    import json

    payload = {
        "schema": merged.schema,
        "task_id": merged.task_id,
        "registry_hash": merged.registry_hash,
        "profile_sha256": merged.profile_sha256,
        "condition_profile_hash": merged.condition_profile_hash,
        "checkpoint": merged.checkpoint,
        "seeds": list(merged.seeds),
        "metric_schema": list(merged.metric_schema),
        "record_count": len(merged.records),
        "records": [
            {
                "document_id": r.document_id,
                "arm": r.arm,
                "seed": r.seed,
                "metrics": r.metrics,
            }
            for r in merged.records
        ],
        "metrics": [
            {
                "arm": m.arm,
                "metric": m.metric,
                "mean": m.mean,
                "ci95": [m.ci95_low, m.ci95_high],
                "n_documents": m.n_documents,
            }
            for m in merged.metrics
        ],
        "bootstrap": {"samples": merged.bootstrap_samples, "unit": "per_document_records"},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
