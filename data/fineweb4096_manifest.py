"""Typed manifest parsing for packed FineWeb shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from .fineweb4096_types import MANIFEST_GLOB

if TYPE_CHECKING:
    from pathlib import Path

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class ManifestShard:
    """A usable packed-pickle shard entry from a manifest."""

    path: Path
    samples: int


def read_pkl_manifest(data_dir: Path) -> list[ManifestShard] | None:
    """Return viable manifest entries without opening shard payloads."""
    for manifest_path in sorted(data_dir.glob(MANIFEST_GLOB)):
        try:
            raw_manifest: JsonValue = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(raw_manifest, dict):
            continue
        raw_shards: JsonValue | None = raw_manifest.get("shards")
        if not isinstance(raw_shards, list):
            continue
        shards: list[ManifestShard] = []
        for raw_entry in raw_shards:
            if not isinstance(raw_entry, dict):
                continue
            raw_path: JsonValue | None = raw_entry.get("path")
            raw_samples: JsonValue = raw_entry.get("samples", 0)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            if not isinstance(raw_samples, int):
                continue
            shard_path = data_dir / raw_path
            if shard_path.exists():
                shards.append(ManifestShard(path=shard_path, samples=raw_samples))
        return shards
    return None
