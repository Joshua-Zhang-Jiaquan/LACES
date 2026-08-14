"""Packed-pickle FineWeb dataset and shard-contiguous sampler."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict, Unpack, final, runtime_checkable

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from .fineweb4096_manifest import read_pkl_manifest
from .fineweb4096_types import (
    LATENT_DIM_DEFAULT,
    MAX_LENGTH_DEFAULT,
    PAD_TOKEN_ID_DEFAULT,
    PKL_GLOB,
    TRAJECTORY_CHUNK_SIZE,
    VOCAB_SIZE,
    FineWebConfigurationError,
    FineWebDataFileError,
    FineWebDataFormatError,
    FineWebSample,
    PackedArray,
    PackedDatasetStats,
    PackedSample,
)

PackedValue: TypeAlias = PackedArray | list["PackedValue"] | dict[str, "PackedValue"]


@runtime_checkable
class _PackedShardDecoder(Protocol):
    def loads(self, payload: bytes) -> PackedValue: ...


def _packed_shard_decoder() -> _PackedShardDecoder:
    decoder = import_module("pickle")
    if isinstance(decoder, _PackedShardDecoder):
        return decoder
    message = "Python pickle decoder does not expose loads"
    raise FineWebDataFormatError(message)


class _PackedDatasetOptions(TypedDict, total=False):
    max_length: int
    latent_dim: int
    pad_token_id: int
    max_samples: int | None
    cache_shards: int
    world_size: int
    rank: int


@dataclass(frozen=True, slots=True)
class _ResolvedPackedOptions:
    max_length: int
    latent_dim: int
    pad_token_id: int
    max_samples: int | None
    cache_shards: int
    world_size: int
    rank: int


def _resolve_options(
    args: tuple[int | None, ...],
    options: _PackedDatasetOptions,
) -> _ResolvedPackedOptions:
    values: list[int | None] = [
        MAX_LENGTH_DEFAULT,
        LATENT_DIM_DEFAULT,
        PAD_TOKEN_ID_DEFAULT,
        None,
        1,
        1,
        0,
    ]
    for index, value in enumerate(args):
        if index >= len(values):
            message = "FineWebPackedPickleDataset accepts seven optional positional arguments"
            raise TypeError(message)
        values[index] = value
    max_length = options.get("max_length", values[0])
    latent_dim = options.get("latent_dim", values[1])
    pad_token_id = options.get("pad_token_id", values[2])
    cache_shards = options.get("cache_shards", values[4])
    world_size = options.get("world_size", values[5])
    rank = options.get("rank", values[6])
    if max_length is None or latent_dim is None or pad_token_id is None:
        message = "FineWebPackedPickleDataset integer options cannot be None"
        raise TypeError(message)
    if cache_shards is None or world_size is None or rank is None:
        message = "FineWebPackedPickleDataset worker options cannot be None"
        raise TypeError(message)
    return _ResolvedPackedOptions(
        max_length=max_length,
        latent_dim=latent_dim,
        pad_token_id=pad_token_id,
        max_samples=options.get("max_samples", values[3]),
        cache_shards=cache_shards,
        world_size=world_size,
        rank=rank,
    )


def list_pkl_shards(data_dir: Path) -> list[Path]:
    """Return packed-pickle shards in the historical lexical order."""
    return sorted(data_dir.glob(PKL_GLOB))


def _fit_length_1d(array: PackedArray, length: int, pad_value: int) -> PackedArray:
    """Truncate or right-pad an array while retaining its source dtype."""
    current_length = array.shape[-1]
    if current_length == length:
        return array
    if current_length > length:
        return array[:length]
    padded = np.full(length, pad_value, dtype=array.dtype)
    padded[:current_length] = array
    return padded


def _load_packed_shard(path: Path) -> list[PackedSample]:
    """Parse a packed shard at the pickle trust boundary into typed records."""
    raw_records = _packed_shard_decoder().loads(path.read_bytes())
    if not isinstance(raw_records, list):
        message = f"Packed shard {path} must contain a list"
        raise FineWebDataFormatError(message)
    records: list[PackedSample] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            message = f"Packed shard {path} contains a non-mapping record"
            raise FineWebDataFormatError(message)
        raw_ids: PackedValue | None = raw_record.get("input_ids")
        if not isinstance(raw_ids, np.ndarray):
            message = f"Packed shard {path} record lacks ndarray input_ids"
            raise FineWebDataFormatError(message)
        record: PackedSample = {"input_ids": raw_ids}
        raw_mask: PackedValue | None = raw_record.get("attention_mask")
        if isinstance(raw_mask, np.ndarray):
            record["attention_mask"] = raw_mask
        records.append(record)
    return records


@final
class FineWebPackedPickleDataset(Dataset[FineWebSample]):
    """Map-style FineWeb dataset with a lazy, bounded packed-shard cache."""

    def __init__(
        self,
        data_dir: str | Path,
        *args: int | None,
        **options: Unpack[_PackedDatasetOptions],
    ) -> None:
        """Accept the legacy packed-dataset positional and keyword options."""
        resolved = _resolve_options(args, options)
        super().__init__()
        if resolved.max_length <= 0 or resolved.max_length % TRAJECTORY_CHUNK_SIZE != 0:
            message = (
                f"max_length ({resolved.max_length}) must be a positive multiple of "
                f"trajectory_chunk_size ({TRAJECTORY_CHUNK_SIZE})."
            )
            raise FineWebConfigurationError(message)
        self.data_dir = Path(data_dir)
        self.max_length = resolved.max_length
        self.latent_dim = resolved.latent_dim
        self.pad_token_id = resolved.pad_token_id
        self.cache_shards = max(1, resolved.cache_shards)
        self.world_size = max(1, resolved.world_size)
        self.rank = resolved.rank

        shard_paths, shard_counts = self._resolve_shards()
        if not shard_paths:
            message = f"No {PKL_GLOB} shards under {self.data_dir}"
            raise FineWebDataFileError(message)
        self._shard_paths: list[Path] = []
        self._shard_counts: list[int] = []
        self._cumulative: list[int] = []
        total = 0
        remaining = resolved.max_samples
        for path, shard_count in zip(shard_paths, shard_counts, strict=False):
            count = shard_count
            if remaining is not None:
                if remaining <= 0:
                    break
                count = min(count, remaining)
                remaining -= count
            if count <= 0:
                continue
            self._shard_paths.append(path)
            self._shard_counts.append(count)
            total += count
            self._cumulative.append(total)
        self._total = total
        self._cache: dict[int, list[PackedSample]] = {}
        self._cache_order: list[int] = []

    def _resolve_shards(self) -> tuple[list[Path], list[int]]:
        """Use manifest counts when valid, otherwise count each shard once."""
        manifest_shards = read_pkl_manifest(self.data_dir)
        if manifest_shards is not None and manifest_shards:
            paths = [entry.path for entry in manifest_shards]
            counts = [entry.samples for entry in manifest_shards]
            if all(count > 0 for count in counts):
                return paths, counts
        paths = list_pkl_shards(self.data_dir)
        return paths, [len(_load_packed_shard(path)) for path in paths]

    def __len__(self) -> int:
        """Return the bounded total number of packed samples."""
        return self._total

    def shard_boundaries(self) -> list[int]:
        """Return cumulative shard ends for ``ShardContiguousSampler``."""
        return list(self._cumulative)

    def _load_shard(self, shard_index: int) -> list[PackedSample]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            return cached
        records = _load_packed_shard(self._shard_paths[shard_index])
        records = records[: self._shard_counts[shard_index]]
        self._cache[shard_index] = records
        self._cache_order.append(shard_index)
        while len(self._cache_order) > self.cache_shards:
            evicted = self._cache_order.pop(0)
            if evicted != shard_index:
                _ = self._cache.pop(evicted, None)
        return records

    @override
    def __getitem__(self, index: int) -> FineWebSample:
        """Return one packed sample with its existing truncation and mask policy."""
        if index < 0:
            index += self._total
        if index < 0 or index >= self._total:
            message = f"index {index} out of range for {self._total} samples"
            raise IndexError(message)
        shard_index = bisect.bisect_right(self._cumulative, index)
        previous_end = self._cumulative[shard_index - 1] if shard_index > 0 else 0
        sample = self._load_shard(shard_index)[index - previous_end]
        raw_ids = np.asarray(sample["input_ids"], dtype=np.int64)
        input_ids = _fit_length_1d(raw_ids, self.max_length, self.pad_token_id)
        _ = np.clip(input_ids, 0, VOCAB_SIZE - 1, out=input_ids)
        raw_mask = sample.get("attention_mask")
        if raw_mask is None:
            attention_mask = np.ones(self.max_length, dtype=np.float32)
            attention_mask[raw_ids.shape[-1] :] = 0.0
        else:
            attention_mask = _fit_length_1d(
                np.asarray(raw_mask).astype(np.float32), self.max_length, 0
            )
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(np.ascontiguousarray(attention_mask, dtype=np.float32)),
            "latent": torch.zeros(1, self.latent_dim, dtype=torch.float32),
        }

    def stats(self) -> PackedDatasetStats:
        """Return the historical packed-pickle diagnostic fields."""
        return {
            "format": "packed_pickle",
            "data_dir": str(self.data_dir),
            "n_shards": len(self._shard_paths),
            "n_samples": self._total,
            "max_length": self.max_length,
            "cache_shards": self.cache_shards,
            "samples_per_shard_first": self._shard_counts[0] if self._shard_counts else 0,
        }
