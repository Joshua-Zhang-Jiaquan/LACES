"""Strict contracts shared by the FineWeb-4096 data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

MAX_LENGTH_DEFAULT: Final = 4096
TRAJECTORY_CHUNK_SIZE: Final = 64
STRIDE_DEFAULT_FACTOR: Final = 2
PAD_TOKEN_ID_DEFAULT: Final = 0
LATENT_DIM_DEFAULT: Final = 32
VOCAB_SIZE: Final = 65536
CLUSTER_FINEWEB_DIR: Final = "/data/fineweb-10t/tokenized/seq2048"

BIN_NAME: Final = "data.bin"
IDX_NAME: Final = "data.idx"
BIN_DTYPE: Final = np.uint16
IDX_DTYPE: Final = np.int64
PKL_GLOB: Final = "*.pkl"
MANIFEST_GLOB: Final = "*manifest*.json"

TokenArray = NDArray[np.int64]
MaskArray = NDArray[np.float32]
PackedArray = NDArray[np.bool_ | np.float32 | np.int32 | np.int64 | np.uint16]


class DocumentIndex(Protocol):
    """Integer document starts read from the FineWeb index file."""

    @property
    def size(self) -> int:
        """Return the number of document starts."""
        ...

    def __len__(self) -> int:
        """Return the number of document starts."""
        ...

    def __getitem__(self, index: int) -> int:
        """Return one document start offset."""
        ...


class TokenMemmap(Protocol):
    """Read-only uint16 token backing store used by the bin dataset."""

    @property
    def size(self) -> int:
        """Return the total number of tokens."""
        ...

    def __getitem__(self, index: slice) -> NDArray[np.uint16]:
        """Return a contiguous token region."""
        ...


class FineWebSample(TypedDict):
    """One model-ready FineWeb training example."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    latent: torch.Tensor


class PackedSample(TypedDict):
    """One serialized packed-pickle record."""

    input_ids: PackedArray
    attention_mask: NotRequired[PackedArray]


class BinDatasetStats(TypedDict):
    """Observability fields emitted by the bin/idx dataset."""

    data_dir: str
    n_docs: int
    n_tokens_total: int
    n_windows: int
    n_full_windows: int
    n_padded_windows: int
    real_tokens: int
    padding_tokens: int
    padding_ratio: float
    max_length: int
    stride: int
    min_doc_tokens: int
    world_size: int
    rank: int


class PackedDatasetStats(TypedDict):
    """Observability fields emitted by the packed-pickle dataset."""

    format: str
    data_dir: str
    n_shards: int
    n_samples: int
    max_length: int
    cache_shards: int
    samples_per_shard_first: int


ConfigScalar = str | int | float | bool


@runtime_checkable
class ConfigLookup(Protocol):
    """The mapping capability required from trainer configuration sections."""

    def get(
        self,
        key: str,
        default: ConfigScalar | ConfigLookup | None = None,
    ) -> ConfigScalar | ConfigLookup | None:
        """Return a scalar option or nested configuration section."""
        ...


class FineWebTokenizer(Protocol):
    """Legacy tokenizer argument contract retained by the loader API."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text into tokenizer ids."""
        ...


class FineWebConfigurationError(ValueError):
    """Raised when FineWeb options violate the trajectory data contract."""


class FineWebDataFileError(FileNotFoundError):
    """Raised when the selected FineWeb shard files are absent."""


class FineWebDataFormatError(TypeError):
    """Raised when a packed FineWeb shard cannot supply tensor arrays."""


def config_section(config: ConfigLookup | None, name: str) -> ConfigLookup | None:
    """Return a nested config section only when it implements the lookup protocol."""
    if config is None:
        return None
    value = config.get(name)
    if isinstance(value, ConfigLookup):
        return value
    return None


def config_value(
    config: ConfigLookup | None,
    section_name: str,
    key: str,
) -> ConfigScalar | None:
    """Read one scalar config value without leaking untyped config objects."""
    section = config_section(config, section_name)
    if section is None:
        return None
    value = section.get(key)
    if isinstance(value, str | int | float | bool):
        return value
    return None


def parse_int_option(value: ConfigScalar | None, default: int) -> int:
    """Parse a scalar integer option, retaining the historical default on malformed text."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_float_option(value: ConfigScalar | None, default: float) -> float:
    """Parse a scalar floating-point option with the historical fallback behavior."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class BinDatasetOptions(TypedDict, total=False):
    """Keyword fields accepted by the import-compatible bin dataset API."""

    max_length: int
    stride: int | None
    min_doc_tokens: int | None
    pad_token_id: int
    latent_dim: int
    world_size: int
    rank: int
    dtype_check: bool


@dataclass(frozen=True, slots=True)
class ResolvedBinOptions:
    """Validated bin dataset settings consumed by dataset initialization."""

    max_length: int
    stride: int | None
    min_doc_tokens: int | None
    pad_token_id: int
    latent_dim: int
    world_size: int
    rank: int
    dtype_check: bool


def resolve_bin_options(
    args: tuple[int | None, ...],
    options: BinDatasetOptions,
) -> ResolvedBinOptions:
    """Parse retained positional and keyword constructor options."""
    values: list[int | None] = [
        MAX_LENGTH_DEFAULT,
        None,
        None,
        PAD_TOKEN_ID_DEFAULT,
        LATENT_DIM_DEFAULT,
        1,
        0,
        1,
    ]
    for index, value in enumerate(args):
        if index >= len(values):
            message = "FineWeb4096Dataset accepts eight optional positional arguments"
            raise TypeError(message)
        values[index] = value
    max_length = options.get("max_length", values[0])
    pad_token_id = options.get("pad_token_id", values[3])
    latent_dim = options.get("latent_dim", values[4])
    world_size = options.get("world_size", values[5])
    rank = options.get("rank", values[6])
    if max_length is None or pad_token_id is None or latent_dim is None:
        message = "FineWeb4096Dataset integer options cannot be None"
        raise TypeError(message)
    if world_size is None or rank is None:
        message = "FineWeb4096Dataset rank options cannot be None"
        raise TypeError(message)
    return ResolvedBinOptions(
        max_length=max_length,
        stride=options.get("stride", values[1]),
        min_doc_tokens=options.get("min_doc_tokens", values[2]),
        pad_token_id=pad_token_id,
        latent_dim=latent_dim,
        world_size=world_size,
        rank=rank,
        dtype_check=options.get("dtype_check", bool(values[7])),
    )
