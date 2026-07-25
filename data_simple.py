# File: data_simple.py
import torch
import numpy as np
import pickle
import bisect
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Optional, Union


class DirectFileDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that reads directly from tokenized .npz and latent .npy files."""

    def __init__(
        self,
        token_dir: str,
        latent_dir: str,
        max_samples: Optional[int] = None,
        use_external_latents: bool = True,
        verbose: bool = True,
    ):
        token_dir_parts = [p.strip() for p in str(token_dir).split(",") if p.strip()]
        self.token_dirs = [Path(p) for p in token_dir_parts]
        self.token_dir = self.token_dirs[0]
        self.latent_dir = Path(latent_dir) if latent_dir else None
        self.use_external_latents = use_external_latents

        collected = []
        for d in self.token_dirs:
            collected.extend(d.glob("*.npz"))
        self.token_files = sorted(collected)
        self._pkl_data = None
        self._pkl_paths = []
        self._pkl_lengths = []
        self._pkl_cumulative = []
        self._pkl_cache_idx = None
        self._pkl_cache_data = None

        # Support .pkl fallback (all samples in one file) when no .npz found
        if len(self.token_files) == 0:
            pkl_files = sorted(list(self.token_dir.glob("*.pkl")))
            if pkl_files:
                if verbose:
                    print(f"Indexing {len(pkl_files)} pre-tokenized pkl shard(s) from {self.token_dir}...")
                remaining = max_samples
                total = 0
                for pkl_path in pkl_files:
                    with open(pkl_path, "rb") as f:
                        shard = pickle.load(f)
                    shard_len = len(shard)
                    if remaining is not None:
                        if remaining <= 0:
                            break
                        shard_len = min(shard_len, remaining)
                        remaining -= shard_len
                    if shard_len <= 0:
                        continue
                    self._pkl_paths.append(pkl_path)
                    self._pkl_lengths.append(shard_len)
                    total += shard_len
                    self._pkl_cumulative.append(total)
                if verbose:
                    print(f"Indexed {total} tokenized samples from pkl shard(s)")
                if self._pkl_paths:
                    first = self._load_pkl_shard(0)[0]
                    self._seq_len = first["input_ids"].shape[-1]
                else:
                    self._seq_len = 512
                self._latent_dim = 32

        if max_samples:
            self.token_files = self.token_files[:max_samples]

        valid_files = []
        missing_latents = 0

        if self.use_external_latents and self.latent_dir:
            for token_file in self.token_files:
                stripped = token_file.stem.replace("_tokens", "")
                candidates = [
                    self.latent_dir / f"{stripped}.npy",
                    self.latent_dir / f"{token_file.stem}.npy",
                ]
                latent_file = next((c for c in candidates if c.exists()), None)
                if latent_file is not None:
                    valid_files.append((token_file, latent_file))
                else:
                    missing_latents += 1
                    if verbose and missing_latents <= 5:
                        print(f"Warning: Missing latent for {token_file.name}")
        else:
            for token_file in self.token_files:
                valid_files.append((token_file, None))

        self.file_pairs = valid_files

        # Infer shapes from first file
        self._seq_len = 512
        self._latent_dim = 32
        if self.file_pairs:
            try:
                token_f, latent_f = self.file_pairs[0]
                token_data = np.load(token_f)
                if "input_ids" in token_data:
                    ids = token_data["input_ids"]
                    if hasattr(ids, "shape") and len(ids.shape) >= 1:
                        self._seq_len = int(ids.shape[-1])
                if latent_f is not None:
                    latent = np.load(latent_f)
                    if hasattr(latent, "shape") and len(latent.shape) >= 1:
                        self._latent_dim = int(latent.shape[-1])
            except Exception:
                pass

        if verbose:
            if self._pkl_data is not None:
                print(f"Found {len(self._pkl_data)} tokenized samples (pkl)")
            elif self._pkl_paths:
                print(f"Found {self._pkl_cumulative[-1]} tokenized samples in {len(self._pkl_paths)} pkl shard(s)")
            else:
                print(f"Found {len(self.token_files)} token files")
                print(f"Found {len(self.file_pairs)} valid file pairs")
                if missing_latents:
                    print(f"Missing latents: {missing_latents}")

    def _sample_get(self, sample, key: str, default=None):
        if hasattr(sample, "get"):
            return sample.get(key, default)
        if hasattr(sample, "files") and key in sample.files:
            return sample[key]
        if hasattr(sample, "__contains__") and key in sample:
            return sample[key]
        return default

    def _optional_response_fields(self, sample, seq_len: int):
        fields = {}
        response_mask = self._sample_get(sample, "response_mask")
        if response_mask is not None:
            response_mask = np.asarray(response_mask).astype(bool)
            if response_mask.ndim == 0:
                response_mask = np.array([bool(response_mask)])
            fields["response_mask"] = torch.tensor(response_mask, dtype=torch.float32)

        prompt_lengths = self._sample_get(
            sample,
            "prompt_lengths",
            self._sample_get(sample, "prompt_length"),
        )
        if prompt_lengths is not None:
            prompt_lengths_arr = np.asarray(prompt_lengths).reshape(-1)
            prompt_len = max(0, min(int(prompt_lengths_arr[0]), seq_len))
            fields["prompt_lengths"] = torch.tensor(prompt_len, dtype=torch.int64)
            if "response_mask" not in fields:
                mask = torch.zeros(seq_len, dtype=torch.float32)
                if prompt_len < seq_len:
                    mask[prompt_len:] = 1.0
                fields["response_mask"] = mask
        return fields

    def __len__(self):
        if self._pkl_data is not None:
            return len(self._pkl_data)
        if self._pkl_paths:
            return self._pkl_cumulative[-1]
        return len(self.file_pairs)

    def _load_pkl_shard(self, shard_idx):
        if self._pkl_cache_idx == shard_idx and self._pkl_cache_data is not None:
            return self._pkl_cache_data
        with open(self._pkl_paths[shard_idx], "rb") as f:
            data = pickle.load(f)
        limit = self._pkl_lengths[shard_idx]
        if len(data) > limit:
            data = data[:limit]
        self._pkl_cache_idx = shard_idx
        self._pkl_cache_data = data
        return data

    def __getitem__(self, idx):
        if self._pkl_data is not None:
            sample = self._pkl_data[idx]
            input_ids_tensor = torch.tensor(sample["input_ids"].astype(np.int64))
            attention_mask_tensor = torch.tensor(sample["attention_mask"].astype(bool), dtype=torch.float32)
            item = {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "latent": torch.zeros(1, self._latent_dim, dtype=torch.float32),
            }
            item.update(self._optional_response_fields(sample, input_ids_tensor.shape[-1]))
            return item
        if self._pkl_paths:
            shard_idx = bisect.bisect_right(self._pkl_cumulative, idx)
            prev = self._pkl_cumulative[shard_idx - 1] if shard_idx > 0 else 0
            sample = self._load_pkl_shard(shard_idx)[idx - prev]
            input_ids_tensor = torch.tensor(sample["input_ids"].astype(np.int64))
            attention_mask_tensor = torch.tensor(sample["attention_mask"].astype(bool), dtype=torch.float32)
            item = {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "latent": torch.zeros(1, self._latent_dim, dtype=torch.float32),
            }
            item.update(self._optional_response_fields(sample, input_ids_tensor.shape[-1]))
            return item

        last_error = None
        for offset in range(min(16, len(self.file_pairs))):
            token_file, latent_file = self.file_pairs[(idx + offset) % len(self.file_pairs)]
            try:
                token_data = np.load(token_file)
                input_ids = token_data["input_ids"].astype(np.int32)
                attention_mask = token_data["attention_mask"].astype(bool)

                if input_ids.ndim == 0:
                    input_ids = np.array([input_ids])
                if attention_mask.ndim == 0:
                    attention_mask = np.array([attention_mask])

                input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64)
                attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.float32)

                if self.use_external_latents and latent_file is not None:
                    latent = np.load(latent_file)
                    latent_tensor = torch.tensor(latent, dtype=torch.float32)
                    # Cross-source trajectory latents are [H, latent_dim] (e.g. 13.3B-S0 dump,
                    # H=16). Keep 2D [H, D] as-is; only lift bare 1D vectors to [1, D]. Do NOT
                    # slice to [0:1] (that would discard the trajectory and break Direction-2).
                    if latent_tensor.dim() == 1:
                        latent_tensor = latent_tensor.unsqueeze(0)
                else:
                    latent_tensor = torch.zeros(1, self._latent_dim, dtype=torch.float32)

                item = {
                    "input_ids": input_ids_tensor,
                    "attention_mask": attention_mask_tensor,
                    "latent": latent_tensor,
                }
                item.update(self._optional_response_fields(token_data, input_ids_tensor.shape[-1]))
                return item

            except Exception as e:
                last_error = e
                print(f"Error loading files {token_file}, {latent_file}: {e}; trying next sample")

        raise RuntimeError(f"Failed to load a valid sample near index {idx}: {last_error}")

    def _create_empty_sample(self):
        return {
            "input_ids": torch.zeros(self._seq_len, dtype=torch.long),
            "attention_mask": torch.zeros(self._seq_len, dtype=torch.float),
            "latent": torch.zeros(1, self._latent_dim, dtype=torch.float),
        }


def get_simple_dataloaders(config, tokenizer=None):
    data_cfg = config.data

    dataset = DirectFileDataset(
        token_dir=data_cfg.token_dir,
        latent_dir=data_cfg.latent_dir,
        max_samples=data_cfg.get("max_samples", None),
        use_external_latents=data_cfg.get("use_external_latents", True),
        verbose=True,
    )

    if len(dataset) == 0:
        raise ValueError("No usable token/latent pairs found")

    val_ratio = float(data_cfg.get("val_ratio", 0.05))
    if val_ratio <= 0.0:
        val_size = 0
        train_size = len(dataset)
    else:
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(config.training.seed)),
    )

    latent_dim = int(config.model.get("latent_dim", 32))

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        latents = []
        for item in batch:
            latent = item["latent"]
            if latent is None:
                latents.append(torch.zeros(1, latent_dim, dtype=torch.float32))
                continue
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            elif latent.dim() == 2 and latent.shape[0] != 1:
                latent = latent[:1]
            latents.append(latent)

        latent_tensor = torch.cat(latents, dim=0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "latent": latent_tensor,
        }

        if any("response_mask" in item for item in batch):
            masks = []
            for item in batch:
                mask = item.get("response_mask")
                if mask is None:
                    mask = torch.zeros_like(item["attention_mask"], dtype=torch.float32)
                masks.append(mask.to(dtype=torch.float32))
            out["response_mask"] = torch.stack(masks)
        if any("prompt_lengths" in item for item in batch):
            prompt_lengths = []
            for item in batch:
                pl = item.get("prompt_lengths")
                if pl is None:
                    pl = torch.tensor(0, dtype=torch.int64)
                prompt_lengths.append(pl)
            out["prompt_lengths"] = torch.stack(prompt_lengths).to(dtype=torch.int64)

        return out

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=int(config.training.seed),
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training.train_batch_size),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.training.eval_batch_size),
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
