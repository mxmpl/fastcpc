import abc
import copy
from os import PathLike
from typing import Any, Callable, Iterator

import numpy as np
import polars as pl
import torch
import torchaudio
from more_itertools import chunked
from torch.utils.data import Dataset, Sampler

from .config import CONFIG

__all__ = ["AudioSequenceDataset", "SameSpeakerBatchSampler", "SameFileBatchSampler"]

Metadata = dict[str, str | int]
Transform = Callable[[torch.Tensor], torch.Tensor]


def split_in_windows(num_frames: int, windows_size: int, allow_overlap: bool = True) -> list[int]:
    """Return a list of indices to split a sequence of num_frames into windows of windows_size.
    If allow_overlap is True, the last window may have some overlap with the previous one."""
    if num_frames < windows_size:
        return []
    offset = list(range(0, num_frames, windows_size))
    if not num_frames % windows_size:
        return offset
    return offset[:-1] if not allow_overlap else offset[:-1] + [num_frames - windows_size]


def map_key_to_indices(data: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    mapping = {}
    for index, item in enumerate(data):
        attribute = item[key]
        if attribute in mapping:
            mapping[attribute].append(index)
        else:
            mapping[attribute] = [index]
    return mapping


def build_sequences(manifest_path: PathLike, window_size: int) -> list[Metadata]:
    df = pl.read_csv(manifest_path)
    assert {"fileid", "path", "num_frames", "speaker"} <= set(df.columns), "Invalid manifest file"
    assert len(df["fileid"].unique()) == len(df), "Duplicate file identifier"
    return (
        df.with_columns(
            pl.col("num_frames").map_elements(
                lambda num_frames: split_in_windows(num_frames, window_size, CONFIG.allow_overlap),
                return_dtype=list[int],
            ),
            pl.col("speaker").cast(pl.Utf8),
        )
        .rename({"num_frames": "offset"})
        .explode("offset")
        .to_dicts()
    )


class SameAttributeBatchSampler(Sampler[list[int]], abc.ABC):
    def __init__(self, sequences: list[Metadata], seed: int) -> None:
        self.seed = seed
        self.batch_size = CONFIG.batch_size
        self.attribute_to_indices = map_key_to_indices(sequences, self.attribute)
        self.generator = np.random.default_rng(seed)
        self.precomputed_indices = self._precompute_indices()

    @property
    @abc.abstractmethod
    def attribute(self) -> str:
        """Attribute by which to group sequences together"""

    def set_epoch(self, epoch: int) -> None:
        self.generator = np.random.default_rng(self.seed)
        for _ in range(epoch + 1):
            self.precomputed_indices = self._precompute_indices()

    def _precompute_indices(self) -> list[list[int]]:
        batch_indices = []
        for attribute_indices in copy.deepcopy(self.attribute_to_indices).values():
            self.generator.shuffle(attribute_indices)
            batch_same_attribute = list(chunked(attribute_indices, CONFIG.batch_size))
            if len(batch_same_attribute[-1]) < CONFIG.batch_size:
                batch_same_attribute.pop()
            batch_indices += batch_same_attribute
        self.generator.shuffle(batch_indices)
        return batch_indices

    @property
    def duration_in_seconds(self) -> float:
        return sum(len(batch) for batch in self.precomputed_indices) * CONFIG.window_size / CONFIG.sample_rate

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.precomputed_indices:
            yield batch
        self.precomputed_indices = self._precompute_indices()

    def __len__(self) -> int:
        return len(self.precomputed_indices)


class SameSpeakerBatchSampler(SameAttributeBatchSampler):
    @property
    def attribute(self) -> str:
        return "speaker"


class SameFileBatchSampler(SameAttributeBatchSampler):
    @property
    def attribute(self) -> str:
        return "fileid"


class AudioSequenceDataset(Dataset):
    def __init__(self, manifest_path: PathLike, transform: Transform | None = None) -> None:
        super().__init__()
        self.sequences = build_sequences(manifest_path, CONFIG.window_size)
        self.transform = transform

    @property
    def duration_in_seconds(self) -> float:
        return len(self) * CONFIG.window_size / CONFIG.sample_rate

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.sequences[index]
        waveform, sr = torchaudio.load(metadata["path"], metadata["offset"], CONFIG.window_size)
        assert sr == CONFIG.sample_rate
        if self.transform is not None:
            past, future = self.transform(waveform), waveform
            return past, future
        return waveform, waveform
