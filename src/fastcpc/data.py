"""Load audio data."""

import abc
import copy
from collections.abc import Callable, Iterator
from os import PathLike
from typing import Any

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


def split_in_windows(num_frames: int, window_size: int, *, allow_overlap: bool = True) -> list[int]:
    """Split a number of frames in windows of the same size.

    Return a list of indices to split a sequence of num_frames into windows of windows_size.
    If allow_overlap is True, the last window may have some overlap with the previous one.
    """
    if num_frames < window_size:
        return []
    offset = list(range(0, num_frames, window_size))
    if not num_frames % window_size:
        return offset
    return offset[:-1] if not allow_overlap else offset[:-1] + [num_frames - window_size]


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
    manifest = pl.read_csv(manifest_path)
    if not {"fileid", "path", "num_frames", "speaker"} <= set(manifest.columns):
        raise ValueError(manifest_path)
    if len(manifest["fileid"].unique()) != len(manifest):
        raise ValueError(manifest_path)
    return (
        manifest.with_columns(
            pl.col("num_frames").map_elements(
                lambda num_frames: split_in_windows(num_frames, window_size, allow_overlap=CONFIG.allow_overlap),
                return_dtype=list[int],
            ),
            pl.col("speaker").cast(pl.Utf8),
        )
        .rename({"num_frames": "offset"})
        .explode("offset")
        .drop_nulls()
        .to_dicts()
    )


class SameAttributeBatchSampler(Sampler[list[int]], abc.ABC):
    """Sample batches such that in a batch all elements share the specified attribute."""

    def __init__(self, sequences: list[Metadata], seed: int) -> None:
        self.seed = seed
        self.batch_size = CONFIG.batch_size
        self.attribute_to_indices = map_key_to_indices(sequences, self.attribute)
        self.generator = np.random.default_rng(seed)
        self.precomputed_indices = self._precompute_indices()

    @property
    @abc.abstractmethod
    def attribute(self) -> str:
        """Attribute by which to group sequences together."""

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
        yield from self.precomputed_indices
        self.precomputed_indices = self._precompute_indices()

    def __len__(self) -> int:
        return len(self.precomputed_indices)


class SameSpeakerBatchSampler(SameAttributeBatchSampler):
    """Sample batches such that in a batch all samples are from the same speaker."""

    @property
    def attribute(self) -> str:
        return "speaker"


class SameFileBatchSampler(SameAttributeBatchSampler):
    """Sample batches such that in a batch all samples are from the same audio file."""

    @property
    def attribute(self) -> str:
        return "fileid"


class AudioSequenceDataset(Dataset):
    """Dataset to load chunks of audio files."""

    def __init__(self, manifest_path: PathLike, transform: Transform | None = None) -> None:
        super().__init__()
        self.sequences = build_sequences(manifest_path, CONFIG.window_size)
        self.transform = transform

    @property
    def duration_in_seconds(self) -> float:
        return len(self) * CONFIG.window_size / CONFIG.sample_rate

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.sequences[index]
        waveform, sr = torchaudio.load(metadata["path"], metadata["offset"], CONFIG.window_size)
        if sr != CONFIG.sample_rate:
            raise ValueError(metadata)
        if self.transform is not None:
            past, future = self.transform(waveform), waveform
            return past, future
        return waveform, waveform, metadata
