import csv
import warnings
from pathlib import Path
from typing import Iterable

import torch

from .config import CONFIG


def previous_checkpoint(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("epoch_*/"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: int(x.stem.removeprefix("epoch_")))


def manifest_duration_in_seconds(manifest_path: Path) -> float:
    with open(manifest_path, "r") as file:
        return sum(int(row["num_frames"]) for row in csv.DictReader(file)) / CONFIG.sample_rate


def params_norm(
    params: torch.Tensor | Iterable[torch.Tensor], norm_type: float = 2.0, error_if_nonfinite: bool = False
) -> float:
    if isinstance(params, torch.Tensor):
        params = [params]
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.vector_norm(p, norm_type) for p in params]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise ValueError("Parameters contain non-finite values")
    return total_norm.item()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log(self) -> float:
        avg = self.avg
        self.reset()
        return avg


class Records:
    """Record average meters for multiple values."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.meters = {name: AverageMeter() for name in names}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.meters[name]

    def log(self) -> dict[str, float]:
        return {name: meter.log() for name, meter in self.meters.items()}


def assert_compatibility(window_size: int, num_predicts: int) -> None:
    warning = ""
    if window_size != CONFIG.window_size:
        warning += (
            f"The model is created with window_size={window_size} instead of {CONFIG.window_size}."
            " -> Dataset and samplers provided in this library will fail.\n"
        )
    if num_predicts != CONFIG.num_predicts:
        warning += (
            f"The model is created with num_predicts={num_predicts} instead of {CONFIG.num_predicts}."
            " -> CPCCriterion will fail.\n"
        )
    if warning:
        warnings.warn(
            "Incompatible parameters\n"
            + warning
            + " Either use the default value(s), or change the global configuration by setting"
            " `FASTCPC_CONFIG` to your JSON config file."
        )
