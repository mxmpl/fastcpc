"""Callbacks to check that training is going well."""

import torch


class SpikeToleranceError(Exception):
    """Wrong argumets for SpikeDetection."""

    def __init__(self) -> None:
        super().__init__("At least one of atol or rtol must be provided")


class SpikeDetection:
    """Detects spikes in the loss function."""

    def __init__(self, window: int = 10, atol: float | None = None, rtol: float | None = 2.0) -> None:
        if atol is None and rtol is None:
            raise SpikeToleranceError
        self.previous_losses = -torch.ones(window)
        self.window = window
        self.batch_idx = 0
        self.atol = atol
        self.rtol = rtol

    @property
    def in_warmup(self) -> bool:
        return self.batch_idx < self.window

    def _add_loss(self, value: float) -> None:
        self.previous_losses[self.batch_idx % self.window] = value
        self.batch_idx += 1

    def update(self, value: torch.Tensor) -> bool:
        if not torch.isfinite(value):
            return True

        last_value = self.previous_losses[self.batch_idx % self.window]
        if self.in_warmup or value < last_value:
            self._add_loss(value)
            return False

        running_mean = self.previous_losses.mean()
        check_atol = (self.atol is not None) and (abs(value - running_mean) < abs(self.atol))
        check_rtol = (self.rtol is not None) and (abs(value - running_mean) < abs(self.rtol * running_mean))
        if check_atol and check_rtol:
            self._add_loss(value)
            return False
        return True
