import numpy as np
import torch
from torchaudio import sox_effects

from .config import CONFIG

__all__ = ["PitchReverbAugment"]


class PitchReverbAugment:
    max_pitch_shift = 300
    max_reverberance = 100
    max_room_scale = 100
    high_frequency_damping = 100

    def __init__(self, random_seed: int = 0) -> None:
        self.generator = np.random.default_rng(random_seed)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        pitch_shift = self.generator.integers(-self.max_pitch_shift, self.max_pitch_shift)
        reverberance = self.generator.integers(0, self.max_reverberance)
        room_scale = self.generator.integers(0, self.max_room_scale)
        effects = [
            ["pitch", str(pitch_shift)],
            ["rate", str(CONFIG.sample_rate)],
            ["reverb", str(reverberance), str(self.high_frequency_damping), str(room_scale)],
            ["channels", "1"],
            ["dither"],
        ]
        transformed = sox_effects.apply_effects_tensor(waveform, CONFIG.sample_rate, effects)[0]
        if transformed.shape == waveform.shape:
            return transformed
        if transformed.shape[1] == waveform.shape[1] + 1:
            return transformed[:, :-1]
        if transformed.shape[1] == waveform.shape[1] - 1:
            return torch.nn.functional.pad(transformed, (0, 1))
        raise ValueError("Shape after applying effects differs by more than one frame")
