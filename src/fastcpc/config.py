"""Global module configuration."""

import dataclasses
import json
import os
from pathlib import Path
from typing import Literal

__all__ = ["CONFIG"]


@dataclasses.dataclass(frozen=True)
class Configuration:
    random_seed: int = 1234
    sample_rate: int = 16000
    hidden_size: int = 256
    num_lstm_layers: int = 2
    num_predicts: int = 12
    negative_samples: int = 128
    fully_connected_dim: int = 2048
    dropout: float = 0.1
    num_heads: int = 8
    learning_rate: float = 2e-4
    scheduler_iters: int = 10
    window_size: int = 20480
    batch_size: int = 8
    num_epochs: int = 100
    num_workers: int = 4
    log_interval: int = 1000
    max_grad_norm: float = 1000
    weight_decay: float = 0.01
    allow_overlap: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"

    @staticmethod
    def load_default() -> "Configuration":
        if "FASTCPC_CONFIG" in os.environ:
            with Path(os.environ["FASTCPC_CONFIG"]).open() as f:
                return Configuration(**json.load(f))
        return Configuration()


CONFIG = Configuration.load_default()
