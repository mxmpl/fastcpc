# Faster and simpler Contrastive Predictive Coding for audio
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository provides a reimplementation of Constrastive Predictive Coding (CPC) for audio.
The main goal is to provide a simpler and faster implementation to facilitate iteration.
The original code can be found here: https://github.com/bootphon/CPC3.

## Installation

Clone this repository and install `fastcpc` via pip:
```bash
git clone https://github.com/mxmpl/fastcpc
cd fastcpc
pip install -e .
```

## Usage

After installing the package, you have access to the `fastcpc` command with two subcommands:
- `fastcpc train`: launch training.
  - If you are on a SLURM cluster, see the [docs](./slurm/README.md) to use it via the [launcher.slurm](./slurm/launcher.slurm) script.
  - Or launch it with `torchrun`.
- `fastcpc extract`: extract features with a pretrained model. First step before ABX evaluation.

To load a pretrained model:
```python
from fastcpc.model import CPC

model = CPC.from_pretrained("path/to/model")
```

A simple script to create the manifest files for training on LibriSpeech can be found in [scripts/librispeech_manifest.py](./scripts/librispeech_manifest.py). Adapt it to your needs.

---

## Know issues

On some systems, if you install Pytorch via conda, data augmentation will fail when
applying pitch transformation. This is likely related to [this issue](https://github.com/pytorch/audio/issues/1021)
and due to multiprocessing incompatibilities between torchaudio and sox.
But it seems to be working fine when installing Pytorch via pip.

TL;DR: install everything with pip.
