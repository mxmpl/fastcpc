"""Extract features."""

import csv
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from .config import CONFIG
from .model import CPC


@torch.inference_mode()
def extract_features(model_path: str, manifest: str, output: str) -> None:
    """Extract features using a pretrained model.

    The expected manifest file is a TSV file in the following format:
    ```
    <root>
    <filename1> <number_of_samples1>
    <filename2> <number_of_samples2>
    ...
    ```
    WARNING: This is not the same format as the one used in the training script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPC().eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.compile(model)

    with Path(manifest).open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        src = Path(next(reader)[0])
        filenames = [Path(row[0]) for row in reader]
    dest = Path(output).resolve()
    for filename in tqdm(filenames):
        path = src / filename
        waveform, sr = torchaudio.load(str(path))
        if sr != CONFIG.sample_rate:
            raise ValueError(filename)
        features = model.extract_features(waveform.to(device).unsqueeze(0)).squeeze().cpu()
        dest_path = dest / filename.with_suffix(".pt")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, dest_path)
