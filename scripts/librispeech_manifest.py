"""Small utility to build Librispeech manifest files."""

import argparse
import csv
from pathlib import Path

import torchaudio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small utility to build Librispeech manifest files")
    parser.add_argument("root")
    parser.add_argument("manifest")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    with Path(args.manifest).open("w", newline="") as csvfile:
        fieldnames = ["fileid", "path", "num_frames", "speaker"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for speaker in root.glob("*/"):
            for chapter in speaker.glob("*/"):
                for file in chapter.glob("*.flac"):
                    fileid = file.stem
                    num_frames = torchaudio.info(str(file)).num_frames
                    writer.writerow(
                        {
                            "fileid": fileid,
                            "path": str(file),
                            "num_frames": num_frames,
                            "speaker": speaker.name,
                        }
                    )
