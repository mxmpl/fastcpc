"""Main entry point."""

import argparse
from importlib.metadata import version


def parser_train(subparsers) -> None:  # noqa: ANN001
    parser = subparsers.add_parser("train", description="Train a CPC model")
    parser.add_argument("run", type=str, help="name of the current run")
    parser.add_argument("workdir", type=str, help="path to the working directory")
    parser.add_argument("train", type=str, help="path to the train manifest file")
    parser.add_argument("val", type=str, help="path to the validation manifest file")
    parser.add_argument("--project", type=str, default="cpc", help="name of the project (default: %(default)s)")


def parser_extract_features(subparsers) -> None:  # noqa: ANN001
    parser = subparsers.add_parser("extract", description="Extract features with a pretrained CPC model")
    parser.add_argument("model", type=str, help="path to the model checkpoint")
    parser.add_argument("manifest", type=str, help="path to the manifest file")
    parser.add_argument("output", type=str, help="path to the output directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CPC fast")
    parser.add_argument("-V", "--version", action="version", version=version("fastcpc"), help="show the version")
    subparsers = parser.add_subparsers(dest="command", help="command to run")
    parser_train(subparsers)
    parser_extract_features(subparsers)
    args = parser.parse_args()

    if args.command == "train":
        from .train import train

        train(args.run, args.workdir, args.train, args.val, args.project)
    elif args.command == "extract":
        from .features import extract_features

        extract_features(args.model, args.manifest, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
