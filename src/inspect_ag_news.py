"""Load and inspect AG News without loading a model or running experiments."""

import argparse
from collections import Counter
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "fancyzhx/ag_news"


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=nonnegative_int, default=3)
    parser.add_argument("--max-text-length", type=nonnegative_int, default=300)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "datasets"
    )
    parser.add_argument("--revision", help="Dataset repository commit SHA or revision")
    args = parser.parse_args()

    # Set before importing datasets so Hub downloads also stay outside Git.
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "huggingface"))
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        cache_dir=str(args.cache_dir.expanduser().resolve()),
        revision=args.revision,
    )
    print(f"Dataset: {DATASET_ID}")
    print(f"Requested revision: {args.revision or 'upstream default (unpinned)'}")
    for split_name, split in dataset.items():
        label_names = split.features["label"].names
        counts = Counter(split["label"])
        print(f"\nSplit: {split_name} | Rows: {len(split):,}")
        print(f"Features: {split.features}")
        print(f"Fingerprint: {split._fingerprint}")
        print("Labels and class counts:")
        for label_id, name in enumerate(label_names):
            print(f"  {label_id}: {name} = {counts[label_id]:,}")
        for index in range(min(args.num_examples, len(split))):
            row = split[index]
            text = row["text"]
            preview = text[:args.max_text_length]
            if len(text) > args.max_text_length:
                preview += "…"
            print(f"  Example {index}: label={row['label']} ({label_names[row['label']]})")
            print(f"    {preview!r}")


if __name__ == "__main__":
    main()
