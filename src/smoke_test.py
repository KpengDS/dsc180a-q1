"""Run exactly five 8-shot AG News predictions; no training or order sweep."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform

from .ag_news import LABELS, select_examples
from .inference import ROOT, DATASET, QwenClassifier, load_data, setup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=180)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--check-hardware", action="store_true", help="No Hub access or weights")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke_test.json")
    args = parser.parse_args()
    setup_environment()
    import torch
    print(f"Python={platform.python_version()} torch={torch.__version__} MPS={torch.backends.mps.is_available()}", flush=True)
    if args.check_hardware:
        return
    dataset, dataset_sha = load_data(args.dataset_revision)
    demo_ids, test_ids = select_examples(dataset, args.seed)
    demos = [dataset["train"][i] for i in demo_ids]
    assert Counter(row["label"] for row in demos) == {i: 2 for i in range(4)}
    classifier = QwenClassifier(args.seed, args.device, args.model_revision)
    records = []
    for index in test_ids:
        row = dataset["test"][index]
        record = dict(test_index=index, true_label=row["label"], true_name=LABELS[row["label"]],
                      **classifier.predict(demos, row["text"]))
        records.append(record)
        print(f"test[{index}] true={record['true_name']} predicted={record['predicted_name']} raw={record['raw_output']!r}", flush=True)
    accuracy = sum(r["true_label"] == r["predicted_label"] for r in records) / 5
    result = dict(**classifier.metadata(), dataset=DATASET, dataset_revision=dataset_sha,
                  seed=args.seed, demo_indices=demo_ids, test_indices=test_ids,
                  article_max_chars=1000, accuracy=accuracy, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Accuracy: {accuracy:.1%} ({int(round(accuracy * 5))}/5). Saved {args.output}")


if __name__ == "__main__":
    main()
