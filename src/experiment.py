"""Configurable crossed AG News sensitivity study; defaults to a small pilot."""
import argparse
import csv
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from .ag_news import LABELS
from .design import ExperimentConfig, make_plan, sensitivities
from .inference import DATASET, QwenClassifier, load_data


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_paired_plan(plan, reference):
    """Fail before inference if paired selections or protocol-independent design differ."""
    for field in ("demonstration_sets", "permutations", "test_indices", "test_label_counts", "configurations"):
        if plan[field] != reference[field]:
            raise ValueError(f"Paired plan mismatch: {field}")
    for field in ("num_sets", "num_permutations", "num_test", "per_class", "seed", "selection",
                  "ordering", "test_sampling", "max_chars", "std_ddof", "dataset_revision"):
        if plan["config"][field] != reference["config"][field]:
            raise ValueError(f"Paired plan configuration mismatch: {field}")


def evaluate_plan(dataset, plan, predict, output_dir, verbose=False):
    """Injectable prediction function enables offline integration tests."""
    config = plan["config"]
    matrix = [[0.0] * config["num_permutations"] for _ in range(config["num_sets"])]
    configurations = []
    inference_seconds = 0.0
    with (output_dir / "predictions.jsonl").open("w") as handle, (output_dir / "configurations.csv").open("w", newline="") as csv_handle:
        fields = ("set_id", "permutation_id", "num_test", "num_correct", "accuracy", "parse_failures",
                  "canonical_demo_indices", "permutation", "ordered_demo_indices")
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for entry in plan["configurations"]:
            demos = [dataset["train"][i] for i in entry["ordered_demo_indices"]]
            correct, failures = 0, 0
            for test_index in plan["test_indices"]:
                row = dataset["test"][test_index]
                result = predict(demos, row["text"], max_chars=config["max_chars"])
                inference_seconds += result.get("seconds", 0.0)
                if verbose:
                    print(f"test[{test_index}] true={LABELS[row['label']]} predicted={result['predicted_name']} raw={result['raw_output']!r} parse_error={result['parse_error']!r} seconds={result.get('seconds', 0.0):.3f}", flush=True)
                is_correct = result["predicted_label"] == row["label"]
                correct += int(is_correct)
                failures += int(result["parse_error"] is not None)
                record = dict(set_id=entry["set_id"], permutation_id=entry["permutation_id"],
                              test_index=test_index, true_label=row["label"], true_name=LABELS[row["label"]],
                              correct=is_correct, **result)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            total = len(plan["test_indices"])
            row_result = dict(**entry, num_test=total, num_correct=correct,
                              accuracy=correct / total, parse_failures=failures)
            configurations.append(row_result)
            matrix[entry["set_id"]][entry["permutation_id"]] = row_result["accuracy"]
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row_result.items()})
            csv_handle.flush()
            print(f"set={entry['set_id']} permutation={entry['permutation_id']} accuracy={correct}/{total} parse_failures={failures}", flush=True)
    summary = dict(**sensitivities(matrix, config["std_ddof"]), configurations=configurations,
                   num_predictions=len(configurations) * len(plan["test_indices"]),
                   parse_failures=sum(row["parse_failures"] for row in configurations),
                   inference_seconds=inference_seconds)
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/pilot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot"))
    for name in ("num-sets", "num-permutations", "num-test", "per-class", "seed", "max-chars", "std-ddof"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("selection", "ordering", "test-sampling", "device", "model-id", "prompt-mode", "model-revision", "dataset-revision"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--paired-plan", type=Path, help="Require identical selections to a saved plan before inference")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Save selections without model loading or inference")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text())
    for key in asdict(ExperimentConfig()):
        override = getattr(args, key)
        if override is not None:
            values[key] = override
    config = ExperimentConfig(**values).validate()
    # Exclusive directory creation protects previous runs from accidental replacement.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="starting", started_utc=datetime.now(timezone.utc).isoformat(), config=asdict(config),
                    methodology="Section 3-inspired adaptation; see report/methodology.md")
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        dataset, dataset_sha = load_data(config.dataset_revision)
        if dataset["train"].features["label"].names != list(LABELS):
            raise ValueError("Dataset labels differ from expected AG News mapping")
        plan = make_plan(dataset, config)
        if args.paired_plan:
            reference_bytes = args.paired_plan.read_bytes()
            verify_paired_plan(plan, json.loads(reference_bytes))
            manifest["paired_reference"] = dict(path=str(args.paired_plan),
                                                 sha256=hashlib.sha256(reference_bytes).hexdigest(),
                                                 selections_verified_identical=True)
        write_json(args.output_dir / "plan.json", plan)
        manifest.update(dataset=DATASET, dataset_revision=dataset_sha,
                        dataset_fingerprints={key: split._fingerprint for key, split in dataset.items()})
        if args.plan_only:
            manifest["status"] = "planned"
        else:
            classifier = QwenClassifier(config.seed, config.device, config.model_revision, config.model_id, config.prompt_mode)
            manifest.update(classifier.metadata())
            manifest["status"] = "running"
            write_json(args.output_dir / "manifest.json", manifest)
            packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
            (args.output_dir / "environment.txt").write_text("\n".join(packages) + "\n")
            snapshot = args.output_dir / "source"
            root = Path(__file__).resolve().parents[1]
            for folder in ("src", "tests", "experiments", "report"):
                for path in (root / folder).rglob("*"):
                    if path.is_file() and path.suffix in (".py", ".json", ".md"):
                        target = snapshot / path.relative_to(root)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(path.read_bytes())
            for name in ("requirements.txt", "README.md"):
                (snapshot / name).write_bytes((root / name).read_bytes())
            summary = evaluate_plan(dataset, plan, classifier.predict, args.output_dir, verbose=args.verbose)
            print(json.dumps({k: summary[k] for k in ("order_sensitivity", "selection_sensitivity", "selection_over_order")}, indent=2))
            print(f"Accuracy: {summary['mean_accuracy']:.1%}; parsing failures: {summary['parse_failures']}; inference seconds: {summary['inference_seconds']:.3f}", flush=True)
            manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
