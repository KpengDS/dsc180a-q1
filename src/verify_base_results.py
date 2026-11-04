"""Offline audit of every saved Base prediction, prompt, and aggregate metric.

This does not run inference, retry predictions, or modify evaluation outputs.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from .ag_news import LABELS
from .base_protocol import POLICY, build_base_prompt, completion_line, parse_base_label
from .inference import ROOT, setup_environment


def independent_metrics(matrix):
    def std(values):
        values = list(values)
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    order_by_set = [std(row) for row in matrix]
    selection_by_order = [std(column) for column in zip(*matrix)]
    order = sum(order_by_set) / len(order_by_set)
    selection = sum(selection_by_order) / len(selection_by_order)
    return dict(mean_accuracy=sum(map(sum, matrix)) / sum(map(len, matrix)),
                order_sensitivity=order, selection_sensitivity=selection,
                selection_over_order=selection / order if order else None,
                order_std_by_set=order_by_set, selection_std_by_permutation=selection_by_order)


def verify(run, previous_hashes=None, baseline=None):
    setup_environment()
    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset
    from transformers import AutoTokenizer
    manifest = json.loads((run / "manifest.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    config = plan["config"]
    assert manifest["status"] == "complete"
    assert manifest["config"] == config
    assert manifest["model"] == config["model_id"]
    assert manifest["model_revision"] == config["model_revision"]
    assert manifest["dataset_revision"] == config["dataset_revision"]
    assert manifest["parser_policy"] == POLICY
    assert manifest["prompt_mode"] == "completion" and manifest["chat_template"] is None
    assert manifest["dtype"] == "torch.bfloat16" and manifest["device"] == "mps"
    assert manifest["generation"]["do_sample"] is False and manifest["generation"]["num_beams"] == 1
    assert manifest["generation"]["max_new_tokens"] == 12 and config["std_ddof"] == 0
    dataset = load_dataset(manifest["dataset"], revision=manifest["dataset_revision"],
                           cache_dir=str(ROOT / "data/datasets"))
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["model_revision"],
                                              cache_dir=str(ROOT / "models"), local_files_only=True)
    for split, fingerprint in manifest["dataset_fingerprints"].items():
        assert dataset[split]._fingerprint == fingerprint
    assert len(set(plan["test_indices"])) == config["num_test"]
    assert len({tuple(sorted(s)) for s in plan["demonstration_sets"]}) == config["num_sets"]
    assert len({tuple(p) for p in plan["permutations"]}) == config["num_permutations"]
    for indices in plan["demonstration_sets"]:
        assert len(set(indices)) == 8
        assert Counter(dataset["train"][i]["label"] for i in indices) == {c: 2 for c in range(4)}
    for permutation in plan["permutations"]:
        assert sorted(permutation) == list(range(8))
    assert plan["test_label_counts"] == {
        name: sum(dataset["test"][i]["label"] == c for i in plan["test_indices"])
        for c, name in enumerate(LABELS)
    }
    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    expected_count = config["num_sets"] * config["num_permutations"] * config["num_test"]
    assert len(records) == expected_count
    lookup = {(r["set_id"], r["permutation_id"], r["test_index"]): r for r in records}
    assert len(lookup) == expected_count
    matrix = [[0.0] * config["num_permutations"] for _ in range(config["num_sets"])]
    csv_records = list(csv.DictReader((run / "configurations.csv").open()))
    csv_lookup = {(int(row["set_id"]), int(row["permutation_id"])): row for row in csv_records}
    assert len(csv_lookup) == len(plan["configurations"]) == config["num_sets"] * config["num_permutations"]
    eos = manifest["generation"]["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    failures = []
    expected_keys = set()
    for entry in plan["configurations"]:
        i, j = entry["set_id"], entry["permutation_id"]
        assert entry["permutation"] == plan["permutations"][j]
        assert entry["canonical_demo_indices"] == plan["demonstration_sets"][i]
        assert entry["ordered_demo_indices"] == [entry["canonical_demo_indices"][p] for p in entry["permutation"]]
        demos = [dataset["train"][index] for index in entry["ordered_demo_indices"]]
        correct = invalid = 0
        for test_index in plan["test_indices"]:
            key = (i, j, test_index)
            expected_keys.add(key)
            row = lookup[key]
            article = dataset["test"][test_index]
            assert row["true_label"] == article["label"] and row["true_name"] == LABELS[article["label"]]
            assert row["rendered_prompt"] == build_base_prompt(demos, article["text"], config["max_chars"])
            assert row["input_tokens"] == len(tokenizer(row["rendered_prompt"], add_special_tokens=False)["input_ids"])
            assert row["raw_output"] == tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True)
            assert 0 < len(row["generated_token_ids"]) <= 12
            reason = ("newline" if "\n" in row["raw_output"] else "eos"
                      if row["generated_token_ids"][-1] in eos_ids else "token_cap")
            assert row["termination_reason"] == reason
            assert row["completion_text"] == completion_line(row["raw_output"])
            error = None
            try:
                if reason == "token_cap":
                    raise ValueError("Base completion reached token cap without newline or EOS")
                predicted = parse_base_label(row["completion_text"])
            except ValueError as exc:
                predicted, error = None, str(exc)
            assert row["predicted_label"] == predicted and row["parse_error"] == error
            assert row["predicted_name"] == (LABELS[predicted] if predicted is not None else None)
            assert row["correct"] == (predicted == row["true_label"])
            assert math.isfinite(row["seconds"]) and row["seconds"] >= 0
            correct += row["correct"]
            invalid += error is not None
            if error:
                failures.append({k: row[k] for k in ("set_id", "permutation_id", "test_index", "raw_output", "parse_error")})
        matrix[i][j] = correct / config["num_test"]
        csv_row = csv_lookup[(i, j)]
        assert float(csv_row["accuracy"]) == matrix[i][j]
        assert int(csv_row["num_correct"]) == correct and int(csv_row["parse_failures"]) == invalid
        assert int(csv_row["num_test"]) == config["num_test"]
        for field in ("canonical_demo_indices", "permutation", "ordered_demo_indices"):
            assert json.loads(csv_row[field]) == entry[field]
        saved = summary["configurations"][i * config["num_permutations"] + j]
        assert all(saved[k] == entry[k] for k in entry)
        assert saved["accuracy"] == matrix[i][j] and saved["num_correct"] == correct
        assert saved["parse_failures"] == invalid
    assert set(lookup) == expected_keys
    assert matrix == summary["accuracy_matrix"]
    metrics = independent_metrics(matrix)
    for key, value in metrics.items():
        if value is None:
            assert summary[key] is None
        elif isinstance(value, list):
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(value, summary[key]))
        else:
            assert math.isclose(value, summary[key], abs_tol=1e-12)
    total_seconds = sum(row["seconds"] for row in records)
    assert math.isclose(total_seconds, summary["inference_seconds"])
    assert summary["num_predictions"] == expected_count and summary["parse_failures"] == len(failures)
    previous = json.loads(previous_hashes.read_text()) if previous_hashes else {}
    assert all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in previous.items())
    report = dict(passed=True, predictions_verified=expected_count, configuration_count=len(csv_records),
                  all_prompts_labels_token_ids_boundaries_and_scores_verified=True,
                  metrics_recomputed_independently=True, accuracy_matrix=matrix, metrics=metrics,
                  parse_failures=len(failures), failure_records=failures, inference_seconds=total_seconds,
                  wall_seconds=(datetime.fromisoformat(manifest["finished_utc"]) - datetime.fromisoformat(manifest["started_utc"])).total_seconds(),
                  previous_artifacts_verified=len(previous), previous_sha256=previous)
    if baseline:
        old_plan = json.loads((baseline / "plan.json").read_text())
        old_summary = json.loads((baseline / "summary.json").read_text())
        report["comparison"] = dict(baseline=str(baseline), baseline_metrics={k: old_summary[k] for k in metrics},
                                    first_demo_sets_preserved=plan["demonstration_sets"][:len(old_plan["demonstration_sets"])] == old_plan["demonstration_sets"],
                                    first_permutations_preserved=plan["permutations"][:len(old_plan["permutations"])] == old_plan["permutations"],
                                    common_test_overlap=len(set(plan["test_indices"]) & set(old_plan["test_indices"])),
                                    note="Different evaluation sample sizes and coverage; descriptive, not a fully paired comparison.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--previous-hashes", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = verify(args.run, args.previous_hashes, args.baseline)
    with (args.run / "verification.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("previous_sha256", "failure_records")}, indent=2))


if __name__ == "__main__":
    main()
