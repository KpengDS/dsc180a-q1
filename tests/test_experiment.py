import json
from pathlib import Path
import tempfile
import unittest

from src.ag_news import LABELS, build_prompt
from src.design import ExperimentConfig, canonical_indices, make_plan, sensitivities
from src.experiment import evaluate_plan, verify_paired_plan


class Rows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def fixture():
    return {split: Rows({"label": i % 4, "text": f"article {n - i:03d} class {LABELS[i % 4]}"}
                        for i in range(n)) for split, n in (("train", 80), ("test", 40))}


class DesignTests(unittest.TestCase):
    def test_crossed_design_and_independent_random_streams(self):
        data = fixture()
        plan = make_plan(data, ExperimentConfig())
        self.assertEqual(plan, make_plan(data, ExperimentConfig()))
        larger = make_plan(data, ExperimentConfig(num_sets=3, num_permutations=4, num_test=12))
        self.assertEqual(plan["demonstration_sets"], larger["demonstration_sets"][:2])
        self.assertEqual(plan["permutations"], larger["permutations"][:3])
        self.assertEqual(len(set(plan["test_indices"])), 10)
        self.assertEqual(sorted(plan["test_label_counts"].values()), [2, 2, 3, 3])
        self.assertNotEqual(set(plan["demonstration_sets"][0]), set(plan["demonstration_sets"][1]))
        for indices in plan["demonstration_sets"]:
            self.assertEqual(indices, canonical_indices(data["train"], indices))
            self.assertEqual([sum(data["train"][i]["label"] == c for i in indices) for c in range(4)], [2] * 4)
        for entry in plan["configurations"]:
            self.assertEqual(entry["permutation"], plan["permutations"][entry["permutation_id"]])
            self.assertEqual(sorted(entry["permutation"]), list(range(8)))
            self.assertEqual(entry["ordered_demo_indices"],
                             [entry["canonical_demo_indices"][p] for p in entry["permutation"]])

    def test_pairing_allows_new_model_but_rejects_changed_indices(self):
        import copy
        reference = make_plan(fixture(), ExperimentConfig())
        paired = copy.deepcopy(reference)
        paired["config"].update(model_id="Qwen/Qwen2.5-1.5B", prompt_mode="completion", model_revision="base-sha")
        verify_paired_plan(paired, reference)
        paired["test_indices"][0] = -1
        with self.assertRaisesRegex(ValueError, "test_indices"):
            verify_paired_plan(paired, reference)
        paired = copy.deepcopy(reference)
        paired["config"]["max_chars"] = 500
        with self.assertRaisesRegex(ValueError, "max_chars"):
            verify_paired_plan(paired, reference)

    def test_known_sensitivities_and_zero_denominator(self):
        result = sensitivities([[0, 1], [0, 1]])
        self.assertEqual(result["order_sensitivity"], .5)
        self.assertEqual(result["selection_sensitivity"], 0)
        self.assertEqual(result["selection_over_order"], 0)
        result = sensitivities([[0, 0], [1, 1]])
        self.assertEqual(result["order_sensitivity"], 0)
        self.assertEqual(result["selection_sensitivity"], .5)
        self.assertIsNone(result["selection_over_order"])
        self.assertAlmostEqual(sensitivities([[0, 1], [0, 1]], ddof=1)["order_sensitivity"], 2 ** -.5)
        self.assertEqual(sensitivities([[.7, .7], [.7, .7]])["selection_sensitivity"], 0)

    def test_invalid_designs_fail_before_inference(self):
        for settings in ({"num_test": 0}, {"num_permutations": 40321},
                         {"ordering": "canonical"}, {"std_ddof": 2}, {"per_class": -1}):
            with self.assertRaises(ValueError):
                ExperimentConfig(**settings).validate()
        with self.assertRaises(ValueError):
            make_plan(fixture(), ExperimentConfig(num_test=100))
        with self.assertRaises(ValueError):
            make_plan(fixture(), ExperimentConfig(per_class=21))
        for matrix in ([], [[1], [1, 0]], [[float("nan")]], [[1.1]]):
            with self.assertRaises(ValueError):
                sensitivities(matrix)

    def test_alternative_selection_and_canonical_order(self):
        plan = make_plan(fixture(), ExperimentConfig(selection="uniform", ordering="canonical", num_permutations=1))
        self.assertEqual(plan["permutations"], [list(range(8))])
        self.assertEqual(len(plan["configurations"]), 2)

    def test_end_to_end_artifacts_and_parse_failures(self):
        data = fixture()
        plan = make_plan(data, ExperimentConfig())
        calls = []

        def fake_predict(demos, article, max_chars):
            calls.append((demos, article))
            return dict(predicted_label=None, predicted_name=None, raw_output="bad",
                        parse_error="invalid", rendered_prompt=build_prompt(demos, article, max_chars))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = evaluate_plan(data, plan, fake_predict, output)
            records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 60)
            self.assertEqual(summary["num_predictions"], 60)
            self.assertEqual(summary["parse_failures"], 60)
            self.assertEqual(summary["accuracy_matrix"], [[0] * 3] * 2)
            self.assertEqual(len((output / "configurations.csv").read_text().splitlines()), 7)
            for entry in plan["configurations"]:
                subset = [r for r in records if (r["set_id"], r["permutation_id"]) ==
                          (entry["set_id"], entry["permutation_id"])]
                self.assertEqual([r["test_index"] for r in subset], plan["test_indices"])
            self.assertEqual(len(calls), 60)


if __name__ == "__main__":
    unittest.main()
