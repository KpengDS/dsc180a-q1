import unittest
from src.base_protocol import build_base_prompt, completion_line, parse_base_label
from src.ag_news import LABELS
from src.design import ExperimentConfig


class BaseProtocolTests(unittest.TestCase):
    def test_plain_balanced_prompt_and_unlabeled_query(self):
        demos = [{"text": f"demo {i}\nwith newline", "label": i % 4} for i in range(8)]
        prompt = build_base_prompt(demos, "unlabeled query")
        self.assertEqual(prompt.count("Article:"), 9)
        self.assertTrue(prompt.endswith('Article: "unlabeled query"\nLabel:'))
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("assistant", prompt)
        for label in LABELS:
            self.assertEqual(prompt.count(f"Label: {label}"), 2)
        with self.assertRaises(ValueError):
            build_base_prompt(demos[:-1], "query")
        with self.assertRaises(ValueError):
            build_base_prompt([{"text": "x", "label": 0}] * 8, "query")

    def test_strict_label_parser(self):
        for index, label in enumerate(LABELS):
            self.assertEqual(parse_base_label(f" {label.lower()} "), index)
        for raw in ("", "Human: World", "Label: World", "World.", '"World"',
                    "World or Sports", "World because politics", "World\nSports", "Technology"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_base_label(raw)

    def test_first_newline_boundary_without_skipping_blank_lines(self):
        self.assertEqual(parse_base_label(completion_line(" World\nArticle: next")), 0)
        self.assertEqual(completion_line(" Sci/Tech"), " Sci/Tech")
        with self.assertRaises(ValueError):
            parse_base_label(completion_line("\nWorld"))
        with self.assertRaises(ValueError):
            parse_base_label(completion_line("Human: World\nSports"))

    def test_runtime_never_calls_chat_template_and_stops_on_generated_newline(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        from transformers import BatchEncoding
        from src.inference import QwenClassifier

        tokenizer = Mock()
        tokenizer.return_value = BatchEncoding({"input_ids": torch.tensor([[1, 2]])})
        tokenizer.decode.side_effect = lambda ids, **kwargs: " World\n" if 99 in ids.tolist() else " World"
        classifier = QwenClassifier.__new__(QwenClassifier)
        classifier.device, classifier.prompt_mode = "cpu", "completion"
        classifier.tokenizer = tokenizer
        classifier.generation = SimpleNamespace(max_new_tokens=12, eos_token_id=0)
        classifier.model = Mock()
        classifier.model.config.max_position_embeddings = 100
        classifier.model.generate.return_value = torch.tensor([[1, 2, 99]])
        demos = [{"text": "demo", "label": i % 4} for i in range(8)]
        result = classifier.predict(demos, "query")
        self.assertEqual(result["predicted_label"], 0)
        self.assertEqual(result["termination_reason"], "newline")
        self.assertEqual(result["raw_output"], " World\n")
        tokenizer.apply_chat_template.assert_not_called()
        stop = classifier.model.generate.call_args.kwargs["stopping_criteria"][0]
        self.assertFalse(stop(torch.tensor([[1, 2, 3]]), None))
        self.assertTrue(stop(torch.tensor([[1, 2, 99]]), None))
        # A generated class name without a boundary is rejected at the token cap.
        classifier.model.generate.return_value = torch.tensor([[1, 2, 3]])
        result = classifier.predict(demos, "query")
        self.assertEqual(result["termination_reason"], "token_cap")
        self.assertIsNone(result["predicted_label"])
        self.assertIsNotNone(result["parse_error"])

    def test_completion_config_requires_balanced_eight_shot(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(prompt_mode="completion", per_class=1).validate()
        with self.assertRaises(ValueError):
            ExperimentConfig(prompt_mode="completion", selection="uniform").validate()


if __name__ == "__main__":
    unittest.main()
