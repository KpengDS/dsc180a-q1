import unittest
from collections import Counter
from src.ag_news import LABELS, build_prompt, parse_label, select_examples


class SelectionTests(unittest.TestCase):
    def test_balanced_unique_reproducible_selection(self):
        data = {"train": {"label": [i % 4 for i in range(80)]}, "test": list(range(20))}
        demos, tests = select_examples(data, 180)
        self.assertEqual((demos, tests), select_examples(data, 180))
        self.assertEqual(len(set(demos)), 8)
        self.assertEqual(len(set(tests)), 5)
        self.assertEqual(Counter(data["train"]["label"][i] for i in demos), {i: 2 for i in range(4)})

    def test_prompt_has_eight_labels_and_unlabeled_query(self):
        demos = [{"text": "article", "label": i % 4} for i in range(8)]
        prompt = build_prompt(demos, "query")
        self.assertEqual(prompt.count("Article:"), 9)
        self.assertTrue(prompt.endswith('Article: "query"\nLabel:'))
        for label in LABELS:
            self.assertEqual(prompt.count(f"Label: {label}"), 2)

    def test_parser_accepts_labels_and_rejects_ambiguity(self):
        for i, label in enumerate(LABELS):
            self.assertEqual(parse_label(label), i)
        self.assertEqual(parse_label("Label: sports."), 1)
        for text in ("World or Sports", "", "Unknown"):
            with self.assertRaises(ValueError):
                parse_label(text)


if __name__ == "__main__":
    unittest.main()
