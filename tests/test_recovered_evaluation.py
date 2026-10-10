"""Checks for literal recovered loaders and utterance-only compatibility path."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_recovered as evaluator


class RecoveredEvaluationTests(unittest.TestCase):
    def namespace(self):
        return evaluator.original_functions({"os": os, "re": re, "json": json})

    def test_byte_preserved_sources_and_original_search(self):
        namespace = self.namespace()
        self.assertIn("_search", namespace)
        self.assertIn("main", namespace)
        for name, expected in evaluator.HASHES.items():
            self.assertEqual(evaluator.sha256(evaluator.ROOT / "legacy" / name), expected)

    def test_bundled_original_splits(self):
        expected = {"snips": (13784, 0, 700), "atis": (4978, 0, 893),
                    "massive": (11514, 2033, 2974), "hwu": (19229, 759, 830)}
        for task, counts in expected.items():
            with self.subTest(task=task), contextlib.redirect_stdout(io.StringIO()):
                rows = evaluator.load_rows(task, evaluator.ROOT / "data/massive", None, self.namespace())
                self.assertEqual(tuple(map(len, rows)), counts)

    def test_clinc_utterance_only_preserves_order_and_comma_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plus.json"
            path.write_text(json.dumps({"train": [["one,two", "a"], ["second", "b"]],
                                        "oos_train": [["outside", "oos"]],
                                        "test": [["test,one", "a"]], "oos_test": [["other", "oos"]]}))
            train, valid, test = evaluator.load_rows("clinc", Path(directory), path, self.namespace())
            self.assertEqual([r["utt"] for r in train], ["one two", "second", "outside"])
            self.assertEqual([r["intent"] for r in test], ["a", "oos"])
            self.assertEqual(valid, [])
            self.assertTrue(all("ptn" not in row for row in train + test))


if __name__ == "__main__":
    unittest.main()
