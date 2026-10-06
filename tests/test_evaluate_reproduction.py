import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_reproduction.py"
spec = importlib.util.spec_from_file_location("evaluate_reproduction", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluationTests(unittest.TestCase):
    def test_massive_splits_and_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "massive.jsonl"
            records = [{"partition": part, "utt": "alarm at five", "annot_utt": "alarm at [time : five]",
                            "intent": "alarm_set", "id": index} for index, part in enumerate(("train", "dev", "test"))]
            path.write_text("\n".join(json.dumps(record) for record in records))
            splits = evaluator.read_massive(path)
            self.assertEqual(set(splits), {"train", "valid", "test"})
            self.assertEqual(splits["valid"][0]["pattern"], "alarm at {SLOT}")

    def test_overlap_removed_before_sampling_and_across_queries(self):
        row = lambda utterance: {"utterance": utterance, "pattern": utterance, "intent": "intent"}
        splits = {"train": [row("Hello WORLD"), row("another")],
                      "valid": [row(" hello   world "), row("validation")],
                      "test": [row("VALIDATION"), row("another"), row("test")]}
        cleaned, counts = evaluator.clean_splits(splits)
        self.assertEqual(counts, {"valid": 1, "test": 2})
        self.assertEqual([row["utterance"] for row in cleaned["test"]], ["test"])
        self.assertEqual(len(splits["test"]), 3)

    def test_empty_cleaned_queries_raise(self):
        with self.assertRaisesRegex(ValueError, "No valid"):
            evaluator.clean_splits({key: [{"utterance": "same"}] for key in ("train", "valid", "test")})

    def test_synthetic_training_contamination_excluded(self):
        row = lambda text: {"utterance": text}
        splits = {"train": [row("original")], "valid": [row(" SYNTHETIC  QUERY "), row("clean valid")],
                      "test": [row("synthetic query"), row("clean test")]}
        cleaned, counts = evaluator.exclude_training_overlap(splits, [("synthetic query", "{SLOT}")])
        self.assertEqual(counts, {"valid": 1, "test": 1})
        self.assertEqual(cleaned["valid"], [row("clean valid")])

    def test_subset_is_seeded_and_keeps_source_order(self):
        rows = list(range(30))
        selected = evaluator.subset(rows, 10, 42)
        self.assertEqual(selected, evaluator.subset(rows, 10, 42))
        self.assertEqual(selected, sorted(selected))
        self.assertNotEqual(selected, evaluator.subset(rows, 10, 43))

    def test_validation_selection_breaks_ties_to_smaller_coefficient(self):
        self.assertEqual(evaluator.select_coefficient({0.0: 0.6, 0.1: 0.8, 0.2: 0.8, 0.5: 0.2}), 0.1)

    @unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "opt-in torch integration")
    def test_cosine_nearest_neighbor_is_scale_invariant_and_batched(self):
        import torch
        refs = torch.tensor([[20.0, 0.0], [0.0, 1.0]])
        queries = torch.tensor([[0.1, 0.9], [0.9, 0.1], [-0.1, 0.9]])
        labels = ["vertical", "horizontal", "vertical"]
        self.assertEqual(evaluator.accuracy(refs, queries, ["horizontal", "vertical"], labels, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
