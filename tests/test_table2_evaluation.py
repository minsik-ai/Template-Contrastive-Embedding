"""Protocol contracts; torch nearest-neighbor checks are opt-in on the cluster."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_table2 as evaluate
import prepare_table2_benchmarks as prepare


class PreparationTests(unittest.TestCase):
    def test_jointbert_preserves_alignment_and_rejects_bad_bio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "dev", "test"):
                path = root / "atis" / split
                path.mkdir(parents=True)
                (path / "seq.in").write_text("book flight\n")
                (path / "seq.out").write_text("O O\n")
                (path / "label").write_text("flight\n")
            splits, files = prepare.jointbert(root, "atis")
            self.assertEqual(splits["test"][0]["intent"], "flight")
            self.assertEqual(len(files), 9)
            (root / "atis/test/seq.out").write_text("O\n")
            with self.assertRaisesRegex(ValueError, "BIO"):
                prepare.jointbert(root, "atis")

    def test_clinc_includes_oos_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plus.json"
            records = {name: [[name, "oos" if name.startswith("oos") else "intent"]]
                       for name in ("train", "val", "test", "oos_train", "oos_val", "oos_test")}
            path.write_text(json.dumps(records))
            splits, _ = prepare.clinc(path)
            self.assertEqual([item["intent"] for item in splits["test"]], ["intent", "oos"])
            self.assertEqual(len(splits["valid"]), 2)

    def test_duplicate_audit_never_filters_historical_rows(self):
        sample = lambda text, identity: prepare.row(text, "intent", identity)
        splits = {"train": [sample("same", 0), sample(" SAME ", 1)],
                  "valid": [], "test": [sample("same", 2), sample("same", 3)]}
        audit = prepare.audit(splits)
        self.assertEqual(audit["train_test_overlap_queries"], 2)
        self.assertEqual(audit["duplicate_rows"]["train"], 1)
        self.assertEqual(len(splits["test"]), 2)

    def test_recovered_atis_keeps_multilabel_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                (root / f"atis_{split}.json").write_text(json.dumps(
                    {"rasa_nlu_data": {"common_examples": [{"text": "flight price", "intent": "flight+airfare"}]}}))
            splits, _ = prepare.recovered_atis(root)
            self.assertEqual(splits["test"][0]["intent"], "flight+airfare")
            self.assertEqual(splits["valid"], [])

    def test_hwu_missing_validation_does_not_invent_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                (root / f"{split}.csv").write_text("text,category\nalarm on,alarm_set\n")
            splits, _ = prepare.hwu_csv(root)
            self.assertEqual(splits["valid"], [])
            self.assertEqual(splits["test"][0]["intent"], "alarm_set")


class EvaluationTests(unittest.TestCase):
    def test_per_intent_metrics_and_error_identity(self):
        refs = [prepare.row("ref a", "a", "r0"), prepare.row("ref b", "b", "r1")]
        queries = [prepare.row("q a", "a", "q0"), prepare.row("q b", "b", "q1"), prepare.row("q a2", "a", "q2")]
        result = evaluate.summarize_predictions(queries, refs, [0, 0, 0], [0.9, 0.7, 0.8])
        self.assertAlmostEqual(result["accuracy"], 200 / 3)
        self.assertEqual(result["per_intent"]["b"]["accuracy"], 0)
        self.assertEqual(result["wrong_predictions"][0]["id"], "q1")
        self.assertEqual(result["wrong_predictions"][0]["reference_id"], "r0")

    def test_prediction_count_must_match(self):
        with self.assertRaises(ValueError):
            evaluate.summarize_predictions([prepare.row("q", "a", "q0")], [], [], [])

    @unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "opt-in torch integration")
    def test_chunked_cosine_preserves_first_reference_ties(self):
        import torch
        refs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [20.0, 0.0]])
        queries = torch.tensor([[2.0, 0.1], [0.1, 9.0], [1.0, 0.0]])
        indices, scores = evaluate.nearest_cosine(refs, queries, query_chunk=2, reference_chunk=1)
        self.assertEqual(indices, [0, 1, 0])
        self.assertAlmostEqual(scores[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
