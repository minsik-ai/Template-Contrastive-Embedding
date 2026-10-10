import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_multiseed as summary


class MultiseedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        self.runs = []
        for condition in summary.CONDITIONS:
            for index, seed in enumerate(summary.SEEDS):
                path = self.root / f"{condition}-{seed}.json"
                # Legacy macro 72,74,76; top-2 macro 73,76,79. Paired deltas 1,2,3.
                base = 70 + 2 * index + ((index + 1) if condition == "all_intent_top2" else 0)
                path.write_text(json.dumps({
                    "benchmark_sha256": "a" * 64,
                    "source_unchanged": True,
                    "export_audit": {"source_unchanged": True, "reload_max_abs_diff": 0.0},
                    "test_diagnostics": {task: {"accuracy": base + i} for i, task in enumerate(summary.TASKS)},
                }))
                self.runs.append({"condition": condition, "seed": seed, "path": path.name})
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"runs": self.runs}))

    def mutate(self, **changes):
        path = self.root / self.runs[0]["path"]
        record = json.loads(path.read_text())
        record.update(changes)
        path.write_text(json.dumps(record))

    def test_known_means_sample_deviations_and_atomic_output(self):
        output = self.root / "nested" / "summary.json"
        result = summary.main(["--manifest", str(self.manifest), "--output", str(output)])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["conditions"]["legacy_top5"]["macro"], {"n": 3, "mean": 74, "sample_std": 2})
        self.assertEqual(result["conditions"]["all_intent_top2"]["macro"], {"n": 3, "mean": 76, "sample_std": 3})
        self.assertEqual(result["paired_top2_minus_legacy"]["summary"]["macro"], {"n": 3, "mean": 2, "sample_std": 1})
        self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_missing_results_excludes_both_sides_of_pair(self):
        (self.root / self.runs[-1]["path"]).unlink()
        result = summary.summarize(self.manifest)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["matched_seeds"], [42, 43])
        self.assertEqual(len(result["missing_results"]), 1)
        self.assertEqual(result["conditions"]["legacy_top5"]["macro"]["mean"], 73)
        self.assertEqual(result["paired_top2_minus_legacy"]["summary"]["macro"]["mean"], 1.5)

    def test_no_results_and_single_pair_have_no_sample_deviation(self):
        for entry in self.runs:
            if entry["seed"] != 42:
                (self.root / entry["path"]).unlink()
        self.assertIsNone(summary.summarize(self.manifest)["conditions"]["legacy_top5"]["macro"]["sample_std"])
        (self.root / self.runs[0]["path"]).unlink()
        result = summary.summarize(self.manifest)
        self.assertEqual(result["matched_seeds"], [])
        self.assertEqual(result["conditions"]["legacy_top5"]["macro"], {"n": 0, "mean": None, "sample_std": None})

    def test_mismatching_benchmark_rejected_even_if_run_is_unpaired(self):
        self.mutate(benchmark_sha256="b" * 64)
        (self.root / self.runs[3]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Mismatching benchmark"):
            summary.summarize(self.manifest)

    def test_duplicate_entries_and_unpaired_manifest_rejected(self):
        self.runs.append(dict(self.runs[0]))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summary.summarize(self.manifest)
        self.runs.pop()
        self.runs.pop()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "both conditions"):
            summary.summarize(self.manifest)

    def test_partial_json_is_incomplete_until_writer_finishes(self):
        (self.root / self.runs[-1]["path"]).write_text('{"test_diagnostics":')
        result = summary.summarize(self.manifest)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["matched_seeds"], [42, 43])
        self.assertIn("writer finishes", result["missing_results"][0]["reason"])

    def test_manifest_benchmark_is_enforced(self):
        self.manifest.write_text(json.dumps({"runs": self.runs, "benchmark_sha256": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "Mismatching benchmark"):
            summary.summarize(self.manifest)

    def test_failed_export_checks_and_task_mismatch_rejected(self):
        self.mutate(source_unchanged=False)
        with self.assertRaisesRegex(ValueError, "unchanged"):
            summary.summarize(self.manifest)
        self.mutate(source_unchanged=True, export_audit={"source_unchanged": True, "reload_max_abs_diff": 0.001})
        with self.assertRaisesRegex(ValueError, "reload"):
            summary.summarize(self.manifest)
        self.mutate(export_audit={"source_unchanged": True, "reload_max_abs_diff": 0.0}, test_diagnostics={})
        with self.assertRaisesRegex(ValueError, "five benchmark"):
            summary.summarize(self.manifest)


if __name__ == "__main__":
    unittest.main()
