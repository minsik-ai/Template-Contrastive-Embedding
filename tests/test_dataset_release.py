"""Exercise the publication boundary with real small CSV/JSONL evidence files."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_dataset_release.py"
spec = importlib.util.spec_from_file_location("dataset_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class DatasetReleaseTests(unittest.TestCase):
    def fixture(self, root):
        original = root / "original.jsonl"
        original.write_text(json.dumps({"utterance": "visit Paris", "template": "visit {SLOT}",
                                            "slots": [{"type": "destination", "value": "Paris"}]}) + "\n")
        data = root / "snips_train.csv"
        with data.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utterance", "pattern"])
            writer.writerow(["visit Paris", "visit {SLOT}"])
            writer.writerow(["visit London", "visit {SLOT}"])
        combined = root / "combined_train.csv"
        combined.write_bytes(data.read_bytes())
        record = {"status": "prepared", "top_k": 5, "train_examples": 1, "output_rows": 2,
                      "max_combinations_per_example": None, "examples_truncated_by_cap": 0,
                      "output": data.name, "sha256": release.sha256(data),
                      "inputs": [{"path": str(original), "sha256": release.sha256(original)}],
                      "original_examples": {"output": original.name, "sha256": release.sha256(original)}}
        manifest = {"sources": {"snips": record}, "combined": {"output": combined.name,
                        "sha256": release.sha256(combined), "output_rows": 2, "source_order": ["snips"]}}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def failed_names(self, report):
        return {c["name"] for c in report["checks"] if not c["passed"]}

    def test_rounding_boundaries_preserve_published_precision(self):
        self.assertEqual(release.rounded_interval("163K"), (162500, 163500))
        self.assertEqual(release.rounded_interval("13.1K"), (13050, 13150))
        self.assertEqual(release.rounded_interval("4.0K"), (3950, 4050))
        self.assertEqual(release.rounded_interval("39"), (39, 40))
        report = {"checks": []}
        for actual in (162499, 162500, 163499, 163500):
            release.count_check(report, "boundary", actual, "163K")
        self.assertEqual([check["passed"] for check in report["checks"]], [False, True, True, False])

    def test_small_actual_corpus_cannot_claim_paper_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = self.fixture(root)
            manifest["sources"]["snips"]["output_rows"] = 163000
            path.write_text(json.dumps(manifest))
            report = release.audit(path)
            self.assertFalse(report["eligible_for_private_upload"])
            self.assertEqual(report["observed"]["snips"]["utterances"], 2)
            self.assertIn("snips.declared_rows", self.failed_names(report))
            self.assertIn("snips.paper.utterances", self.failed_names(report))
            self.assertIn("sources.exact_membership", self.failed_names(report))
            self.assertIn("atis.present", self.failed_names(report))

    def test_hash_topk_cap_and_missing_original_evidence_fail_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.fixture(Path(directory))
            record = manifest["sources"]["snips"]
            record.update(sha256="0" * 64, top_k=1, max_combinations_per_example=10,
                          examples_truncated_by_cap=1)
            del record["original_examples"]
            path.write_text(json.dumps(manifest))
            failed = self.failed_names(release.audit(path))
            self.assertTrue({"snips.csv.sha256", "snips.top_k", "snips.uncapped",
                             "snips.original_examples.available", "snips.paper.values"} <= failed)

    def test_original_stats_values_are_strings_not_typed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "originals.jsonl"
            records = [{"utterance": "from Paris", "template": "from {SLOT}", "slots": [{"type": "from", "value": "Paris"}]},
                       {"utterance": "to Paris", "template": "to {SLOT}", "slots": [{"type": "to", "value": "Paris"}]},
                       {"utterance": "to London", "template": "to {SLOT}", "slots": [{"type": "to", "value": "London"}]}]
            path.write_text("\n".join(json.dumps(row) for row in records))
            self.assertEqual(release.original_statistics(path), {"originals": 3, "slots": 2, "values": 2, "templates": 2})

    def test_original_input_change_is_detected_even_if_output_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            with (root / "original.jsonl").open("a") as handle:
                handle.write("\n")
            failed = self.failed_names(release.audit(path))
            self.assertIn("snips.input.0.sha256", failed)
            self.assertIn("snips.original_examples.sha256", failed)

    def test_success_boundary_and_combined_content_with_small_fixture_targets(self):
        # Replaces target data, not the auditor: demonstrates the full passing path
        # without materializing 834k test rows. Production CLI has no target override.
        targets = {"snips": {"top_k": 5, "slots": "1", "values": "1", "templates": "1", "originals": "1", "utterances": "2"}}
        totals = {key: value for key, value in targets["snips"].items() if key != "top_k"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(release, "PAPER", targets), mock.patch.object(release, "TOTAL", totals):
            root = Path(directory)
            path, manifest = self.fixture(root)
            self.assertTrue(release.audit(path)["eligible_for_private_upload"])
            combined = root / "combined_train.csv"
            combined.write_text("utterance,pattern\nvisit Rome,visit {SLOT}\nvisit London,visit {SLOT}\n")
            manifest["combined"]["sha256"] = release.sha256(combined)
            path.write_text(json.dumps(manifest))
            report = release.audit(path)
            self.assertFalse(report["eligible_for_private_upload"])
            self.assertIn("combined.matches_sources", self.failed_names(report))

    def test_cli_persists_failed_audit_without_packaging_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            output = root / "release"
            with mock.patch.object(release, "package") as package:
                self.assertEqual(release.main(["--manifest", str(path), "--output-dir", str(output)]), 2)
                package.assert_not_called()
            audit_bytes = (output / "verification.json").read_bytes()
            report = json.loads(audit_bytes)
            self.assertFalse(report["eligible_for_private_upload"])
            self.assertTrue(report["failures"])
            self.assertIn("UNVERIFIED / BLOCKED", (output / "README.md").read_text())
            self.assertFalse((output / "data").exists())
            with self.assertRaises(SystemExit):
                release.main(["--manifest", str(path), "--output-dir", str(output)])
            self.assertEqual((output / "verification.json").read_bytes(), audit_bytes)

    def test_missing_manifest_still_produces_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit"
            self.assertEqual(release.main(["--manifest", str(root / "missing.json"), "--output-dir", str(output)]), 2)
            report = json.loads((output / "verification.json").read_text())
            self.assertIn("manifest.readable", self.failed_names(report))

    def test_artifact_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = self.fixture(root)
            manifest["sources"]["snips"]["output"] = "../outside.csv"
            path.write_text(json.dumps(manifest))
            self.assertIn("snips.csv.available", self.failed_names(release.audit(path)))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet integration requires pinned optional runtime")
    def test_passing_audit_packages_actual_rows_and_records_hashes(self):
        import pyarrow.parquet as pq
        targets = {"snips": {"top_k": 5, "slots": "1", "values": "1", "templates": "1", "originals": "1", "utterances": "2"}}
        totals = {key: value for key, value in targets["snips"].items() if key != "top_k"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(release, "PAPER", targets), mock.patch.object(release, "TOTAL", totals):
            root = Path(directory)
            path, _ = self.fixture(root)
            output = root / "release"
            self.assertEqual(release.main(["--manifest", str(path), "--output-dir", str(output)]), 0)
            report = json.loads((output / "verification.json").read_text())
            self.assertTrue(report["eligible_for_private_upload"])
            self.assertTrue(report["packaged"])
            parquet = output / report["parquet_files"][0]["path"]
            self.assertEqual(release.sha256(parquet), report["parquet_files"][0]["sha256"])
            self.assertEqual(pq.read_table(parquet).to_pylist(), [
                {"utterance": "visit Paris", "pattern": "visit {SLOT}", "source": "snips"},
                {"utterance": "visit London", "pattern": "visit {SLOT}", "source": "snips"}])
            self.assertIn("config_name: combined", (output / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
