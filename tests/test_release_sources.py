"""Real two-source orchestration tests; the augmentation implementation is unchanged."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_release_sources.py"
GENERATOR = ROOT / "data/massive/augment_data.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseSourcesTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        records = [
            {"partition": "train", "utt": "cross source", "annot_utt": "cross source"},
            {"partition": "train", "utt": "massive own", "annot_utt": "massive own"},
            {"partition": "train", "utt": "go Paris", "annot_utt": "go [city : Paris]"},
            {"partition": "dev", "utt": " MASSIVE   OWN ", "annot_utt": " MASSIVE   OWN "},
            {"partition": "test", "utt": "another heldout", "annot_utt": "another heldout"},
        ]
        (raw / "massive_en-US.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        with (raw / "hwu64_orig.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["answer_normalised", "answer_annotation"])
            writer.writerows([(text, text) for text in ("hwu keep", "hwu own", "massive own")])
        (raw / "hwu64_valid_raw").write_text(" HWU   OWN \n")
        (raw / "hwu64_test_raw").write_text("cross source\n")
        return raw

    def run_cli(self, raw, output, sources=("massive", "hwu")):
        return subprocess.run([sys.executable, str(SCRIPT), "--data-root", str(raw),
                               "--clinc-splits-json", str(raw / "unused-clinc.json"),
                               "--output-dir", str(output), "--sources", *sources],
                              capture_output=True, text=True, check=False)

    def rows(self, path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_per_source_filtering_combination_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            output = root / "candidate"
            code_hash = sha256(GENERATOR)
            result = self.run_cli(raw, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sha256(GENERATOR), code_hash)
            manifest = json.loads((output / "release-manifest.json").read_text())
            self.assertEqual(manifest["heldout_scope"], "per_source")
            self.assertEqual(manifest["combined"]["source_order"], ["massive", "hwu"])
            massive = self.rows(output / manifest["sources"]["massive"]["output"])
            hwu = self.rows(output / manifest["sources"]["hwu"]["output"])
            self.assertEqual([row["utterance"] for row in massive], ["cross source", "go Paris"])
            self.assertEqual([row["utterance"] for row in hwu], ["hwu keep", "massive own"])
            combined = output / manifest["combined"]["output"]
            self.assertEqual(self.rows(combined), massive + hwu)
            self.assertEqual(manifest["combined"]["output_rows"], 4)
            self.assertEqual(manifest["combined"]["sha256"], sha256(combined))
            for source in ("massive", "hwu"):
                record = manifest["sources"][source]
                self.assertEqual(record["top_k"], 3)
                self.assertIsNone(record["max_combinations_per_example"])
                self.assertEqual(record["train_examples"], 2)
                self.assertEqual(record["sha256"], sha256(output / record["output"]))
                evidence = output / record["original_examples"]["output"]
                self.assertEqual(record["original_examples"]["sha256"], sha256(evidence))
                self.assertEqual(len(evidence.read_text().splitlines()), 2)
                individual = json.loads((output / source / "release-manifest.json").read_text())
                self.assertEqual(record["inputs"], individual["sources"][source]["inputs"])
                self.assertEqual(individual["sources"][source]["output"], f"{source}_train.csv")
                for reference in manifest["source_manifests"][source].values():
                    self.assertEqual(reference["sha256"], sha256(output / reference["output"]))

    def test_existing_output_including_empty_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            output = root / "candidate"
            output.mkdir()
            self.assertNotEqual(self.run_cli(raw, output).returncode, 0)
            self.assertEqual(list(output.iterdir()), [])
            keep = output / "keep.txt"
            keep.write_text("existing artifact")
            self.assertNotEqual(self.run_cli(raw, output).returncode, 0)
            self.assertEqual(keep.read_text(), "existing artifact")
            self.assertEqual(list(output.iterdir()), [keep])

    def test_duplicate_sources_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            output = root / "candidate"
            self.assertNotEqual(self.run_cli(raw, output, ("massive", "massive")).returncode, 0)
            self.assertFalse(output.exists())

    def test_missing_source_never_produces_successful_root_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            (raw / "massive_en-US.jsonl").unlink()
            output = root / "candidate"
            result = self.run_cli(raw, output, ("massive",))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Requested source was not prepared", result.stderr)
            self.assertFalse((output / "release-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
