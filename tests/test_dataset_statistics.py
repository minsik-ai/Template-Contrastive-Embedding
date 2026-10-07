"""Provenance tests for enrichment using the real unchanged augmentation CLI."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "data/massive/augment_data.py"
INSPECTOR = ROOT / "scripts/inspect_dataset_statistics.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DatasetStatisticsTests(unittest.TestCase):
    def fixture(self, root, pooled=False):
        raw = root / "raw"
        raw.mkdir()
        records = [
            {"partition": "train", "utt": "go Paris", "annot_utt": "go [city : Paris]"},
            {"partition": "train", "utt": "play jazz", "annot_utt": "play [genre : jazz]"},
            {"partition": "dev", "utt": "held out", "annot_utt": "held out"},
        ]
        source = raw / "massive_en-US.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        sources = ["massive"]
        if pooled:
            train = [{"text": "book Boston", "entities": [{"start": 5, "end": 11, "entity": "city", "value": "Boston"}]}]
            test = [{"text": " GO   PARIS ", "entities": []}]
            for name, examples in (("train", train), ("test", test)):
                (raw / f"atis_{name}.json").write_text(json.dumps({"rasa_nlu_data": {"common_examples": examples}}))
            sources.append("atis")
        output = root / "generated"
        subprocess.run([sys.executable, str(GENERATOR), "--data-root", str(raw),
                        "--output-dir", str(output), "--sources", *sources],
                       check=True, capture_output=True, text=True)
        return raw, output, output / "manifest.json"

    def inspect(self, raw, manifest):
        return subprocess.run([sys.executable, str(INSPECTOR), "--manifest", str(manifest),
                               "--data-root", str(raw), "--clinc-splits-json", str(raw / "unused-clinc.json")],
                              capture_output=True, text=True, check=False)

    def test_sidecars_use_same_pooled_filter_and_preserve_original_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output, manifest = self.fixture(Path(directory), pooled=True)
            before = {path.name: digest(path) for path in output.iterdir()}
            result = self.inspect(raw, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            enriched = json.loads((output / "release-manifest.json").read_text())
            self.assertEqual(before, {name: digest(output / name) for name in before})
            self.assertEqual(enriched["statistics_provenance"]["generation_manifest_sha256"], before["manifest.json"])
            original = enriched["sources"]["massive"]["original_examples"]
            evidence = output / original["output"]
            self.assertEqual(digest(evidence), original["sha256"])
            self.assertEqual([json.loads(line) for line in evidence.read_text().splitlines()], [
                {"utterance": "play jazz", "template": "play {SLOT}", "slots": [{"type": "genre", "value": "jazz"}]}
            ])

    def test_changed_source_blocks_before_any_sidecar_write(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output, manifest = self.fixture(Path(directory))
            source = raw / "massive_en-US.jsonl"
            source.write_text(source.read_text() + "\n")
            result = self.inspect(raw, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Original source changed", result.stderr)
            self.assertFalse((output / "release-manifest.json").exists())
            self.assertEqual(list(output.glob("*_original_examples.jsonl")), [])

    def test_original_count_mismatch_blocks_before_any_sidecar_write(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output, manifest = self.fixture(Path(directory))
            content = json.loads(manifest.read_text())
            content["sources"]["massive"]["train_examples"] += 1
            manifest.write_text(json.dumps(content))
            result = self.inspect(raw, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Original-example count", result.stderr)
            self.assertFalse((output / "release-manifest.json").exists())
            self.assertEqual(list(output.glob("*_original_examples.jsonl")), [])

    def test_existing_sidecar_or_release_manifest_is_preserved(self):
        for filename in ("massive_original_examples.jsonl", "release-manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                raw, output, manifest = self.fixture(Path(directory))
                existing = output / filename
                existing.write_text("keep previous evidence\n")
                before = {path.name: digest(path) for path in output.iterdir()}
                result = self.inspect(raw, manifest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already exist", result.stderr)
                self.assertEqual(before, {path.name: digest(path) for path in output.iterdir()})

    def test_alternate_root_with_same_rowcount_cannot_relabel_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output, manifest = self.fixture(root)
            alternate = root / "different-inputs"
            alternate.mkdir()
            (alternate / "massive_en-US.jsonl").write_text(
                (raw / "massive_en-US.jsonl").read_text().replace("Paris", "London"))
            result = self.inspect(alternate, manifest)
            self.assertNotEqual(result.returncode, 0, "Unrelated same-size input must not create trusted original evidence")
            self.assertIn("Loaded sources differ from verified generation inputs", result.stderr)
            self.assertFalse((output / "release-manifest.json").exists())
            self.assertEqual(list(output.glob("*_original_examples.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
