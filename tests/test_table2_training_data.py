"""Replay original code in subprocesses, preserving writes and legacy truncation."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_table2_training.py"
ORIGINAL = ROOT / "legacy/augment_data.py"
LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyTrainingDataTests(unittest.TestCase):
    def run_cli(self, root, output, sources, hashseed="42", generator=ORIGINAL, extra_args=()):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        return subprocess.run([sys.executable, str(SCRIPT), "--data-root", str(root),
                               "--output-dir", str(output), "--sources", *sources,
                               "--original-generator", str(generator),
                               "--clinc-splits-json", str(root / "clinc_splits.json"), *extra_args],
                              capture_output=True, text=True, check=False, env=env)

    def csv_rows(self, path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def snips_fixture(self, root):
        for index, label in enumerate(LABELS):
            directory = root / "snips" / label
            directory.mkdir(parents=True)
            text = f"intent {index}"
            if index == 6:
                text = "final, intent\nsecond line"
            (directory / f"train_{label}_full.json").write_text(json.dumps({label: [{"data": [{"text": text}]}]}))

    def test_original_snips_overwrite_and_embedded_newline_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snips_fixture(root)
            before = sha256(ORIGINAL)
            result = self.run_cli(root, root / "result", ["snips"])
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / "result"
            manifest = json.loads((output / "manifest.json").read_text())
            record = manifest["sources"]["snips"]
            self.assertEqual(record["write_attempts"], 7)
            self.assertEqual(record["retained_rows"], 1)
            self.assertEqual(record["output_opens"], 7)
            self.assertEqual(record["rows_per_output_open"], [1] * 7)
            self.assertEqual(record["raw_physical_lines"], 3)
            expected = "final  intent\nsecond line"
            self.assertEqual((output / "snips_legacy.csv").read_bytes(), f"{expected},{expected},0\n".encode())
            self.assertEqual(self.csv_rows(output / "snips_train.csv"), [{"utterance": expected, "pattern": expected, "slot_count": "0"}])
            self.assertEqual(self.csv_rows(output / "combined_train.csv"), self.csv_rows(output / "snips_train.csv"))
            self.assertEqual(sha256(ORIGINAL), before)
            # Independently run the original unwrapped: raw bytes must be identical.
            direct = root / "direct.csv"
            subprocess.run([sys.executable, str(ORIGINAL), "-d", str(root / "snips"),
                            "-p", "train", "-k", "5", "-o", str(direct), "-t", "snips"],
                           check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONHASHSEED="42"))
            self.assertEqual(direct.read_bytes(), (output / "snips_legacy.csv").read_bytes())
            self.assertEqual(manifest["combined"]["sha256"], sha256(output / "combined_train.csv"))

    def test_snips_all_intents_append_records_without_repeated_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snips_fixture(root)
            legacy = root / "legacy"
            complete = root / "complete"
            self.assertEqual(self.run_cli(root, legacy, ["snips"]).returncode, 0)
            result = self.run_cli(root, complete, ["snips"],
                                  extra_args=("--snips-top-k", "2", "--snips-retain-all-intents"))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.csv_rows(complete / "snips_train.csv")
            self.assertEqual(len(rows), 7)
            self.assertEqual([row["utterance"] for row in rows[:6]], [f"intent {i}" for i in range(6)])
            self.assertEqual(rows[-1], self.csv_rows(legacy / "snips_train.csv")[0])
            self.assertEqual(self.csv_rows(complete / "combined_train.csv"), rows)
            self.assertEqual((complete / "snips_legacy.csv").read_bytes(), (legacy / "snips_legacy.csv").read_bytes())
            record = json.loads((complete / "manifest.json").read_text())["sources"]["snips"]
            self.assertEqual(record["top_k"], 2)
            self.assertEqual(record["trainable_retention"], "all_intents")
            self.assertEqual(record["retained_rows"], 7)
            self.assertEqual(record["rows_per_intent"], dict.fromkeys(LABELS, 1))

    def test_snips_top_k_controls_original_slot_substitutions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snips_fixture(root)
            for index, label in enumerate(LABELS):
                path = root / "snips" / label / f"train_{label}_full.json"
                path.write_text(json.dumps({label: [{"data": [
                    {"text": f"intent {index} "}, {"text": f"city{index}", "entity": "city"}]}]}))
            totals = []
            for k in (1, 2):
                output = root / f"k{k}"
                result = self.run_cli(root, output, ["snips"], extra_args=(
                    "--snips-top-k", str(k), "--snips-retain-all-intents"))
                self.assertEqual(result.returncode, 0, result.stderr)
                totals.append(len(self.csv_rows(output / "snips_train.csv")))
            self.assertEqual(totals, [13, 19])
            bad = self.run_cli(root, root / "bad", ["snips"], extra_args=("--snips-top-k", "-1"))
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root / "bad").exists())

    def test_clinc_cached_annotation_membership_and_original_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clinc_raw_ner_spacy_lg_topk30.log").write_text(
                "Text : go Paris - 0\nPtn : go [GPE : Paris]\nText : visit London - 1\nPtn : visit [GPE : London]\n")
            (root / "clinc_splits.json").write_text(json.dumps({"train": [["go Paris", "travel"]],
                                                                "oos_train": [["visit London", "oos"]],
                                                                "test": [["go Paris", "travel"]]}))
            result = self.run_cli(root, root / "result", ["clinc"])
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.csv_rows(root / "result/clinc_train.csv")
            self.assertEqual({row["utterance"] for row in rows}, {"go Paris", "go London", "visit Paris", "visit London"})
            self.assertTrue(all(row["slot_count"] == "1" for row in rows))
            self.assertEqual({row["pattern"] for row in rows}, {"go {SLOT}", "visit {SLOT}"})
            (root / "clinc_splits.json").write_text(json.dumps({"train": [["unrelated", "travel"]]}))
            bad = self.run_cli(root, root / "bad", ["clinc"])
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("multiset differs", bad.stderr)
            self.assertFalse((root / "bad/manifest.json").exists())

    def test_ambiguous_legacy_commas_fail_without_silent_column_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "massive_en-US.jsonl").write_text(json.dumps({"partition": "train", "utt": "hello, world", "annot_utt": "hello, world"}) + "\n")
            result = self.run_cli(root, root / "result", ["massive"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Ambiguous unescaped commas", result.stderr)
            self.assertFalse((root / "result/manifest.json").exists())

    def test_clinc_literal_brackets_are_not_invented_ner_entities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "is there a travel alert for [country]"
            (root / "clinc_raw_ner_spacy_lg_topk30.log").write_text(
                f"Text : {text} - 0\nPtn : is there a travel alert for [ country ]\n")
            (root / "clinc_splits.json").write_text(json.dumps({"train": [[text, "travel"]]}))
            result = self.run_cli(root, root / "result", ["clinc"])
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.csv_rows(root / "result/clinc_train.csv")
            self.assertEqual(rows, [{"utterance": text,
                                    "pattern": "is there a travel alert for {SLOT}", "slot_count": "1"}])

    def test_hashseed_and_original_source_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snips_fixture(root)
            wrong_seed = self.run_cli(root, root / "wrongseed", ["snips"], hashseed="0")
            self.assertNotEqual(wrong_seed.returncode, 0)
            self.assertFalse((root / "wrongseed").exists())
            modified = root / "modified.py"
            modified.write_bytes(ORIGINAL.read_bytes() + b"\n")
            wrong_source = self.run_cli(root, root / "wrongsource", ["snips"], generator=modified)
            self.assertNotEqual(wrong_source.returncode, 0)
            self.assertFalse((root / "wrongsource").exists())

    def test_repeated_run_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snips_fixture(root)
            output = root / "result"
            self.assertEqual(self.run_cli(root, output, ["snips"]).returncode, 0)
            before = {path.name: sha256(path) for path in output.iterdir()}
            self.assertNotEqual(self.run_cli(root, output, ["snips"]).returncode, 0)
            self.assertEqual(before, {path.name: sha256(path) for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
