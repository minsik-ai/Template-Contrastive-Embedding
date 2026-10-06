import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "data/massive/augment_data.py"
spec = importlib.util.spec_from_file_location("augmentation", SCRIPT)
aug = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = aug
spec.loader.exec_module(aug)


class AugmentationTests(unittest.TestCase):
    def test_multislot_cartesian_original_and_repeated_types(self):
        row = aug.annotated("from original to original", "from [city : original] to [city : original]")
        values = list(aug.variants(row, {"city": [("a", 2), ("b", 1)]}))
        self.assertEqual(values, [row.text, "from a to a", "from a to b", "from b to a", "from b to b"])
        self.assertEqual(row.pattern, "from {SLOT} to {SLOT}")
        self.assertEqual(len(list(aug.variants(row, {"city": [("a", 2), ("b", 1)]}, 2))), 3)

    def test_offsets_unicode_sorted_entities_and_invalid_overlap(self):
        row = {"text": "fly Zürich to 東京", "entities": [
            {"start": 14, "end": 16, "entity": "to"},
            {"start": 4, "end": 10, "entity": "from"}]}
        example = aug.offsets(row)
        self.assertEqual(example.pattern, "fly {SLOT} to {SLOT}")
        self.assertEqual(example.parts[1], ("Zürich", "from"))
        row["entities"].append({"start": 5, "end": 8, "entity": "bad"})
        with self.assertRaises(ValueError):
            aug.offsets(row)

    def test_train_only_vocabulary_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"partition": "train", "utt": 'go Zürich, please', "annot_utt": 'go [city : Zürich], please'},
                {"partition": "train", "utt": 'visit Paris', "annot_utt": 'visit [city : Paris]'},
                {"partition": "test", "utt": 'go Paris, please', "annot_utt": 'go [city : SECRET], please'},
                {"partition": "test", "utt": 'held out', "annot_utt": '[city : SECRET]'}]
            (root / "massive_en-US.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            first = aug.prepare("massive", root, root / "out")
            second = aug.prepare("massive", root, root / "out-repeat")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertNotIn("SECRET", str(first["slot_vocabulary"]))
            with (root / "out/massive_train.csv").open(newline="") as handle:
                output = list(csv.DictReader(handle))
            self.assertIn({"utterance": 'go Zürich, please', "pattern": 'go {SLOT}, please'}, output)
            self.assertNotIn('go Paris, please', [row["utterance"] for row in output])
            self.assertEqual(first["generated_heldout_matches_removed"], 1)

    def test_ties_follow_first_appearance_and_literal_brackets(self):
        rows = [aug.annotated("b", "[city : b]"), aug.annotated("a", "[city : a]")]
        self.assertEqual(aug.slot_vocabulary(rows, 1), {"city": [("b", 1)]})
        self.assertEqual(aug.annotated("[literal]", "[literal]").pattern, "[literal]")

    def test_hwu_semicolon_csv_excludes_heldout_before_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "hwu64_orig.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["answer_normalised", "answer_annotation"])
                writer.writerow(["play jazz; now", "play [music : jazz]; now"])
                writer.writerow(["play secret", "play [music : secret]"])
            (root / "hwu64_valid_raw").write_text("play secret\n")
            (root / "hwu64_test_raw").write_text("")
            rows, _, _, _ = aug.load_source("hwu", root)
            self.assertEqual(aug.slot_vocabulary(rows, 3), {"music": [("jazz", 1)]})
            self.assertEqual(rows[0].text, "play jazz; now")

    def test_clinc_multiset_proof_and_holdout_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clinc_raw_ner_spacy_lg_topk30.log").write_text(
                "Text : hello - 1\nPtn : hello\nText : go Paris - 2\nPtn : go [GPE : Paris]\n")
            path = root / "splits.json"
            path.write_text(json.dumps({"train": [["hello", "greeting"], ["go Paris", "travel"]],
                                        "val": [["hello", "greeting"]]}))
            rows, _, _, _ = aug.load_source("clinc", root, path)
            self.assertEqual([row.text for row in rows], ["go Paris"])
            path.write_text(json.dumps({"train": [["hello", "greeting"]], "test": [["go Paris", "travel"]]}))
            with self.assertRaises(ValueError):
                aug.load_source("clinc", root, path)

    def test_pooled_holdout_filter_applies_before_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = ([aug.annotated("go Rome", "go [city : Rome]"),
                       aug.annotated("go Paris", "go [city : Paris]")], set(), [], [])
            result = aug.prepare("massive", root, root, loaded=loaded, extra_heldout={"  GO   PARIS\t"})
            self.assertEqual(result["train_examples"], 1)
            self.assertEqual(result["slot_vocabulary"], {"city": [("Rome", 1)]})
            self.assertEqual(result["output_rows"], 1)

    def test_cli_combines_selected_sources_and_records_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "massive_en-US.jsonl").write_text(json.dumps({
                "partition": "train", "utt": "hello", "annot_utt": "hello"}) + "\n")
            output = root / "out"
            subprocess.run([sys.executable, str(SCRIPT), "--sources", "massive",
                            "--data-root", str(root), "--output-dir", str(output)],
                           check=True, capture_output=True, text=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["combined"]["source_order"], ["massive"])
            self.assertEqual(manifest["combined"]["output_rows"], 1)
            self.assertEqual(manifest["combined"]["sha256"], aug.sha256(output / "combined_train.csv"))
            self.assertEqual((output / "combined_train.csv").read_bytes(),
                             (output / "massive_train.csv").read_bytes())

    def test_case_and_whitespace_holdout_matches_excluded_preserving_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"partition": "train", "utt": "go SECRET", "annot_utt": "go [city : SECRET]"},
                {"partition": "train", "utt": "Go  Zürich", "annot_utt": "Go  [city : Zürich]"},
                {"partition": "train", "utt": "visit Rome", "annot_utt": "visit [city : Rome]"},
                {"partition": "test", "utt": "  GO  secret\t", "annot_utt": "GO [city : secret]"},
                {"partition": "test", "utt": "go rome", "annot_utt": "go [city : rome]"}]
            (root / "massive_en-US.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            result = aug.prepare("massive", root, root / "out")
            self.assertEqual(result["train_examples"], 2)
            self.assertNotIn("SECRET", str(result["slot_vocabulary"]))
            self.assertEqual(result["generated_heldout_matches_removed"], 1)
            with (root / "out/massive_train.csv").open(newline="") as handle:
                utterances = [row["utterance"] for row in csv.DictReader(handle)]
            self.assertIn("Go  Zürich", utterances)
            self.assertNotIn("Go  Rome", utterances)

    def test_prepare_refuses_to_replace_existing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "massive_train.csv"
            target.write_bytes(b"existing source")
            with self.assertRaises(FileExistsError):
                aug.prepare("massive", root, root)
            self.assertEqual(target.read_bytes(), b"existing source")

    def test_cli_preserves_existing_outputs_and_rejects_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            expected = {"combined_train.csv": b"original combined", "manifest.json": b"original manifest"}
            for name, content in expected.items():
                (output / name).write_bytes(content)
            for flags in ([], ["--top-k", "0"], ["--max-combinations", "-1"]):
                with self.subTest(flags=flags):
                    result = subprocess.run([sys.executable, str(SCRIPT), "--sources", "massive",
                                             "--output-dir", str(output)] + flags,
                                            capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, expected)
            for flag in ("--top-k", "--max-combinations"):
                fresh = root / flag[2:]
                result = subprocess.run([sys.executable, str(SCRIPT), flag, "0",
                                         "--output-dir", str(fresh)], capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(fresh.exists())

    def test_cli_validates_input_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "massive_en-US.jsonl").write_text("not json")
            output = root / "out"
            result = subprocess.run([sys.executable, str(SCRIPT), "--sources", "massive",
                                     "--data-root", str(root), "--output-dir", str(output)],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_clinc_requires_membership_evidence(self):
        with self.assertRaises(FileNotFoundError):
            aug.load_source("clinc", Path("."))


if __name__ == "__main__":
    unittest.main()
