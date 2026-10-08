import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_head_export as probe


class ProbeSourceChecks(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"fixture weight identity")
        benchmark = root / "benchmark.json"
        benchmark.write_text(json.dumps({"tasks": {}}))
        record = {"pooling": "bert-pooler", "benchmark_sha256": probe.sha256(benchmark),
                  "checkpoint_sha256": probe.checkpoint_hashes(checkpoint),
                  "args": {"max_length": 128, "text_preprocessing": "legacy", "baseline": False}}
        return checkpoint, benchmark, record

    def test_changed_checkpoint_or_benchmark_blocks_comparison(self):
        for changed in ("checkpoint", "benchmark"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                checkpoint, benchmark, record = self.fixture(Path(directory))
                probe.compare_source(checkpoint, benchmark, record)
                target = checkpoint / "model.safetensors" if changed == "checkpoint" else benchmark
                target.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    probe.compare_source(checkpoint, benchmark, record)

    def test_baseline_or_different_tokenization_is_not_a_paired_comparison(self):
        for field, value in (("baseline", True), ("max_length", 32), ("text_preprocessing", "none")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                checkpoint, benchmark, record = self.fixture(Path(directory))
                record["args"][field] = value
                with self.assertRaises(ValueError):
                    probe.compare_source(checkpoint, benchmark, record)

    def test_existing_probe_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep").write_text("existing")
            with self.assertRaises(FileExistsError):
                probe.main(["--checkpoint", "unused", "--benchmarks", "unused",
                            "--original-test-results", "unused", "--paper-row", "tadse",
                            "--output-dir", str(root)])
            self.assertEqual((root / "keep").read_text(), "existing")

    def test_nested_probe_output_is_rejected_before_creating_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            output = source / "nested"
            with self.assertRaisesRegex(ValueError, "outside"):
                probe.main(["--checkpoint", str(source), "--benchmarks", "unused",
                            "--original-test-results", "unused", "--paper-row", "tadse",
                            "--output-dir", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
