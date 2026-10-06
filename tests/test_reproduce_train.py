"""Input contracts plus an opt-in real, offline forward/backward/reload test.

Run integration with TADSE_RUN_TRAINING_SMOKE=1 python -m unittest discover -s tests.
"""
import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_train.py"
spec = importlib.util.spec_from_file_location("reproduce_train", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class InputTests(unittest.TestCase):
    def write_csv(self, directory, header, rows):
        path = Path(directory) / "pairs.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_both_schemas_and_sample_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            for header in (("utterance", "pattern"), ("utt", "ptn")):
                path = self.write_csv(directory, header, [(" a ", "[x]"), ("b", "[y]"), ("c", "[z]")])
                self.assertEqual(runner.read_pairs(path, 2), [("a", "[x]"), ("b", "[y]")])

    def test_missing_columns_and_empty_values_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            for header, rows in [(("text", "label"), [("a", "b")]),
                                 (("utt", "ptn"), [("a", "x"), ("", "y")]),
                                 (("utt", "ptn"), [("a", "x")])]:
                path = self.write_csv(directory, header, rows)
                with self.assertRaises(ValueError):
                    runner.read_pairs(path)

    def test_invalid_sample_limit(self):
        with self.assertRaises(ValueError):
            runner.read_pairs("unused.csv", 1)

    def test_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keep.txt"
            path.write_text("existing experiment")
            with self.assertRaisesRegex(ValueError, "empty output"):
                runner.main(["--tiny-random-bert", "--output-dir", directory])
            self.assertEqual(path.read_text(), "existing experiment")

    def test_four_view_order(self):
        # A tiny tokenizer stub catches ordering without requiring ML dependencies.
        class FakeTensor:
            def reshape(self, batch, views, remaining):
                return batch, views, remaining

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                self.texts = texts
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

        tokenizer = Tokenizer()
        batch = runner.FourViewCollator(tokenizer, 32)([("hello", "[greeting]"), ("bye", "[farewell]")])
        self.assertEqual(tokenizer.texts, ["hello", "hello", "[greeting]", "[greeting]",
                                          "bye", "bye", "[farewell]", "[farewell]"])
        self.assertEqual(batch["input_ids"], (2, 4, -1))


@unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "opt-in torch/transformers integration")
class TrainingSmokeTests(unittest.TestCase):
    def test_exported_bert_pooler_is_restored_to_custom_inference_mlp(self):
        import torch
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        source = BertModel(config)
        with torch.no_grad():
            source.pooler.dense.weight.fill_(0.125)
            source.pooler.dense.bias.fill_(-0.25)
        source.eval()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            model_args = dict(runner.MODEL_DEFAULTS, hidden_size=config.hidden_size)
            loaded, audit = runner.load_initial_model(directory, model_args)
            loaded.eval()
            self.assertIn("pooler", audit["inference_mlp_mapping"])
            torch.testing.assert_close(loaded.mlp.dense.weight, source.pooler.dense.weight)
            torch.testing.assert_close(loaded.mlp.dense.bias, source.pooler.dense.bias)
            inputs = {"input_ids": torch.tensor([[2, 5, 3], [2, 8, 3]]),
                          "attention_mask": torch.ones(2, 3, dtype=torch.long)}
            with torch.no_grad():
                expected = source(**inputs).pooler_output
                actual = loaded(**inputs, sent_emb=True).pooler_output
            torch.testing.assert_close(actual, expected)

    def test_missing_encoder_weights_fail_instead_of_random_initialization(self):
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        source = BertModel(config)
        state = source.state_dict()
        del state["embeddings.word_embeddings.weight"]
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory, state_dict=state)
            with self.assertRaisesRegex(ValueError, "encoder/inference parameters"):
                runner.load_initial_model(directory, dict(runner.MODEL_DEFAULTS, hidden_size=8))

    def test_real_training_and_exact_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            metrics = runner.main(["--tiny-random-bert", "--output-dir", str(output), "--max-steps", "2"])
            self.assertTrue(metrics["embeddings_changed"])
            self.assertLess(metrics["reload_max_abs_diff"], 1e-5)
            self.assertEqual(json.loads((output / "run_args.json").read_text())["rows"], 4)
            self.assertEqual(json.loads((output / "model_args.json").read_text())["hidden_size"], 32)
            self.assertTrue((output / "model.safetensors").exists())
            self.assertTrue((output / "tokenizer.json").exists())
            restored, _ = runner.load_checkpoint(output)
            self.assertIs(restored.lm_head.bias, restored.lm_head.decoder.bias)


if __name__ == "__main__":
    unittest.main()
