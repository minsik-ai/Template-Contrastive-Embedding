import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_table2_head as exporter
import reproduce_train as runner


class ExportGuards(unittest.TestCase):
    def test_existing_output_is_preserved_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing"
            output.mkdir()
            (output / "keep").write_text("preserve")
            with patch.object(exporter, "load_checkpoint") as load, self.assertRaises(FileExistsError):
                exporter.export_head(root / "missing", output)
            load.assert_not_called()
            self.assertEqual((output / "keep").read_text(), "preserve")

    def test_rejects_non_cls_and_nested_export_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "model_args.json").write_text(json.dumps({"pooler_type": "avg"}))
            with patch.object(exporter, "load_checkpoint") as load:
                with self.assertRaisesRegex(ValueError, "CLS-trained"):
                    exporter.export_head(source, root / "export")
                with self.assertRaisesRegex(ValueError, "outside"):
                    exporter.export_head(source, source / "export")
                load.assert_not_called()
            self.assertFalse((root / "export").exists())


@unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "opt-in torch integration")
class HeadExportIntegration(unittest.TestCase):
    def test_trained_head_mapping_reload_and_source_immutability(self):
        import torch
        from transformers import AutoModel, BertConfig, BertTokenizerFast
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source"
            checkpoint.mkdir()
            vocab = checkpoint / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nneutral\ntext\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
            args = dict(runner.MODEL_DEFAULTS, hidden_size=8, mlp_only_train=True)
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=16, max_position_embeddings=64)
            model = runner.pattern_model_class()(config, model_args=SimpleNamespace(**args)).eval()
            with torch.no_grad():
                model.bert.pooler.dense.weight.fill_(0.01)
                model.bert.pooler.dense.bias.fill_(0.02)
                model.mlp.dense.weight.copy_(torch.eye(8) * 0.3)
                model.mlp.dense.bias.fill_(0.4)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            (checkpoint / "model_args.json").write_text(json.dumps(args))
            files = exporter.directory_hashes(checkpoint)
            memory = exporter.state_hashes(model)
            with patch.object(exporter, "load_checkpoint", return_value=(model, tokenizer)):
                audit = exporter.export_head(checkpoint, root / "export")
            self.assertEqual(exporter.directory_hashes(checkpoint), files)
            self.assertEqual(exporter.state_hashes(model), memory)
            self.assertEqual(audit["reload_max_abs_diff"], 0.0)
            restored = AutoModel.from_pretrained(root / "export", local_files_only=True)
            torch.testing.assert_close(restored.pooler.dense.weight, model.mlp.dense.weight, atol=0, rtol=0)
            torch.testing.assert_close(restored.pooler.dense.bias, model.mlp.dense.bias, atol=0, rtol=0)
            for name, tensor in model.bert.state_dict().items():
                if not name.startswith("pooler."):
                    self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
            self.assertTrue((root / "export/model.safetensors").exists())
            self.assertTrue((root / "export/export-audit.json").exists())
            # Also exercise the real checkpoint loader, not only the in-memory guard.
            actual = exporter.export_head(checkpoint, root / "real-loader")
            self.assertTrue(actual["source_unchanged"])
            self.assertTrue(actual["nonpooler_encoder_byte_identical"])


if __name__ == "__main__":
    unittest.main()
