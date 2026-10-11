import dataclasses
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
import export_recovered_training as exporter


@dataclasses.dataclass
class Arguments:
    hidden_size: int = 8
    pooler_type: str = "cls"
    apply_ptn_transform: bool = False
    simcse_utt_loss_ratio: float = 1.0
    simcse_ptn_loss_ratio: float = 1.0
    pairwise_loss_ratio: float = 0.01
    pairwise_neg_type: str = "pattern"


class ExportGuards(unittest.TestCase):
    def test_preserve_existing_and_reject_nested_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with patch.object(exporter, "recovered_module") as load:
                with self.assertRaises(FileExistsError):
                    exporter.export_checkpoint(source, source)
                with self.assertRaisesRegex(ValueError, "outside"):
                    exporter.export_checkpoint(source, source / "export")
                load.assert_not_called()

    def test_metadata_is_authoritative_and_must_be_complete(self):
        module = SimpleNamespace(ModelArguments=Arguments)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "args.json"
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "lacks"):
                exporter.resolve_model_args(module, path)
            saved = Arguments(pairwise_loss_ratio=0.7)
            path.write_text(json.dumps(dataclasses.asdict(saved)))
            actual = exporter.resolve_model_args(module, path)
            self.assertEqual(actual.pairwise_loss_ratio, 0.7)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                exporter.resolve_model_args(module, path, True)
            saved.pooler_type = "avg"
            path.write_text(json.dumps(dataclasses.asdict(saved)))
            with self.assertRaisesRegex(ValueError, "CLS-trained"):
                exporter.resolve_model_args(module, path)

    def test_paper_overrides_without_metadata(self):
        args = exporter.resolve_model_args(SimpleNamespace(ModelArguments=Arguments))
        self.assertEqual(args.pairwise_loss_ratio, 0.5)
        self.assertEqual(args.pairwise_neg_type, "utterance")
        self.assertFalse(args.apply_ptn_transform)


@unittest.skipUnless(os.environ.get("TADSE_RUN_LEGACY_EXPORT_SMOKE") == "1",
                     "requires original training environment and real SentEval")
class LegacyExportIntegration(unittest.TestCase):
    def test_both_pattern_modes_export_exactly_and_preserve_sources(self):
        import torch
        from transformers import BertConfig, BertTokenizer
        module = exporter.recovered_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pattern in (False, True):
                source = root / ("pattern" if pattern else "plain")
                source.mkdir()
                (source / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nneutral\ntext\n")
                tokenizer = BertTokenizer(str(source / "vocab.txt"))
                args = module.ModelArguments(hidden_size=8, apply_ptn_transform=pattern,
                                             pairwise_loss_ratio=0.5, pairwise_neg_type="utterance",
                                             use_fast_tokenizer=False)
                config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=16, max_position_embeddings=64)
                model = module.BertForPatternCL(config, model_args=args)
                with torch.no_grad():
                    model.bert.pooler.dense.weight.fill_(0.01)
                    model.mlp.dense.weight.fill_(0.2)
                model.save_pretrained(str(source))
                tokenizer.save_pretrained(str(source))
                (source / "model_args.json").write_text(json.dumps(dataclasses.asdict(args)))
                before = exporter.directory_hashes(source)
                audit = exporter.export_checkpoint(source, root / (source.name + "-export"))
                self.assertEqual(audit["reload_max_abs_diff"], 0.0)
                self.assertEqual(audit["model_args"]["apply_ptn_transform"], pattern)
                self.assertEqual(before, exporter.directory_hashes(source))
                self.assertTrue(audit["nonpooler_encoder_byte_identical"])


if __name__ == "__main__":
    unittest.main()
