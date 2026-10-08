import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_train as runner


@unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "requires ML runtime")
class HistoricalTrainingTests(unittest.TestCase):
    def test_original_loading_preserves_encoder_and_does_not_remap_custom_head(self):
        import torch
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        source = BertModel(config)
        with torch.no_grad():
            source.pooler.dense.weight.fill_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            model, audit = runner.load_initial_model(
                directory, dict(runner.MODEL_DEFAULTS, hidden_size=8), map_inference_mlp=False)
            self.assertEqual(audit["mode"], "original_no_mlp_remap")
            self.assertIsNone(audit["inference_mlp_mapping"])
            torch.testing.assert_close(model.bert.pooler.dense.weight, source.pooler.dense.weight)
            torch.testing.assert_close(model.bert.embeddings.word_embeddings.weight,
                                       source.embeddings.word_embeddings.weight)
            self.assertFalse(torch.equal(model.mlp.dense.weight, source.pooler.dense.weight))

    def test_pattern_mlp_receives_gradients_only_when_enabled(self):
        import torch
        from transformers import BertConfig
        config = BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = dict(runner.MODEL_DEFAULTS, hidden_size=8, apply_ptn_transform=enabled,
                            pairwise_loss_ratio=0.5, pairwise_neg_type="utterance")
                model = runner.pattern_model_class()(config, model_args=SimpleNamespace(**args))
                tokens = torch.randint(1, 12, (2, 4, 6))
                loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).loss
                loss.backward()
                grad = model.ptn_transform.dense.weight.grad
                if enabled:
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                else:
                    self.assertIsNone(grad)


if __name__ == "__main__":
    unittest.main()
