"""Tiny, offline Qwen checks for the native-width TaDSE adapter."""

import os
import tempfile
import unittest
from types import SimpleNamespace

if os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1":
    import torch
    from transformers import AutoModel, Qwen3Config, Qwen3Model

    from model.tadse.qwen_models import Qwen3ForPatternCL


def model_args(pattern=False):
    return SimpleNamespace(
        pooler_type="pad_left_no_pooler", hidden_size=768,
        simcse_utt_temp=0.05, simcse_ptn_temp=0.05, pairwise_temp=0.05,
        apply_ptn_transform=pattern, pairwise_neg_type="utterance",
        simcse_utt_loss_ratio=1.0, simcse_ptn_loss_ratio=1.0,
        pairwise_loss_ratio=0.5, mlp_only_train=False,
    )


def config():
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, pad_token_id=0,
        attention_dropout=0.1,
    )


@unittest.skipUnless(os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1", "opt-in torch integration")
class QwenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_four_view_loss_and_gradients(self):
        for pattern in (False, True):
            with self.subTest(pattern=pattern):
                torch.manual_seed(42)
                model = Qwen3ForPatternCL(config(), model_args(pattern))
                ids = torch.randint(1, 64, (3, 4, 7))
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids))
                self.assertTrue(torch.isfinite(output.loss))
                self.assertEqual(output.logits.shape, (3, 3))
                output.loss.backward()
                gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                self.assertEqual(sum(isinstance(m, Qwen3Model) for m in model.modules()), 1)
                self.assertFalse(hasattr(model, "mlp"))
                if pattern:
                    self.assertEqual(model.ptn_transform.dense.in_features, 32)
                    self.assertGreater(model.ptn_transform.dense.weight.grad.abs().sum(), 0)
                else:
                    self.assertFalse(hasattr(model, "ptn_transform"))

    def test_left_padding_invariance_and_rejections(self):
        model = Qwen3ForPatternCL(config(), model_args()).eval()
        batch = torch.tensor([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]])
        with torch.no_grad():
            batched = model(input_ids=batch, attention_mask=batch.ne(0), sent_emb=True)
            single = model(input_ids=batch[:1, 2:], attention_mask=torch.ones(1, 3), sent_emb=True)
        torch.testing.assert_close(batched.pooler_output[:1], single.pooler_output, atol=1e-6, rtol=1e-5)
        self.assertEqual(batched.pooler_output.shape, (2, 32))
        for mask in ([[1, 1, 0]], [[0, 0, 0]], [[1, 0, 1]], [[0, 2, 1]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                model(input_ids=torch.ones(1, 3, dtype=torch.long), attention_mask=torch.tensor(mask), sent_emb=True)
        with self.assertRaises(ValueError):
            model(input_ids=torch.ones(2, 3, 4, dtype=torch.long), attention_mask=torch.ones(2, 3, 4))

    def test_pretrained_identity_reload_and_native_export(self):
        torch.manual_seed(17)
        original = Qwen3Model(config()).eval()
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as saved, tempfile.TemporaryDirectory() as exported:
            original.save_pretrained(base)
            reference = AutoModel.from_pretrained(base)
            wrapped = Qwen3ForPatternCL.from_encoder_pretrained(base, model_args=model_args(True)).eval()
            self.assertEqual(wrapped.config.attention_dropout, 0.1)
            self.assertEqual(set(reference.state_dict()), set(wrapped.model.state_dict()))
            for key, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, wrapped.model.state_dict()[key]), key)
            wrapped.save_pretrained(saved)
            reloaded = Qwen3ForPatternCL.from_pretrained(saved).eval()
            ids = torch.tensor([[0, 3, 4], [5, 6, 7]])
            mask = ids.ne(0)
            with torch.no_grad():
                before = wrapped(input_ids=ids, attention_mask=mask, sent_emb=True).pooler_output
                after = reloaded(input_ids=ids, attention_mask=mask, sent_emb=True).pooler_output
            torch.testing.assert_close(before, after, atol=0, rtol=0)
            wrapped.model.save_pretrained(exported)
            native = AutoModel.from_pretrained(exported).eval()
            for key, tensor in wrapped.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, native.state_dict()[key]), key)
            positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
            with torch.no_grad():
                result = native(input_ids=ids, attention_mask=mask, position_ids=positions).last_hidden_state[:, -1]
            torch.testing.assert_close(before, result, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
