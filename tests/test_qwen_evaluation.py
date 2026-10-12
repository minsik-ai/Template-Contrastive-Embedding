import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_qwen as evaluator


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class QwenPoolingTests(unittest.TestCase):
    def test_last_real_token_with_left_padding(self):
        import torch
        hidden = torch.arange(24.0).reshape(2, 3, 4)
        mask = torch.tensor([[0, 0, 1], [0, 1, 1]])
        torch.testing.assert_close(evaluator.last_token_pool(hidden, mask), hidden[:, -1])

    def test_mixed_batch_matches_individual_pooling(self):
        import torch
        sequences = [torch.arange(12.0).reshape(3, 4), torch.arange(4.0).reshape(1, 4)]
        hidden = torch.stack([sequences[0], torch.cat([torch.zeros(2, 4), sequences[1]])])
        batched = evaluator.truncate_normalize(evaluator.last_token_pool(hidden, torch.tensor([[1, 1, 1], [0, 0, 1]])), 3)
        for index, sequence in enumerate(sequences):
            single = evaluator.truncate_normalize(evaluator.last_token_pool(sequence[None], torch.ones(1, len(sequence))), 3)
            torch.testing.assert_close(batched[index], single[0], rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires transformers")
    def test_tiny_qwen_mixed_batch_matches_singleton_encoding(self):
        import numpy as np
        import torch
        from transformers import Qwen3Config, Qwen3Model

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                ids = [[int(token) for token in text.split()] for text in texts]
                width = max(map(len, ids))
                return {"input_ids": torch.tensor([[0] * (width - len(row)) + row for row in ids]),
                        "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids])}

        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=8, attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = Qwen3Model(config).eval()
        texts = ["1 2 3 4", "5", "6 7"]
        batched = evaluator.encode(model, Tokenizer(), texts, 3, 128, 8)
        singles = evaluator.encode(model, Tokenizer(), texts, 1, 128, 8)
        np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)

    def test_truncation_precedes_normalization(self):
        import torch
        result = evaluator.truncate_normalize(torch.tensor([[3.0, 4.0, 100.0]]), 2)
        torch.testing.assert_close(result, torch.tensor([[0.6, 0.8]]))
        torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))

    def test_reject_all_padding_and_nonfinite(self):
        import torch
        with self.assertRaisesRegex(ValueError, "All-padding"):
            evaluator.last_token_pool(torch.ones(1, 2, 3), torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            evaluator.last_token_pool(torch.full((1, 2, 3), float("nan")), torch.ones(1, 2))
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                evaluator.truncate_normalize(torch.tensor([[value, 1.0]]), 2)
        with self.assertRaisesRegex(ValueError, "zero"):
            evaluator.truncate_normalize(torch.zeros(1, 3), 2)
        with self.assertRaisesRegex(ValueError, "Dimensions"):
            evaluator.truncate_normalize(torch.ones(1, 3), 4)


if __name__ == "__main__":
    unittest.main()
