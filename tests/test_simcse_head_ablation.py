import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_simcse_head_ablation import initialize_from_simcse, tensor_hashes


@unittest.skipUnless(os.environ.get('TADSE_RUN_LEGACY_EXPORT_SMOKE') == '1', 'requires recovered ML runtime')
class InitializationTests(unittest.TestCase):
    def test_only_shared_head_changes_without_rng_or_pattern_changes(self):
        import torch
        from export_recovered_training import recovered_module
        from transformers import BertConfig
        module = recovered_module()
        args = module.ModelArguments(hidden_size=8, apply_ptn_transform=True,
                                     simcse_utt_loss_ratio=1, simcse_ptn_loss_ratio=1,
                                     pairwise_loss_ratio=0.5, pairwise_neg_type='utterance')
        model = module.BertForPatternCL(BertConfig(vocab_size=20, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2), model_args=args)
        with torch.no_grad():
            model.bert.pooler.dense.weight.fill_(0.25)
            model.bert.pooler.dense.bias.fill_(0.1)
        before = tensor_hashes(model)
        rng = torch.get_rng_state().clone()
        audit = initialize_from_simcse(model)
        after = tensor_hashes(model)
        self.assertEqual(set(audit['changed_tensors']), {'mlp.dense.weight', 'mlp.dense.bias'})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name in before:
            if not name.startswith('mlp.dense.'):
                self.assertEqual(before[name], after[name], name)
        torch.testing.assert_close(model.mlp.dense.weight, model.bert.pooler.dense.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.mlp.dense.bias, model.bert.pooler.dense.bias, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
