"""CPU integration checks for the recovered-loop Qwen compatibility adapter."""

import ast
import hashlib
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("TADSE_RUN_TRAINING_SMOKE") == "1",
    "requires modern Torch/Transformers and recovered 4.2.1 source",
)
class QwenTrainerTest(unittest.TestCase):
    def test_original_loop_two_epochs_save_and_reload(self):
        from dataclasses import dataclass

        import torch
        from transformers import BertConfig, BertModel, TrainingArguments
        from transformers.utils import ModelOutput

        from scripts.qwen_trainer import (
            RECOVERED_METHODS,
            RECOVERED_SOURCE,
            QwenCLTrainer,
        )

        @dataclass
        class LossOutput(ModelOutput):
            loss: object = None
            simcse_utt_loss: object = None
            simcse_ptn_loss: object = None
            mlm_loss: object = None
            pairwise_loss: object = None

        class TinyModel(BertModel):
            def forward(self, input_ids, **kwargs):
                out = super().forward(input_ids=input_ids, **kwargs)
                loss = (out.pooler_output - 0.25).square().mean()
                return LossOutput(
                    loss=loss, simcse_utt_loss=loss, pairwise_loss=loss * 0.5
                )

        source = os.environ["TADSE_LEGACY_TRANSFORMERS_SOURCE"]
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            model = TinyModel(
                BertConfig(
                    vocab_size=20,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                )
            )
            before = model.embeddings.word_embeddings.weight.detach().clone()
            args = TrainingArguments(
                output_dir=temp,
                use_cpu=True,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                logging_steps=1,
                save_steps=2,
                save_total_limit=1,
                learning_rate=1e-3,
                report_to=[],
                disable_tqdm=True,
            )
            trainer = QwenCLTrainer(
                model=model,
                args=args,
                train_dataset=[{"input_ids": [1, 2, 3]}] * 4,
                legacy_transformers_source=source,
            )
            self.assertEqual(
                trainer.train.__func__.__code__.co_filename, str(RECOVERED_SOURCE)
            )
            tree = ast.parse(RECOVERED_SOURCE.read_text())
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "CLTrainer"
            )
            for name in RECOVERED_METHODS:
                node = next(n for n in cls.body if getattr(n, "name", None) == name)
                sha = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode()
                ).hexdigest()
                self.assertEqual(
                    trainer.legacy_audit["ast_sha256"][f"trainers.py:CLTrainer.{name}"],
                    sha,
                )
            result = trainer.train()
            self.assertEqual(result.global_step, 4)
            self.assertEqual(trainer.state.epoch, 2)
            self.assertFalse(
                torch.equal(before, model.embeddings.word_embeddings.weight)
            )
            self.assertTrue(
                any("simcse_utt_loss" in row for row in trainer.state.log_history)
            )
            checkpoint = Path(temp) / "checkpoint-4"
            for filename in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
                self.assertTrue((checkpoint / filename).is_file())
            reloaded = TinyModel.from_pretrained(checkpoint)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, reloaded.state_dict()[name]), name)
            self.assertEqual(type(trainer.optimizer).__name__, "AdamW")
            self.assertTrue(trainer.optimizer.param_groups[0]["correct_bias"])

    def test_rejects_gradient_accumulation(self):
        from transformers import BertConfig, BertModel, TrainingArguments

        from scripts.qwen_trainer import QwenCLTrainer

        with tempfile.TemporaryDirectory() as temp:
            args = TrainingArguments(
                output_dir=temp,
                use_cpu=True,
                gradient_accumulation_steps=2,
                report_to=[],
            )
            with self.assertRaisesRegex(ValueError, "gradient_accumulation"):
                QwenCLTrainer(
                    model=BertModel(
                        BertConfig(
                            hidden_size=8,
                            intermediate_size=16,
                            num_hidden_layers=1,
                            num_attention_heads=2,
                        )
                    ),
                    args=args,
                    legacy_transformers_source=os.environ[
                        "TADSE_LEGACY_TRANSFORMERS_SOURCE"
                    ],
                )


if __name__ == "__main__":
    unittest.main()
