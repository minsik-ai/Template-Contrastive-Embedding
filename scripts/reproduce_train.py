#!/usr/bin/env python3
"""Bounded BERT TaDSE reproduction with the upstream loss and stock HF Trainer.

Example (no network):
  python scripts/reproduce_train.py --tiny-random-bert --output-dir /tmp/tadse-smoke

Reload a completed run with ``load_checkpoint(output_dir)``. Tiny random runs
exercise plumbing only; they are not pretrained models or paper reproductions.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

# Model/loss defaults copied from ModelArguments in pairwise_pattern_train.py.
# Deliberately do not import its obsolete Trainer implementation.
MODEL_DEFAULTS = {
    "pooler_type": "cls", "mlp_only_train": False, "apply_ptn_transform": False,
    "simcse_utt_temp": 0.05, "simcse_ptn_temp": 0.05, "pairwise_temp": 0.05,
    "simcse_utt_loss_ratio": 1.0, "simcse_ptn_loss_ratio": 1.0,
    "pairwise_loss_ratio": 0.01, "pairwise_neg_type": "pattern", "mlm_loss_ratio": 0,
}
SMOKE_ROWS = [
    ("book a flight to paris", "book a flight to [city]"),
    ("play jazz music", "play [genre] music"),
    ("weather in london", "weather in [city]"),
    ("set a timer for five minutes", "set a timer for [duration]"),
]


def read_pairs(path, max_samples=None):
    """Read explicit utterance/template pairs without silently dropping bad data."""
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2 for contrastive negatives")
    pairs = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if {"utterance", "pattern"} <= fields:
            utt, ptn = "utterance", "pattern"
        elif {"utt", "ptn"} <= fields:
            utt, ptn = "utt", "ptn"
        else:
            raise ValueError("CSV requires utterance,pattern or utt,ptn columns")
        for line, row in enumerate(reader, 2):
            pair = ((row.get(utt) or "").strip(), (row.get(ptn) or "").strip())
            if not all(pair):
                raise ValueError(f"Empty utterance or pattern at CSV line {line}")
            pairs.append(pair)
            if max_samples is not None and len(pairs) >= max_samples:
                break
    if len(pairs) < 2:
        raise ValueError("At least two pairs are required for contrastive negatives")
    return pairs


class FourViewCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        # Duplicate text is intentional: encoder dropout supplies independent views.
        texts = [text for utt, ptn in rows for text in (utt, utt, ptn, ptn)]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        return {key: value.reshape(len(rows), 4, -1) for key, value in encoded.items()}


def pattern_model_class():
    """Declare the upstream LM head's existing bias alias for safetensors I/O.

    BertLMPredictionHead aliases decoder.bias to its top-level bias. The custom
    wrapper omits the metadata that HF's built-in task wrappers normally set.
    This declares that alias without changing weights or the upstream model.
    """
    from tadse.models import BertForPatternCL
    keys = list(BertForPatternCL._tied_weights_keys or [])
    if "lm_head.decoder.bias" not in keys:
        keys.append("lm_head.decoder.bias")
    BertForPatternCL._tied_weights_keys = keys
    return BertForPatternCL


def load_checkpoint(directory):
    """Load custom TaDSE BERT weights and the arguments required by its constructor."""
    from transformers import AutoTokenizer
    BertForPatternCL = pattern_model_class()
    directory = Path(directory)
    args = SimpleNamespace(**json.loads((directory / "model_args.json").read_text()))
    model = BertForPatternCL.from_pretrained(directory, model_args=args,
                                             local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model, tokenizer


def load_initial_model(source, model_args, *, revision="main", config=None):
    """Load a pretrained encoder without silently randomizing its inference MLP.

    The official SimCSE export renames ``mlp.dense`` to ``pooler.dense`` and
    strips the ``bert.`` prefix. Transformers restores that prefix for our
    wrapper, but does not restore the custom MLP name. Copy the loaded pooler
    only when both custom MLP parameters are absent. Partially loaded heads or
    missing encoder weights are errors, rather than a new random experiment.
    """
    BertForPatternCL = pattern_model_class()
    kwargs = {"model_args": SimpleNamespace(**model_args), "revision": revision,
                  "output_loading_info": True}
    if config is not None:
        kwargs["config"] = config
    model, info = BertForPatternCL.from_pretrained(source, **kwargs)
    missing = set(info.get("missing_keys", []))
    if info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"Invalid pretrained parameters: {info}")
    mlp_keys = {"mlp.dense.weight", "mlp.dense.bias"}
    mapping = None
    if missing & mlp_keys:
        if not mlp_keys <= missing:
            raise ValueError("Pretrained checkpoint only partially initializes the inference MLP")
        pooler_missing = any(key in missing for key in (
            "bert.pooler.dense.weight", "bert.pooler.dense.bias",
            "pooler.dense.weight", "pooler.dense.bias"))
        if pooler_missing:
            raise ValueError("Checkpoint contains neither a complete custom MLP nor loaded BERT pooler")
        model.mlp.dense.load_state_dict(model.bert.pooler.dense.state_dict())
        mapping = "bert.pooler.dense -> mlp.dense (official SimCSE export naming)"
        missing -= mlp_keys
    unexpected_missing = sorted(key for key in missing
                                if not key.startswith(("lm_head.", "ptn_transform."))
                                and not key.endswith("position_ids"))
    if unexpected_missing:
        raise ValueError(f"Pretrained encoder/inference parameters were not loaded: {unexpected_missing}")
    return model, {"loading_info": info, "inference_mlp_mapping": mapping}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", type=Path)
    p.add_argument("--model-name-or-path", default="bert-base-uncased")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tiny-random-bert", action="store_true")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--num-train-epochs", type=float, default=2.0,
                   help="Used when --max-steps -1; a positive max-steps takes precedence")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--pooler-type", choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], default="cls")
    p.add_argument("--mlp-only-train", action="store_true")
    p.add_argument("--apply-ptn-transform", action="store_true")
    for name in ("simcse-utt-temp", "simcse-ptn-temp", "pairwise-temp",
                 "simcse-utt-loss-ratio", "simcse-ptn-loss-ratio", "pairwise-loss-ratio"):
        p.add_argument("--" + name, type=float, default=MODEL_DEFAULTS[name.replace("-", "_")])
    p.add_argument("--pairwise-neg-type", choices=["pattern", "utterance", "balanced"], default="pattern")
    return p


def main(argv=None):
    cli = parser().parse_args(argv)
    if cli.max_steps != -1 and cli.max_steps < 1:
        raise ValueError("max_steps must be -1 (epochs) or a positive integer")
    if cli.batch_size < 2 or cli.max_length < 2:
        raise ValueError("Need batch_size >= 2 and max_length >= 2")
    if not math.isfinite(cli.num_train_epochs) or cli.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be finite and positive")
    if cli.max_samples is not None and cli.max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if not math.isfinite(cli.learning_rate) or cli.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for name in ("simcse_utt_temp", "simcse_ptn_temp", "pairwise_temp"):
        if not math.isfinite(getattr(cli, name)) or getattr(cli, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing runs")
    if cli.train_file:
        cli.train_file = cli.train_file.resolve()
        pairs = read_pairs(cli.train_file, cli.max_samples)
    elif cli.tiny_random_bert:
        pairs = SMOKE_ROWS[:cli.max_samples] if cli.max_samples is not None else SMOKE_ROWS
    else:
        raise ValueError("--train-file is required unless --tiny-random-bert is set")
    if len(pairs) < cli.batch_size:
        raise ValueError("Dataset must contain at least one complete batch")

    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        BertConfig,
        BertTokenizerFast,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    BertForPatternCL = pattern_model_class()
    set_seed(cli.seed)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if cli.tiny_random_bert:
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocab += sorted(set(" ".join(text for pair in pairs for text in pair).lower()
                            .replace("[", " [ ").replace("]", " ] ").split()) - set(vocab))
        vocab_file = cli.output_dir / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=64,
                            max_position_embeddings=max(64, cli.max_length))
    else:
        config = AutoConfig.from_pretrained(cli.model_name_or_path, revision=cli.revision)
        if config.model_type != "bert":
            raise ValueError("Only BERT is supported: upstream RoBERTa/Qwen3 forward paths need repair")
        tokenizer = AutoTokenizer.from_pretrained(cli.model_name_or_path, revision=cli.revision)
    model_args = {key: getattr(cli, key, default) for key, default in MODEL_DEFAULTS.items()}
    model_args["hidden_size"] = config.hidden_size
    model_kwargs = {"config": config, "model_args": SimpleNamespace(**model_args)}
    if cli.tiny_random_bert:
        model = BertForPatternCL(**model_kwargs)
        initialization = {"mode": "deliberately_random_tiny_bert"}
    else:
        model, initialization = load_initial_model(cli.model_name_or_path, model_args,
                                                    revision=cli.revision, config=config)
    collator = FourViewCollator(tokenizer, cli.max_length)
    training_args = TrainingArguments(
        output_dir=str(cli.output_dir), max_steps=cli.max_steps,
        num_train_epochs=cli.num_train_epochs,
        per_device_train_batch_size=cli.batch_size, learning_rate=cli.learning_rate,
        seed=cli.seed, data_seed=cli.seed, use_cpu=cli.use_cpu or cli.tiny_random_bert,
        report_to=[], save_strategy="no", logging_steps=1, remove_unused_columns=False,
        dataloader_drop_last=True, dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=pairs, data_collator=collator)
    probe = tokenizer([pair[0] for pair in pairs[:2]], padding=True, truncation=True,
                      max_length=cli.max_length, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        initial = model(**{k: v.to(model.device) for k, v in probe.items()}, sent_emb=True).pooler_output.cpu()
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Training produced a non-finite loss")
    trainer.save_model(str(cli.output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(cli.output_dir)
    (cli.output_dir / "model_args.json").write_text(json.dumps(model_args, indent=2) + "\n")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(cli).items()}
    metadata.update(rows=len(pairs), torch_version=torch.__version__, transformers_version=transformers.__version__,
                    training_arguments=training_args.to_dict(), backbone_commit=getattr(config, "_commit_hash", None),
                    initialization=initialization,
                    train_sha256=hashlib.sha256(cli.train_file.read_bytes()).hexdigest() if cli.train_file else None)
    (cli.output_dir / "run_args.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.eval()
    with torch.no_grad():
        expected = model(**{k: v.to(model.device) for k, v in probe.items()}, sent_emb=True).pooler_output.cpu()
    restored, restored_tokenizer = load_checkpoint(cli.output_dir)
    restored.to(model.device)
    restored.eval()
    reloaded_probe = restored_tokenizer([pair[0] for pair in pairs[:2]], padding=True,
                                       truncation=True, max_length=cli.max_length, return_tensors="pt")
    with torch.no_grad():
        actual = restored(**{k: v.to(restored.device) for k, v in reloaded_probe.items()},
                          sent_emb=True).pooler_output.cpu()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    changed = not torch.equal(initial, expected)
    if not changed:
        raise RuntimeError("Training did not change the probe embeddings")
    metrics = dict(result.metrics, reload_max_abs_diff=(actual - expected).abs().max().item(),
                   embeddings_changed=changed, tiny_random_bert=cli.tiny_random_bert)
    (cli.output_dir / "verification.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
