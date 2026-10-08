#!/usr/bin/env python3
"""Offline SimCSE-style export of a CLS-trained TaDSE checkpoint's trained MLP."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from reproduce_train import load_checkpoint

PROBES = ["Hello.", "Please set a timer for five minutes.", "What is the weather in Paris today?",
          "Find a train from Boston to New York tomorrow morning.", "", "neutral text " * 80]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(directory):
    return {str(path.relative_to(directory)): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def state_hashes(model):
    import torch
    return {name: hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def export_head(checkpoint: Path, output_dir: Path, device="cpu"):
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new export directory; existing artifacts are preserved")
    if output_dir.is_relative_to(checkpoint):
        raise ValueError("Export must be outside the source checkpoint directory")
    if not checkpoint.is_dir():
        raise ValueError("Checkpoint must be an existing directory")
    args = json.loads((checkpoint / "model_args.json").read_text(encoding="utf-8"))
    if args.get("pooler_type") != "cls":
        raise ValueError("Trained-head export requires a CLS-trained checkpoint (pooler_type=cls)")
    source_files = directory_hashes(checkpoint)
    import torch
    from transformers import AutoModel, AutoTokenizer, BertModel
    model, tokenizer = load_checkpoint(checkpoint)
    if not isinstance(model.bert, BertModel) or model.bert.pooler is None:
        raise ValueError("Expected a BERT encoder with an existing pooler")
    if model.training:
        raise ValueError("Source loader must return an evaluation-mode model")
    original_state = state_hashes(model)
    encoder_state = state_hashes(model.bert)
    exported = copy.deepcopy(model.bert)
    with torch.no_grad():
        exported.pooler.dense.weight.copy_(model.mlp.dense.weight)
        exported.pooler.dense.bias.copy_(model.mlp.dense.bias)
    mapped = {"pooler.dense.weight", "pooler.dense.bias"}
    changed_state = state_hashes(exported)
    unchanged = [name for name in encoder_state if name not in mapped]
    if set(changed_state) != set(encoder_state) or any(changed_state[name] != encoder_state[name] for name in unchanged):
        raise RuntimeError("Export changed encoder tensors outside pooler.dense")
    output_dir.mkdir(parents=True)
    exported.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    restored = AutoModel.from_pretrained(output_dir, local_files_only=True).to(device).eval()
    restored_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    if state_hashes(restored) != changed_state:
        raise RuntimeError("Reloaded export is not byte-identical to the mapped encoder")
    max_length = min(32, model.config.max_position_embeddings)
    if max_length < 4:
        raise ValueError("Checkpoint position capacity is too small for export verification")
    differences = []
    # Include mixed lengths, singleton and truncated batches; use each saved tokenizer.
    for texts in (PROBES, PROBES[:1], PROBES[-2:]):
        source_tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        target_tokens = restored_tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        if set(source_tokens) != set(target_tokens) or any(not torch.equal(source_tokens[key], target_tokens[key]) for key in source_tokens):
            raise RuntimeError("Export changed tokenizer outputs")
        with torch.inference_mode():
            cls = model.bert(**source_tokens).last_hidden_state[:, 0]
            expected = model.mlp(cls).cpu()
            actual = restored(**{key: value.to(device) for key, value in target_tokens.items()}).pooler_output.cpu()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        differences.append((actual - expected).abs().max().item())
    if state_hashes(model) != original_state or directory_hashes(checkpoint) != source_files:
        raise RuntimeError("Source checkpoint or original in-memory model changed during export")
    audit = {"schema_version": 1, "checkpoint": str(checkpoint), "output_dir": str(output_dir),
             "mapping": {"mlp.dense.weight": "pooler.dense.weight", "mlp.dense.bias": "pooler.dense.bias"},
             "pooler_type": "cls", "source_mlp_only_train": args.get("mlp_only_train"),
             "reference": "original model.mlp(original model.bert(tokens).last_hidden_state[:,0])",
             "source_files_sha256": source_files, "source_unchanged": True,
             "nonpooler_encoder_byte_identical": True, "nonpooler_tensor_count": len(unchanged),
             "reload_state_byte_identical": True, "reload_max_abs_diff": max(differences),
             "verification_device": str(device), "probe_batches": 3, "probe_max_length": max_length,
             "export_files_sha256": directory_hashes(output_dir),
             "limitations": ["Head conversion only; it does not establish benchmark replication.",
                             "Exports the trained CLS MLP even when source inference used mlp_only_train=True."]}
    with (output_dir / "export-audit.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(export_head(args.checkpoint, args.output_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
