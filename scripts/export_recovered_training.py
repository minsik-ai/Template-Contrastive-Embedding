#!/usr/bin/env python3
"""Export a recovered-training checkpoint using its original Python 3.8 stack."""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "training" / "model"
PROBES = ["Hello.", "Book a flight from Boston to Paris.", "", "neutral text " * 80]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(path):
    return {str(p.relative_to(path)): file_hash(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def recovered_module():
    """Import the intact entrypoint, including its actual custom trainer imports."""
    sys.path.insert(0, str(LEGACY))
    for name in ("tadse", "tadse.models", "tadse.trainers"):
        module = sys.modules.get(name)
        if module is not None and LEGACY not in Path(module.__file__).resolve().parents:
            raise RuntimeError("A non-recovered tadse module is already loaded: " + name)
    spec = importlib.util.spec_from_file_location(
        "recovered_pairwise_pattern_train", LEGACY / "pairwise_pattern_train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_model_args(module, metadata=None, apply_ptn_transform=None):
    if metadata is not None:
        values = json.loads(Path(metadata).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Model arguments metadata must be a JSON object")
        required = {"pooler_type", "apply_ptn_transform", "hidden_size",
                    "simcse_utt_loss_ratio", "simcse_ptn_loss_ratio",
                    "pairwise_loss_ratio", "pairwise_neg_type"}
        if not required.issubset(values):
            raise ValueError("Model arguments metadata lacks: " + ", ".join(sorted(required - set(values))))
        args = module.ModelArguments(**values)
        if apply_ptn_transform is not None and args.apply_ptn_transform != apply_ptn_transform:
            raise ValueError("Pattern-transform flag conflicts with saved model arguments")
    else:
        args = module.ModelArguments(simcse_utt_loss_ratio=1.0, simcse_ptn_loss_ratio=1.0,
                                     pairwise_loss_ratio=0.5, pairwise_neg_type="utterance",
                                     pooler_type="cls", apply_ptn_transform=bool(apply_ptn_transform))
    if args.pooler_type != "cls":
        raise ValueError("Trained-head export requires a CLS-trained checkpoint")
    return args


def state_hashes(model):
    return {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.state_dict().items()}


def export_checkpoint(checkpoint, output_dir, model_args_json=None, apply_ptn_transform=None):
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a fresh export directory")
    if checkpoint in output_dir.parents:
        raise ValueError("Export must be outside the source checkpoint")
    if not checkpoint.is_dir():
        raise ValueError("Checkpoint must be an existing directory")
    if model_args_json is None and (checkpoint / "model_args.json").is_file():
        model_args_json = checkpoint / "model_args.json"
    before_files = directory_hashes(checkpoint)
    module = recovered_module()
    args = resolve_model_args(module, model_args_json, apply_ptn_transform)
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    model, loading = module.BertForPatternCL.from_pretrained(
        str(checkpoint), model_args=args, local_files_only=True, output_loading_info=True)
    problems = {key: value for key, value in loading.items() if value}
    if problems:
        raise RuntimeError("Checkpoint did not load exactly: " + repr(problems))
    model.cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                              use_fast=args.use_fast_tokenizer)
    original = state_hashes(model)
    encoder = state_hashes(model.bert)
    exported = copy.deepcopy(model.bert).cpu().eval()
    with torch.no_grad():
        exported.pooler.dense.load_state_dict(model.mlp.dense.state_dict())
    mapped = {"pooler.dense.weight", "pooler.dense.bias"}
    exported_hashes = state_hashes(exported)
    if set(encoder) != set(exported_hashes) or any(
            encoder[name] != exported_hashes[name] for name in encoder if name not in mapped):
        raise RuntimeError("Export changed encoder tensors outside the trained head")
    output_dir.mkdir(parents=True)
    exported.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    restored = AutoModel.from_pretrained(str(output_dir), local_files_only=True).cpu().eval()
    restored_tokenizer = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True,
                                                       use_fast=args.use_fast_tokenizer)
    if state_hashes(restored) != exported_hashes:
        raise RuntimeError("Reload changed exported tensors")
    max_length = min(32, model.config.max_position_embeddings)
    max_diff = 0.0
    with torch.no_grad():
        for texts in (PROBES, PROBES[:1], PROBES[-2:]):
            source = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            target = restored_tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            if set(source) != set(target) or any(not torch.equal(source[k], target[k]) for k in source):
                raise RuntimeError("Reload changed tokenizer outputs")
            expected = model.mlp(model.bert(**source).last_hidden_state[:, 0])
            actual = restored(**target).pooler_output
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                raise RuntimeError("Non-finite probe output")
            max_diff = max(max_diff, (expected - actual).abs().max().item())
            if not torch.equal(expected, actual):
                raise RuntimeError("CPU head-export equivalence is not exact")
    if state_hashes(model) != original or directory_hashes(checkpoint) != before_files:
        raise RuntimeError("Source model/checkpoint changed during export")
    audit = {"schema_version": 1, "checkpoint": str(checkpoint), "output_dir": str(output_dir),
             "model_args": dataclasses.asdict(args),
             "model_args_source": str(model_args_json) if model_args_json else "original defaults plus explicit paper overrides",
             "model_args_sha256": file_hash(model_args_json) if model_args_json else None,
             "source_files_sha256": before_files, "source_unchanged": True,
             "legacy_source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                                      (LEGACY / "pairwise_pattern_train.py", LEGACY / "tadse" / "models.py")},
             "transformers_version": transformers.__version__, "torch_version": torch.__version__,
             "mapping": {"mlp.dense.weight": "pooler.dense.weight", "mlp.dense.bias": "pooler.dense.bias"},
             "reference": "source.mlp(source.bert(tokens).last_hidden_state[:, 0])",
             "nonpooler_encoder_byte_identical": True, "reload_state_byte_identical": True,
             "reload_max_abs_diff": max_diff, "verification_device": "cpu", "probe_batches": 3,
             "export_files_sha256": directory_hashes(output_dir),
             "limitations": ["Exports the trained CLS head even if mlp_only_train disables it at inference.",
                             "Export equivalence does not establish benchmark replication."]}
    (output_dir / "export-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-args-json", type=Path)
    parser.add_argument("--apply-ptn-transform", action="store_true", default=None)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir,
                                      args.model_args_json, args.apply_ptn_transform), indent=2))


if __name__ == "__main__":
    main()
