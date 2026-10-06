#!/usr/bin/env python3
"""Validation-selected semantic compression and cosine 1NN on MASSIVE.

Compression follows arXiv:2305.14299v3 section 3.3 equation (5):
  representation = (1 - coefficient) * utterance + coefficient * template
Vectors are cosine-normalized *after* mixing. Validation selects the coefficient;
test labels never participate in selection. This is a reconstructed evaluator,
not a claim of exact historical benchmark parity.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_train import load_checkpoint, load_initial_model, read_pairs


def text_key(text):
    return " ".join(text.casefold().split())


def read_massive(path):
    splits = {"train": [], "valid": [], "test": []}
    with open(path, encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            item = json.loads(raw)
            split = {"dev": "valid", "validation": "valid"}.get(item["partition"], item["partition"])
            if split not in splits:
                raise ValueError(f"Unknown split on line {line}: {split}")
            pattern = re.sub(r"\[([^\[\]:]+?)\s*:\s*([^\[\]]+)\]",
                             "{SLOT}", item["annot_utt"])
            if "[" in pattern or "]" in pattern:
                raise ValueError(f"Unparsed slot annotation on line {line}")
            row = {"utterance": item["utt"], "pattern": pattern, "intent": str(item["intent"]),
                       "id": str(item.get("id", line))}
            if not all(row[key].strip() for key in ("utterance", "pattern", "intent")):
                raise ValueError(f"Empty evaluation field on line {line}")
            splits[split].append(row)
    if not all(splits.values()):
        raise ValueError("Source must contain train, dev/valid and test rows")
    return splits


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"utterance", "pattern", "intent"} <= set(reader.fieldnames or []):
            raise ValueError("Evaluation CSV requires utterance,pattern,intent")
        rows = [dict(row, id=str(index)) for index, row in enumerate(reader)]
    if not rows or any(not (row.get(key) or "").strip() for row in rows
                       for key in ("utterance", "pattern", "intent")):
        raise ValueError("Evaluation CSV contains no rows or empty fields")
    return rows


def clean_splits(splits):
    """Remove query matches against the *entire* train set before pilot sampling.

    Validation takes precedence over test for repeated query text. Reference rows
    retain original order, which defines deterministic ties for exact similarity.
    """
    seen = {text_key(row["utterance"]) for row in splits["train"]}
    result = {"train": list(splits["train"])}
    exclusions = {}
    for split in ("valid", "test"):
        kept = [row for row in splits[split] if text_key(row["utterance"]) not in seen]
        exclusions[split] = len(splits[split]) - len(kept)
        result[split] = kept
        seen.update(text_key(row["utterance"]) for row in kept)
        if not kept:
            raise ValueError(f"No {split} queries remain after excluding split overlap")
    return result, exclusions


def exclude_training_overlap(splits, pairs):
    """Filter queries against the actual fine-tuning utterances, including synthetic rows."""
    training_texts = {text_key(utterance) for utterance, _ in pairs}
    cleaned = {"train": splits["train"]}
    exclusions = {}
    for split in ("valid", "test"):
        cleaned[split] = [row for row in splits[split]
                          if text_key(row["utterance"]) not in training_texts]
        exclusions[split] = len(splits[split]) - len(cleaned[split])
        if not cleaned[split]:
            raise ValueError(f"No {split} queries remain after fine-tuning contamination exclusion")
    return cleaned, exclusions


def subset(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("Sample limits must be positive")
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def select_coefficient(scores):
    # Prefer the smaller coefficient on a validation tie.
    return min(scores, key=lambda coefficient: (-scores[coefficient], coefficient))


def embed(model, tokenizer, texts, batch_size, max_length):
    import torch
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(model.device) for key, value in tokens.items()}
            outputs.append(model(**tokens, sent_emb=True).pooler_output.cpu())
    return torch.cat(outputs)


def accuracy(refs, queries, ref_labels, query_labels, batch_size):
    import torch.nn.functional as F
    refs, queries = F.normalize(refs, dim=-1), F.normalize(queries, dim=-1)
    correct = 0
    for start in range(0, len(queries), batch_size):
        nearest = (queries[start:start + batch_size] @ refs.T).argmax(dim=1).tolist()
        correct += sum(ref_labels[index] == query_labels[start + offset]
                       for offset, index in enumerate(nearest))
    return correct / len(query_labels)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline", action="store_true", help="Use the recorded original backbone with the same model/pooling settings")
    p.add_argument("--massive-source", type=Path)
    p.add_argument("--train-csv", type=Path)
    p.add_argument("--valid-csv", type=Path)
    p.add_argument("--test-csv", type=Path)
    p.add_argument("--training-csv", type=Path,
                   help="Relocated actual fine-tuning CSV; must match the recorded training SHA256")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-refs", type=int)
    p.add_argument("--max-queries", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--coefficients", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5])
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 2 or any(not 0 <= c <= 1 for c in args.coefficients):
        raise ValueError("Invalid batch size, sequence length, or compression coefficient")
    if any(value is not None and value < 1 for value in (args.max_refs, args.max_queries)):
        raise ValueError("Sample limits must be positive")
    if args.output.exists():
        raise ValueError("Output already exists; preserve prior evaluation results")
    if args.massive_source:
        if any((args.train_csv, args.valid_csv, args.test_csv)):
            raise ValueError("Choose MASSIVE source or three CSVs")
        splits = read_massive(args.massive_source)
        sources = [args.massive_source]
    else:
        sources = [args.train_csv, args.valid_csv, args.test_csv]
        if not all(sources):
            raise ValueError("Provide --massive-source or all three split CSV files")
        splits = {split: read_csv(path) for split, path in zip(("train", "valid", "test"), sources)}
    original_counts = {key: len(value) for key, value in splits.items()}
    run_args = json.loads((args.checkpoint / "run_args.json").read_text())
    model_args = json.loads((args.checkpoint / "model_args.json").read_text())
    training_csv = args.training_csv or (Path(run_args["train_file"]) if run_args.get("train_file") else None)
    training_audit = {"status": "tiny random fixture; no training CSV", "exclusions": {"valid": 0, "test": 0}}
    if training_csv is not None:
        train_hash = hashlib.sha256(training_csv.read_bytes()).hexdigest()
        if train_hash != run_args.get("train_sha256"):
            raise ValueError("Actual fine-tuning CSV does not match the training manifest SHA256")
        training_pairs = read_pairs(training_csv, run_args.get("max_samples"))
        splits, train_exclusions = exclude_training_overlap(splits, training_pairs)
        training_audit = {"path": str(training_csv), "sha256": train_hash,
                              "rows": len(training_pairs), "exclusions": train_exclusions}
    elif not run_args.get("tiny_random_bert"):
        raise ValueError("Training CSV is required to audit fine-tuning/query contamination")
    splits, exclusions = clean_splits(splits)
    splits = {key: subset(rows, args.max_refs if key == "train" else args.max_queries, args.seed)
              for key, rows in splits.items()}
    from transformers import AutoTokenizer, set_seed
    set_seed(run_args["seed"])
    if args.baseline:
        if run_args["tiny_random_bert"]:
            raise ValueError("Tiny random checkpoints have no pretrained baseline")
        revision = run_args.get("backbone_commit") or run_args["revision"]
        model, baseline_initialization = load_initial_model(run_args["model_name_or_path"],
                                                           model_args, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(run_args["model_name_or_path"], revision=revision)
    else:
        model, tokenizer = load_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    vectors = {split: {field: embed(model, tokenizer, [row[field] for row in rows], args.batch_size, args.max_length)
                       for field in ("utterance", "pattern")} for split, rows in splits.items()}
    labels = {split: [row["intent"] for row in rows] for split, rows in splits.items()}

    def score(coefficient, split):
        mix = lambda split: ((1 - coefficient) * vectors[split]["utterance"]
                             + coefficient * vectors[split]["pattern"])
        return accuracy(mix("train"), mix(split), labels["train"], labels[split], args.batch_size)

    scores = {c: score(c, "valid") for c in sorted(set(args.coefficients + [0.0]))}
    chosen = select_coefficient(scores)
    results = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "checkpoint_model_args": model_args, "original_counts": original_counts,
        "training_contamination_audit": training_audit,
        "initialization": baseline_initialization if args.baseline else run_args.get("initialization"),
        "evaluated_model": "initialized TaDSE with identical inference settings" if args.baseline else "trained TaDSE checkpoint",
        "overlap_exclusions": exclusions, "sampled_counts": {key: len(value) for key, value in splits.items()},
        "sampled_ids": {key: [row["id"] for row in rows] for key, rows in splits.items()},
        "validation_accuracy": scores, "selected_coefficient": chosen,
        "test_accuracy": score(chosen, "test"), "plain_test_accuracy": score(0.0, "test"),
        "compression_source": "https://arxiv.org/html/2305.14299v3#S3.SS3",
        "limitations": ["Reconstructed evaluation, not historical benchmark parity.",
                     "Casefolded whitespace-normalized text overlaps excluded before sampling.",
                     "Baseline uses TaDSE inference pooling settings; it is not automatically standard SimCSE CLS-only.",
                     "Subsampled runs are pilots; reference sampling may omit intent classes."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: results[key] for key in ("sampled_counts", "overlap_exclusions",
                     "validation_accuracy", "selected_coefficient", "test_accuracy", "plain_test_accuracy")}, indent=2))
    return results


if __name__ == "__main__":
    main()
