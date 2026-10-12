#!/usr/bin/env python3
"""Qwen utterance-only evaluation with recovered loaders and cosine 1-NN.

This is a documented Qwen adaptation, not a recovered historical Qwen launch.
No instructions, templates, NER, or semantic compression are used at inference.
"""
import argparse
import builtins
import contextlib
import hashlib
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

from evaluate_recovered import HASHES, ROOT, load_rows, original_functions, sha256

TASKS = ("snips", "atis", "massive", "hwu", "clinc")


def last_token_pool(hidden, attention_mask):
    """Select the last real token, including variable-length left-padded batches."""
    import torch
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [batch, length, width] and matching attention mask")
    if not torch.isfinite(hidden).all():
        raise ValueError("Nonfinite hidden states")
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("Attention mask must be binary")
    if not attention_mask.bool().any(dim=1).all():
        raise ValueError("All-padding input is not an utterance")
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
    indices = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def truncate_normalize(embeddings, dimensions):
    import torch
    if embeddings.ndim != 2 or not 0 < dimensions <= embeddings.shape[1]:
        raise ValueError("Dimensions must fit the native embedding width")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    vectors = embeddings[:, :dimensions].float()
    norms = vectors.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Cannot normalize zero or nonfinite embedding norms")
    return vectors / norms


def encode(model, tokenizer, texts, batch_size, max_length, dimensions):
    import torch
    vectors = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True,
                              truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            batch["position_ids"] = (batch["attention_mask"].long().cumsum(dim=-1) - 1).clamp(min=0)
            hidden = model(**batch).last_hidden_state
            vectors.append(truncate_normalize(last_token_pool(hidden, batch["attention_mask"]),
                                              dimensions).cpu())
    if not vectors:
        raise ValueError("Cannot encode an empty split")
    return torch.cat(vectors).numpy()


def checkpoint_hashes(path):
    result = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and ".cache" not in file.relative_to(path).parts:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            result[str(file.relative_to(path))] = digest.hexdigest()
    return result


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    if not args.checkpoint.is_dir():
        raise ValueError("--checkpoint must be a downloaded local snapshot or encoder export")
    if args.batch_size < 1 or args.max_length < 1 or args.dimensions < 1:
        raise ValueError("Batch size, max length and dimensions must be positive")
    if "clinc" in args.tasks and args.clinc_splits_json is None:
        raise ValueError("CLINC requires --clinc-splits-json")
    import numpy as np
    import torch
    import transformers
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoModel, AutoTokenizer

    namespace = original_functions({"np": np, "os": os, "re": re, "json": json,
                                    "cosine_similarity": cosine_similarity})
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        raise ValueError("Checkpoint tokenizer must define its padding token")
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True,
                                      torch_dtype=torch.float32, attn_implementation="eager")
    if model.config.hidden_size < args.dimensions:
        raise ValueError("Requested dimensions exceed model hidden size")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    report = {"protocol": "qwen-recovered-loaders-utterance-only", "source_commit": "48d5f94",
              "original_sources_sha256": HASHES, "evaluator_sha256": sha256(__file__),
              "loader_wrapper_sha256": sha256(ROOT / "scripts/evaluate_recovered.py"),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hashes(args.checkpoint),
              "settings": {"pooler": "last_nonpadding_token", "padding_side": "left",
                           "native_dimensions": model.config.hidden_size, "dimensions": args.dimensions,
                           "dimension_selection": "first_dimensions_then_l2_normalize", "dtype": "float32",
                           "position_ids": "attention_mask_cumsum_minus_one_clamped_zero",
                           "attention_implementation": "eager", "device": str(next(model.parameters()).device),
                           "max_length": args.max_length, "batch_size": args.batch_size,
                           "instruction": None, "prompt": None, "aux_ratio": 0,
                           "search_target": "train", "utterance_topk": 1,
                           "torch_version": torch.__version__, "transformers_version": transformers.__version__},
              "historical_qwen_launch_recovered": False,
              "adaptations": ["Unchanged recovered data loaders, text preprocessing and _search",
                              "HWU fixed filenames remapped under data-root, no split substitution",
                              "CLINC plus JSON utterances only, no NER or pattern inference",
                              "Qwen last-token embedding replaces BERT pooler; no instructions",
                              "Explicit max length and truncation are reconstruction settings"],
              "finite_checks": {"hidden_states": True, "embeddings": True, "positive_norms": True},
              "inputs": {}, "tasks": {}, "complete": False}
    real_open = builtins.open
    data_root = args.data_root.resolve()

    def tracked_open(file, mode="r", *a, **kw):
        if isinstance(file, (str, os.PathLike)) and "r" in mode:
            if str(file) in ("data/massive/hwu64_valid_raw", "data/massive/hwu64_test_raw"):
                file = data_root / Path(file).name
            path = Path(file).resolve()
            if path.is_file() and str(path) not in report["inputs"]:
                report["inputs"][str(path)] = sha256(path)
        return real_open(file, mode, *a, **kw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if task == "clinc":
            report["inputs"][str(args.clinc_splits_json.resolve())] = sha256(args.clinc_splits_json)
        with real_open(args.output.with_suffix(f".{task}.log"), "w") as log, \
                patch("builtins.open", tracked_open), contextlib.redirect_stdout(log):
            train, valid, test = load_rows(task, data_root, args.clinc_splits_json, namespace)
        if not train or not test:
            raise ValueError(f"Empty reference/test split for {task}")
        target_vecs = encode(model, tokenizer, [row["utt"] for row in train], args.batch_size,
                             args.max_length, args.dimensions)
        predictions = []
        for start in range(0, len(test), args.batch_size):
            query_vecs = encode(model, tokenizer, [row["utt"] for row in test[start:start + args.batch_size]],
                                args.batch_size, args.max_length, args.dimensions)
            matches, scores = namespace["_search"](query_vecs, target_vecs, train, topk_utterances=1)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite nearest-neighbour scores")
            predictions.extend(row["intent"] for row in matches)
        labels = [row["intent"] for row in test]
        correct = sum(a == b for a, b in zip(labels, predictions))
        result = {"accuracy": 100 * correct / len(test), "correct": correct,
                  "test_count": len(test), "reference_count": len(train), "validation_count": len(valid),
                  "labels": labels, "predictions": predictions}
        report["tasks"][task] = result
        report["mean_accuracy"] = sum(x["accuracy"] for x in report["tasks"].values()) / len(report["tasks"])
        report["complete"] = set(report["tasks"]) == set(TASKS)
        report["five_dataset_mean"] = report["mean_accuracy"] if report["complete"] else None
        if {"snips", "atis"}.issubset(report["tasks"]):
            report["table6_snips_atis_mean"] = sum(report["tasks"][key]["accuracy"] for key in ("snips", "atis")) / 2
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{task}: {result['accuracy']:.6f}% ({correct}/{len(test)})", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/massive")
    parser.add_argument("--clinc-splits-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
