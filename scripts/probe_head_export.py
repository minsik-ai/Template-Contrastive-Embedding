#!/usr/bin/env python3
"""Test a fixed trained-head export on validation, then report test diagnostics.

No training or head selection takes place. Both available validation sets are
evaluated first. All five test scores are then reported for the same exported
head, irrespective of which validation score is higher.
"""
import argparse
import json
from pathlib import Path

from evaluate_table2 import (
    PAPER,
    checkpoint_hashes,
    nearest_cosine,
    summarize_predictions,
)
from export_table2_head import export_head
from prepare_table2_benchmarks import TEST_COUNTS, sha256
from reproduce_train import load_checkpoint


def embed(encoder, tokenizer, rows, task, batch_size, max_length, device):
    import torch
    parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [row["utterance"] for row in rows[start:start + batch_size]]
            if task != "massive":
                texts = [text.replace(",", " ") for text in texts]
            batch = tokenizer(texts, padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            vectors = encoder(**{k: v.to(device) for k, v in batch.items()}, return_dict=True).pooler_output
            parts.append(vectors.float().cpu())
    return torch.cat(parts)


def score(reference_vectors, query_vectors, refs, queries, device):
    indices, similarities = nearest_cosine(reference_vectors, query_vectors, device=device)
    return summarize_predictions(queries, refs, indices, similarities)


def compare_source(checkpoint, benchmarks, original):
    if original.get("pooling") != "bert-pooler":
        raise ValueError("Existing comparison must use the original BERT pooler")
    if original["benchmark_sha256"] != sha256(benchmarks):
        raise ValueError("Existing test results use different benchmark content")
    if original["checkpoint_sha256"] != checkpoint_hashes(checkpoint):
        raise ValueError("Checkpoint changed since the original evaluation")
    args = original["args"]
    if args.get("baseline"):
        raise ValueError("Compare with the trained checkpoint evaluation, not its initialization baseline")
    if args.get("max_length") != 128 or args.get("text_preprocessing") != "legacy":
        raise ValueError("Existing evaluation has different tokenization settings")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmarks", required=True, type=Path)
    parser.add_argument("--original-test-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--paper-row", required=True, choices=list(PAPER))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior exports; choose a fresh output directory")
    if args.output_dir.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError("Probe output must be outside the source checkpoint directory")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    original = json.loads(args.original_test_results.read_text())
    compare_source(args.checkpoint, args.benchmarks, original)
    benchmark = json.loads(args.benchmarks.read_text())
    validation_tasks = [name for name in TEST_COUNTS if benchmark["tasks"][name]["splits"]["valid"]]
    if not validation_tasks:
        raise ValueError("No validation examples available for the first-stage comparison")
    for task, expected in TEST_COUNTS.items():
        if len(benchmark["tasks"][task]["splits"]["test"]) != expected:
            raise ValueError(f"Unexpected test population for {task}")
    args.output_dir.mkdir(parents=True)
    # Keep both sides of the exact export-equivalence probe on CPU; scoring uses
    # the requested device below, without conflating CPU/GPU rounding with export.
    audit = export_head(args.checkpoint, args.output_dir / "model", device="cpu")

    from transformers import AutoModel, AutoTokenizer
    original_model, tokenizer = load_checkpoint(args.checkpoint)
    original_model.to(args.device).eval()
    exported = AutoModel.from_pretrained(args.output_dir / "model", local_files_only=True).to(args.device).eval()
    exported_tokenizer = AutoTokenizer.from_pretrained(args.output_dir / "model", local_files_only=True)
    results = {
        "hypothesis": "Replace only the inference pooler with the trained contrastive MLP",
        "selection": "Fixed export on every task; no validation/test-based head or checkpoint selection",
        "benchmark_sha256": sha256(args.benchmarks),
        "original_results_sha256": sha256(args.original_test_results),
        "checkpoint_sha256": checkpoint_hashes(args.checkpoint),
        "export_audit": audit, "validation": {}, "test_diagnostics": {},
        "validation_unavailable": [name for name in TEST_COUNTS if name not in validation_tasks],
    }
    # Retain reference vectors so validation and test use precisely the same index.
    exported_references = {}
    for task in validation_tasks:
        splits = benchmark["tasks"][task]["splits"]
        refs, queries = splits["train"], splits["valid"]
        old_refs = embed(original_model.bert, tokenizer, refs, task, args.batch_size, 128, args.device)
        old_queries = embed(original_model.bert, tokenizer, queries, task, args.batch_size, 128, args.device)
        new_refs = embed(exported, exported_tokenizer, refs, task, args.batch_size, 128, args.device)
        new_queries = embed(exported, exported_tokenizer, queries, task, args.batch_size, 128, args.device)
        old_score = score(old_refs, old_queries, refs, queries, args.device)
        new_score = score(new_refs, new_queries, refs, queries, args.device)
        exported_references[task] = new_refs
        results["validation"][task] = {"original": old_score, "exported": new_score,
                                        "delta_percentage_points": new_score["accuracy"] - old_score["accuracy"]}
        print(json.dumps({"split": "valid", "task": task, "original": old_score["accuracy"],
                          "exported": new_score["accuracy"]}), flush=True)
    (args.output_dir / "validation.json").write_text(json.dumps(results, indent=2) + "\n")

    for task in TEST_COUNTS:
        splits = benchmark["tasks"][task]["splits"]
        refs, queries = splits["train"], splits["test"]
        new_refs = exported_references.get(task)
        if new_refs is None:
            new_refs = embed(exported, exported_tokenizer, refs, task, args.batch_size, 128, args.device)
        new_queries = embed(exported, exported_tokenizer, queries, task, args.batch_size, 128, args.device)
        metrics = score(new_refs, new_queries, refs, queries, args.device)
        metrics.update(original_accuracy=original["tasks"][task]["accuracy"],
                       paper_accuracy=PAPER[args.paper_row][task],
                       delta_to_original=metrics["accuracy"] - original["tasks"][task]["accuracy"],
                       delta_to_paper=metrics["accuracy"] - PAPER[args.paper_row][task])
        results["test_diagnostics"][task] = metrics
        print(json.dumps({"split": "test-diagnostic", "task": task,
                          "original": metrics["original_accuracy"], "exported": metrics["accuracy"]}), flush=True)
    results["macro_test_accuracy"] = sum(v["accuracy"] for v in results["test_diagnostics"].values()) / 5
    results["original_macro_test_accuracy"] = sum(v["original_accuracy"] for v in results["test_diagnostics"].values()) / 5
    if checkpoint_hashes(args.checkpoint) != results["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint changed during the probe")
    results["source_unchanged"] = True
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
