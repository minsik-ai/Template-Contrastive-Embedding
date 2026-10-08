#!/usr/bin/env python3
"""Full-reference cosine 1NN for all five TaDSE Table 2 benchmarks.

Default bert-pooler/max_length128 reproduces the recovered inference path.
There is no semantic compression, validation tuning, or test-row filtering.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_table2_benchmarks import TEST_COUNTS, sha256
from reproduce_train import load_checkpoint, load_initial_model

PAPER = {
    "tadse": {"snips": 97.00, "atis": 89.70, "massive": 78.18, "hwu": 82.77, "clinc": 70.56},
    "tadse_mlp": {"snips": 96.29, "atis": 89.14, "massive": 79.15, "hwu": 82.29, "clinc": 72.49},
}


def summarize_predictions(queries, references, indices, similarities):
    counts, correct = Counter(), Counter()
    wrong = []
    for query, index, score in zip(queries, indices, similarities):
        nearest = references[index]
        label = query["intent"]
        counts[label] += 1
        if label == nearest["intent"]:
            correct[label] += 1
        else:
            wrong.append({"id": query["id"], "expected": label, "predicted": nearest["intent"],
                              "reference_id": nearest["id"], "cosine": float(score)})
    if not queries or len(indices) != len(queries) or len(similarities) != len(queries):
        raise ValueError("Predictions must align with every query")
    return {"accuracy": 100 * sum(correct.values()) / len(queries),
                "correct": sum(correct.values()), "total": len(queries),
                "per_intent": {label: {"correct": correct[label], "total": count,
                                        "accuracy": 100 * correct[label] / count} for label, count in sorted(counts.items())},
                "wrong_predictions": wrong}


def nearest_cosine(references, queries, *, query_chunk=256, reference_chunk=4096, device="cpu"):
    """Chunk both axes; strict greater-than keeps earliest reference on ties."""
    import torch
    import torch.nn.functional as F
    if len(references) == 0 or len(queries) == 0 or query_chunk < 1 or reference_chunk < 1:
        raise ValueError("Need nonempty vectors and positive chunk sizes")
    references = F.normalize(references.float(), dim=-1)
    queries = F.normalize(queries.float(), dim=-1)
    if not torch.isfinite(references).all() or not torch.isfinite(queries).all():
        raise ValueError("Non-finite embeddings")
    indices, similarities = [], []
    for start in range(0, len(queries), query_chunk):
        q = queries[start:start + query_chunk].to(device)
        best = torch.full((len(q),), -float("inf"), device=device)
        selected = torch.zeros(len(q), dtype=torch.long, device=device)
        for offset in range(0, len(references), reference_chunk):
            scores = q @ references[offset:offset + reference_chunk].to(device).T
            score, index = scores.max(dim=1)
            improved = score > best
            selected = torch.where(improved, index + offset, selected)
            best = torch.maximum(best, score)
        indices.extend(selected.cpu().tolist())
        similarities.extend(best.cpu().tolist())
    return indices, similarities


def embed(model, tokenizer, rows, *, pooling, batch_size, max_length, text_preprocessing):
    import torch
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [row["utterance"] for row in rows[start:start + batch_size]]
            if text_preprocessing == "legacy-commas":
                texts = [text.replace(",", " ") for text in texts]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(model.device) for key, value in tokens.items()}
            if pooling == "checkpoint":
                output = model(**tokens, sent_emb=True).pooler_output
            else:
                encoded = model.bert(**tokens, return_dict=True)
                if pooling == "bert-pooler":
                    output = encoded.pooler_output
                elif pooling == "cls-before-pooler":
                    output = encoded.last_hidden_state[:, 0]
                else:
                    output = model.mlp(encoded.last_hidden_state[:, 0])
            vectors.append(output.float().cpu())
    return torch.cat(vectors)


def checkpoint_hashes(directory):
    weights = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError("No checkpoint weight files found")
    paths = weights + [directory / name for name in ("config.json", "model_args.json", "run_args.json")]
    return {path.name: sha256(path) for path in paths if path.exists()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--baseline-model-name-or-path")
    p.add_argument("--baseline-revision")
    p.add_argument("--pooling", choices=["bert-pooler", "checkpoint", "cls-before-pooler", "custom-mlp"], default="bert-pooler")
    p.add_argument("--text-preprocessing", choices=["none", "legacy", "legacy-commas"], default="legacy",
                   help="Recovered legacy replaces commas except MASSIVE; legacy-commas applies it to every task")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--query-chunk", type=int, default=256)
    p.add_argument("--reference-chunk", type=int, default=4096)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tasks", nargs="+", choices=list(TEST_COUNTS), default=list(TEST_COUNTS))
    p.add_argument("--paper-row", choices=list(PAPER), default="tadse")
    p.add_argument("--allow-count-mismatch", action="store_true", help="Diagnostic raw HWU profile only; cannot claim headline count parity")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists; use a fresh result filename")
    if min(args.batch_size, args.query_chunk, args.reference_chunk) < 1 or args.max_length < 2:
        raise ValueError("Batch/chunk sizes and max length must be positive")
    if not args.baseline and (args.baseline_model_name_or_path or args.baseline_revision):
        raise ValueError("Baseline overrides require --baseline")
    benchmark = json.loads(args.benchmarks.read_text())
    run = json.loads((args.checkpoint / "run_args.json").read_text())
    model_args = json.loads((args.checkpoint / "model_args.json").read_text())
    from transformers import AutoTokenizer, set_seed
    set_seed(run.get("seed", 42))
    initialization = None
    if args.baseline:
        source = args.baseline_model_name_or_path or run["model_name_or_path"]
        revision = args.baseline_revision or run.get("backbone_commit") or run.get("revision", "main")
        model, initialization = load_initial_model(source, model_args, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        baseline_identity = {"source": source, "revision": getattr(model.config, "_commit_hash", None) or revision}
    else:
        model, tokenizer = load_checkpoint(args.checkpoint)
        baseline_identity = None
    model.to(args.device).eval()
    result = {"protocol": "Table 2 full original references and test, cosine 1NN, no semantic compression",
                  "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "benchmark_sha256": sha256(args.benchmarks), "checkpoint_sha256": checkpoint_hashes(args.checkpoint),
                  "benchmark_profile": benchmark.get("profile"), "hwu_profile": benchmark.get("hwu_profile"),
                  "baseline_identity": baseline_identity, "initialization": initialization,
                  "saved_model_args": model_args, "pooling": args.pooling, "tasks": {}}
    for task in dict.fromkeys(args.tasks):
        data = benchmark["tasks"][task]
        refs, queries = data["splits"]["train"], data["splits"]["test"]
        if len(queries) != TEST_COUNTS[task] and not args.allow_count_mismatch:
            raise ValueError(f"{task}: expected all {TEST_COUNTS[task]} test rows")
        kwargs = {"pooling": args.pooling, "batch_size": args.batch_size,
                      "max_length": args.max_length,
                      "text_preprocessing": ("none" if task == "massive" else "legacy-commas")
                      if args.text_preprocessing == "legacy" else args.text_preprocessing}
        ref_vectors = embed(model, tokenizer, refs, **kwargs)
        query_vectors = embed(model, tokenizer, queries, **kwargs)
        indices, scores = nearest_cosine(ref_vectors, query_vectors, query_chunk=args.query_chunk,
                                         reference_chunk=args.reference_chunk, device=args.device)
        metrics = summarize_predictions(queries, refs, indices, scores)
        metrics.update(reference_count=len(refs), source_audit=data["audit"], source_hashes=data["sources"],
                       paper_accuracy=PAPER[args.paper_row][task],
                       delta_percentage_points=metrics["accuracy"] - PAPER[args.paper_row][task])
        result["tasks"][task] = metrics
        print(json.dumps({"task": task, "accuracy": metrics["accuracy"], "total": len(queries), "references": len(refs)}), flush=True)
    result["macro_task_accuracy"] = sum(item["accuracy"] for item in result["tasks"].values()) / len(result["tasks"])
    result["all_five_tasks"] = set(result["tasks"]) == set(TEST_COUNTS)
    result["paper_test_counts_match"] = all(metrics["total"] == TEST_COUNTS[task]
                                             for task, metrics in result["tasks"].items())
    result["paper_macro_task_accuracy"] = sum(PAPER[args.paper_row][task] for task in result["tasks"]) / len(result["tasks"])
    result["limitations"] = ["Duplicates preserved and reported for historical count comparability.",
                              "Canonical input preparation does not prove identity with unavailable historical launch artifacts.",
                              "bert-pooler is the recovered inference path; checkpoint/custom-mlp use different head paths."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
