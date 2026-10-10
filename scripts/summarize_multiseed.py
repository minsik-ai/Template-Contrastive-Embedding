#!/usr/bin/env python3
"""Summarize fixed, paired TaDSE seeds without selecting runs or combining models."""

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

CONDITIONS = ("legacy_top5", "all_intent_top2")
SEEDS = (42, 43, 44)
TASKS = ("snips", "atis", "massive", "hwu", "clinc")


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    runs = {}
    paths = set()
    for entry in manifest["runs"]:
        condition, seed = entry["condition"], entry["seed"]
        key = (condition, seed)
        if condition not in CONDITIONS or type(seed) is not int or seed not in SEEDS:
            raise ValueError(f"Unexpected condition/seed: {key}")
        if key in runs:
            raise ValueError(f"Duplicate condition/seed: {key}")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if path in paths:
            raise ValueError(f"Duplicate result path: {path}")
        paths.add(path)
        runs[key] = path
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(runs) != expected:
        raise ValueError(f"Manifest must contain both conditions for seeds {SEEDS}; missing {sorted(expected - set(runs))}")

    available, missing = {}, []
    benchmark_hash = manifest.get("benchmark_sha256")
    if benchmark_hash is not None and not valid_digest(benchmark_hash):
        raise ValueError("Invalid manifest benchmark_sha256")
    for (condition, seed), path in sorted(runs.items()):
        if not path.exists():
            missing.append({"condition": condition, "seed": seed, "path": str(path)})
            continue
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            missing.append({"condition": condition, "seed": seed, "path": str(path),
                            "reason": "Result JSON is incomplete or malformed; retry after the writer finishes"})
            continue
        digest = result["benchmark_sha256"]
        if not valid_digest(digest):
            raise ValueError(f"Invalid benchmark_sha256: {path}")
        if benchmark_hash is not None and digest != benchmark_hash:
            raise ValueError(f"Mismatching benchmark_sha256: {path}")
        benchmark_hash = digest
        audit = result["export_audit"]
        if result.get("source_unchanged") is not True or audit.get("source_unchanged") is not True:
            raise ValueError(f"Source was not verified unchanged: {path}")
        diff = audit.get("reload_max_abs_diff")
        if type(diff) not in (int, float) or diff != 0:
            raise ValueError(f"Export reload difference must be zero: {path}")
        tasks = result["test_diagnostics"]
        if set(tasks) != set(TASKS):
            raise ValueError(f"Expected exactly the five benchmark tasks: {path}")
        scores = {task: tasks[task]["accuracy"] for task in TASKS}
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in scores.values()):
            raise ValueError(f"Invalid accuracy: {path}")
        scores["macro"] = statistics.mean(scores.values())
        available[condition, seed] = scores

    paired_seeds = [seed for seed in SEEDS if all((condition, seed) in available for condition in CONDITIONS)]
    metrics = (*TASKS, "macro")
    conditions = {
        condition: {
            metric: stats([available[condition, seed][metric] for seed in paired_seeds])
            for metric in metrics
        }
        for condition in CONDITIONS
    }
    deltas = {
        str(seed): {metric: available[CONDITIONS[1], seed][metric] - available[CONDITIONS[0], seed][metric]
                    for metric in metrics}
        for seed in paired_seeds
    }
    return {
        "status": "incomplete" if missing else "complete",
        "manifest": str(manifest_path),
        "predetermined_seeds": list(SEEDS),
        "matched_seeds": paired_seeds,
        "missing_results": missing,
        "available_runs": len(available),
        "benchmark_sha256": benchmark_hash,
        "conditions": conditions,
        "paired_top2_minus_legacy": {
            "per_seed": deltas,
            "summary": {metric: stats([deltas[str(seed)][metric] for seed in paired_seeds]) for metric in metrics},
        },
        "units": "accuracy percentage points",
        "interpretation": "Descriptive matched-seed statistics only; no significance or ensemble improvement claim. "
                          "Sample standard deviation is null for fewer than two matched seeds.",
    }


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.manifest)
    atomic_write(args.output, result)
    return result


if __name__ == "__main__":
    main()
