#!/usr/bin/env python3
"""Table 2 utterance-only evaluation using preserved 2023 loaders and SimCSE.

Original source files remain byte-identical. This wrapper extracts the original
non-CLINC loading statements, then runs original SimCSE.encode and _search.
CLINC reads official plus utterances directly, bypassing unused pattern CSVs.
No annotation inference, pattern embedding, or semantic compression is performed.
"""
import argparse
import ast
import builtins
import contextlib
import hashlib
import json
import os
import re
import runpy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    "accuracy_infer.py": "abc573b62ccecd203f9ff0454d82e47af446a508d7b2944d261b8c04829ef7d0",
    "simcse_tool.py": "d67a2fc6dec066ce7e85534e82f2cd23725592ebbf1ba1bd0b12bb3a4196fdde",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_functions(namespace):
    for name, expected in HASHES.items():
        if sha256(ROOT / "legacy" / name) != expected:
            raise ValueError(f"Recovered source changed: {name}")
    path = ROOT / "legacy/accuracy_infer.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def load_rows(task, data_root, clinc_json, namespace):
    if task == "clinc":
        raw = json.loads(clinc_json.read_text())
        # Original plus dataset order: in-domain rows followed by OOS rows.
        # String labels replace numeric class IDs; equality/accuracy is identical.
        splits = {split: [{"utt": namespace["_txt_proc"](text), "intent": intent}
                          for text, intent in raw[split] + raw[f"oos_{split}"]]
                  for split in ("train", "test")}
        return splits["train"], [], splits["test"]
    paths = {"snips": data_root / "snips", "atis": data_root,
             "massive": data_root / "massive_en-US.jsonl", "hwu": data_root / "hwu64_orig.csv"}
    path = ROOT / "legacy/accuracy_infer.py"
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = []
    for node in main.body:
        # Stop before the original post-loading count print and model creation.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args:
            arg = node.value.args[0]
            if isinstance(arg, ast.JoinedStr) and isinstance(arg.values[0], ast.Constant) and arg.values[0].value == "Train : ":
                break
        body.append(node)
    else:
        raise ValueError("Could not locate original loader boundary")
    local = dict(namespace, task="snips_full" if task == "snips" else task, data_path=str(paths[task]))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), local)  # noqa: S102
    return local["train_utts"], local["dev_utts"], local["test_utts"]


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    import numpy as np
    import sklearn.metrics as sklm
    from sklearn.metrics.pairwise import cosine_similarity

    namespace = original_functions({"np": np, "os": os, "re": re, "json": json,
                                    "cosine_similarity": cosine_similarity})
    simcse = runpy.run_path(str(ROOT / "legacy/simcse_tool.py"))["SimCSE"]
    model = simcse(str(args.checkpoint), pooler="cls")
    data_root = args.data_root.resolve()
    report = {"protocol": "recovered-48d5f94-utterance-only", "checkpoint": str(args.checkpoint.resolve()),
              "source_commit": "48d5f94", "original_sources_sha256": HASHES,
              "settings": {"pooler": "cls", "max_length": 128, "batch_size": 128,
                           "aux_ratio": 0, "search_target": "train", "utterance_topk": 1},
              "wrapper_adaptations": ["AST executes unchanged original loading statements for SNIPS/ATIS/MASSIVE/HWU",
                                      "CLINC official plus JSON supplies utterances offline in original row order; intent names replace IDs",
                                      "No CLINC pattern CSV loading or pattern inference: utterance-only Table 2",
                                      "Original SimCSE.encode and _search execute unchanged; accuracy uses sklearn as original",
                                      "HWU fixed filenames resolve under --data-root; no canonical split substitution",
                                      "Original model loop replaced by nonempty batches; no claim historical CLI was recovered"],
              "tasks": {}, "inputs": {}}
    real_open = builtins.open

    def tracked_open(file, mode="r", *a, **kw):
        if isinstance(file, (str, os.PathLike)) and "r" in mode:
            if str(file) in ("data/massive/hwu64_valid_raw", "data/massive/hwu64_test_raw"):
                file = data_root / Path(file).name
            p = Path(file).resolve()
            if p.is_file() and str(p) not in report["inputs"]:
                report["inputs"][str(p)] = sha256(p)
        return real_open(file, mode, *a, **kw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if task == "clinc":
            if args.clinc_splits_json is None:
                raise ValueError("CLINC requires --clinc-splits-json")
            report["inputs"][str(args.clinc_splits_json.resolve())] = sha256(args.clinc_splits_json)
        with real_open(args.output.with_suffix(f".{task}.log"), "w") as log, patch("builtins.open", tracked_open), contextlib.redirect_stdout(log):
            train, valid, test = load_rows(task, data_root, args.clinc_splits_json, namespace)
        if not train or not test:
            raise ValueError(f"Empty reference/test split for {task}")
        target_vecs = model.encode([row["utt"] for row in train], batch_size=128).detach()
        predictions = []
        for start in range(0, len(test), 128):
            query_vecs = model.encode([row["utt"] for row in test[start:start + 128]], batch_size=128).detach()
            matches, _ = namespace["_search"](query_vecs, target_vecs, train, topk_utterances=1)
            predictions.extend(row["intent"] for row in matches)
        labels = [row["intent"] for row in test]
        result = {"accuracy": float(sklm.accuracy_score(labels, predictions)) * 100,
                  "correct": sum(a == b for a, b in zip(labels, predictions)), "test_count": len(test),
                  "reference_count": len(train), "validation_count": len(valid),
                  "labels": labels, "predictions": predictions}
        report["tasks"][task] = result
        report["mean_accuracy"] = sum(x["accuracy"] for x in report["tasks"].values()) / len(report["tasks"])
        report["complete"] = set(report["tasks"]) == {"snips", "atis", "massive", "hwu", "clinc"}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{task}: {result['accuracy']:.6f}% ({result['correct']}/{result['test_count']})", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/massive")
    parser.add_argument("--clinc-splits-json", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=["snips", "atis", "massive", "hwu", "clinc"],
                        default=["snips", "atis", "massive", "hwu", "clinc"])
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
