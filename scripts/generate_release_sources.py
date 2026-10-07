#!/usr/bin/env python3
"""Run the unchanged augmenter separately per source, then combine its artifacts.

This candidate uses each source's own held-out exclusion set. Cross-source
held-out overlap is deliberately not filtered and must be audited before model
evaluation. No cap, top-k override, truncation, new NER, or upload is performed.
The paper-count publication gate remains a separate mandatory step.
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from prepare_dataset_release import PAPER, csv_rows, relative_file, sha256

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "data/massive/augment_data.py"
INSPECTOR = ROOT / "scripts/inspect_dataset_statistics.py"


def prefixed_record(record, source, source_dir):
    result = copy.deepcopy(record)
    path = relative_file(source_dir, result["output"])
    if sha256(path) != result["sha256"]:
        raise ValueError(f"Generated artifact hash mismatch: {path}")
    result["output"] = str(Path(source) / path.relative_to(source_dir))
    return result


def generate(output, data_root, clinc_splits, sources):
    if not sources or len(set(sources)) != len(sources) or any(source not in PAPER for source in sources):
        raise ValueError("Select distinct known sources")
    if output.exists():
        raise FileExistsError("Output directory must be new; previous candidates are preserved")
    output = output.resolve()
    data_root = data_root.resolve()
    clinc_splits = clinc_splits.resolve()
    code_hashes = {str(path): sha256(path) for path in (GENERATOR, INSPECTOR)}
    output.mkdir(parents=True)
    merged = {
        "schema_version": 1,
        "algorithm": "typed-slot training-frequency top-k Cartesian substitution",
        "heldout_filter": "Separate generation per source: only that source's casefolded, whitespace-normalized held-out strings are excluded. No cross-source exclusion.",
        "heldout_scope": "per_source",
        "limitations": [
            "Reconstructed candidate; matching published rounded counts is not guaranteed.",
            "Cross-source held-out overlap can remain; audit the actual training corpus before evaluating models.",
            "Missing source splits and historical annotation/recipe gaps remain unchanged.",
            "No source rows are truncated and no top-k/combinations settings are overridden.",
        ],
        "sources": {},
        "source_manifests": {},
        "orchestration": {"code_sha256": code_hashes, "source_order": sources},
    }
    immutable = {}
    for source in sources:
        source_dir = output / source
        subprocess.run([sys.executable, str(GENERATOR), "--data-root", str(data_root),
                        "--clinc-splits-json", str(clinc_splits), "--output-dir", str(source_dir),
                        "--sources", source], check=True)
        original_manifest = source_dir / "manifest.json"
        generated = json.loads(original_manifest.read_text(encoding="utf-8"))
        if generated.get("sources", {}).get(source, {}).get("status") != "prepared":
            raise ValueError(f"Requested source was not prepared: {source}")
        original_hash = sha256(original_manifest)
        subprocess.run([sys.executable, str(INSPECTOR), "--manifest", str(original_manifest),
                        "--data-root", str(data_root), "--clinc-splits-json", str(clinc_splits)], check=True)
        if sha256(original_manifest) != original_hash:
            raise ValueError(f"Original generation manifest changed: {source}")
        enriched_manifest = source_dir / "release-manifest.json"
        enriched = json.loads(enriched_manifest.read_text(encoding="utf-8"))
        if set(enriched["sources"]) != {source} or enriched["combined"]["source_order"] != [source]:
            raise ValueError(f"Unexpected per-source manifest membership: {source}")
        record = prefixed_record(enriched["sources"][source], source, source_dir)
        record["original_examples"] = prefixed_record(record["original_examples"], source, source_dir)
        merged["sources"][source] = record
        provenance = {}
        for key, path in (("generation", original_manifest), ("release", enriched_manifest)):
            value = sha256(path)
            immutable[path] = value
            provenance[key] = {"output": str(path.relative_to(output)), "sha256": value}
        merged["source_manifests"][source] = provenance
    count = 0
    combined = output / "combined_train.csv"
    with combined.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance", "pattern"])
        for source in sources:
            record = merged["sources"][source]
            path = relative_file(output, record["output"])
            before = count
            for row in csv_rows(path):
                writer.writerow([row["utterance"], row["pattern"]])
                count += 1
            if count - before != record["output_rows"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Source changed or row count differs: {source}")
    for path, expected in immutable.items():
        if sha256(path) != expected:
            raise ValueError(f"Individual manifest changed during orchestration: {path}")
    for path, expected in code_hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Generation/statistics code changed during orchestration: {path}")
    merged["combined"] = {"output": combined.name, "output_rows": count,
                          "sha256": sha256(combined), "source_order": sources}
    target = output / "release-manifest.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(target), "rows": count,
                      "heldout_scope": "per_source", "publication_gate_run": False}), flush=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--clinc-splits-json", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", choices=list(PAPER), default=list(PAPER))
    args = parser.parse_args(argv)
    generate(args.output_dir, args.data_root, args.clinc_splits_json, args.sources)


if __name__ == "__main__":
    main()
