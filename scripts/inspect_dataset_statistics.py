#!/usr/bin/env python3
"""Attach auditable original-example records without changing augmentation."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

GENERATOR = Path(__file__).resolve().parents[1] / "data/massive/augment_data.py"
spec = importlib.util.spec_from_file_location("tadse_augmentation", GENERATOR)
augmentation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = augmentation
spec.loader.exec_module(augmentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--clinc-splits-json", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.manifest.parent / "release-manifest.json"
    names = manifest["combined"]["source_order"]
    targets = [output] + [output.parent / f"{name}_original_examples.jsonl" for name in names]
    if any(path.exists() for path in targets):
        raise FileExistsError("Statistics outputs already exist; preserve the prior audit")
    verified_inputs = {}
    for name in names:
        verified_inputs[name] = set()
        for source in manifest["sources"][name]["inputs"]:
            input_path = Path(source["path"])
            if not input_path.is_absolute():
                input_path = args.manifest.parent / input_path
            if augmentation.sha256(input_path) != source["sha256"]:
                raise ValueError(f"Original source changed: {source['path']}")
            verified_inputs[name].add((input_path.resolve(), source["sha256"]))
    loaded = {name: augmentation.load_source(name, args.data_root, args.clinc_splits_json)
              for name in names}
    for name in names:
        actual_inputs = {(path.resolve(), augmentation.sha256(path)) for path in loaded[name][2]}
        if actual_inputs != verified_inputs[name]:
            raise ValueError(f"Loaded sources differ from verified generation inputs: {name}")
    heldout = set().union(*(item[1] for item in loaded.values()))
    originals = {name: [row for row in loaded[name][0]
                        if augmentation.text_key(row.text) not in heldout] for name in names}
    for name, rows in originals.items():
        if len(rows) != manifest["sources"][name]["train_examples"]:
            raise ValueError(f"Original-example count does not match generation: {name}")
    for name, rows in originals.items():
        path = output.parent / f"{name}_original_examples.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                item = {"utterance": row.text, "template": row.pattern,
                        "slots": [{"type": kind, "value": value}
                                  for value, kind in row.parts if kind]}
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest["sources"][name]["original_examples"] = {
            "output": path.name, "sha256": augmentation.sha256(path)}
    manifest["statistics_provenance"] = {
        "generation_manifest": args.manifest.name,
        "generation_manifest_sha256": augmentation.sha256(args.manifest),
        "generator_sha256": augmentation.sha256(GENERATOR),
        "original_rows": "Effective training examples after existing generator's pooled held-out filtering",
        "values": "Unique value strings across original examples; not type/value pairs",
        "templates": "Unique generic masked template strings across original examples",
    }
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
