#!/usr/bin/env python3
"""Replay recovered legacy augmentation and capture logical write records safely.

Run with PYTHONHASHSEED=42. SNIPS's repeated output truncation is preserved: only
its final intent remains in its trainable CSV. The recovered generator is never
modified. CLINC reuses cached annotation records and original AST-loaded helpers.
No new NER, normalization, held-out filtering, cap, or top-k override is added.
"""
import argparse
import ast
import builtins
import collections
import contextlib
import csv
import hashlib
import json
import os
import re
import runpy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_HASH = "d7f3daa57b58e79c391d86caa9e5cb1aebc8b98f44188dabef2307a59a1636a6"
EXTRACTOR_HASH = "f882fd4841c417d8b67e1a1d3e9a2f5fd70e78612ea79769bb41a18f44170174"
TOP_K = {"snips": 5, "atis": 2, "massive": 3, "hwu": 3, "clinc": 5}
SNIPS_LABELS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
HEADER = ["utterance", "pattern", "slot_count"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class WriteCapture:
    def __init__(self, raw_path, csv_path, task):
        self.raw_path, self.csv_path, self.task = raw_path, csv_path, task
        self.real_open = builtins.open
        self.write_attempts = self.retained_rows = self.output_opens = 0
        self.rows_per_open = []

    def open(self, file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == self.raw_path:
            if mode != "w":
                raise ValueError("Unexpected legacy output mode")
            self.output_opens += 1
            self.retained_rows = 0
            self.rows_per_open.append(0)
            return CapturedFile(self, self.real_open(file, mode, *args, **kwargs),
                                self.real_open(self.csv_path, "w", encoding="utf-8", newline=""))
        return self.real_open(file, mode, *args, **kwargs)


class CapturedFile:
    def __init__(self, capture, raw, sidecar):
        self.capture, self.raw, self.sidecar = capture, raw, sidecar
        self.writer = csv.writer(sidecar)
        self.writer.writerow(HEADER)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()
        self.sidecar.close()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def write(self, text):
        if not text.endswith("\n"):
            raise ValueError("Legacy write is not a complete record")
        fields = text[:-1].rsplit(",", 2)
        if len(fields) != 3:
            raise ValueError("Legacy record does not have three fields")
        count = int(fields[2])
        if text.count(",") != 2:
            raise ValueError("Ambiguous unescaped commas in legacy write; ingestion stopped without guessing")
        if count < 0 or fields[1].count("{SLOT}") != count:
            raise ValueError("Invalid legacy slot count")
        result = self.raw.write(text)
        self.writer.writerow(fields)
        self.capture.write_attempts += 1
        self.capture.retained_rows += 1
        self.capture.rows_per_open[-1] += 1
        return result


def source_inputs(source, root, clinc_splits):
    if source == "snips":
        return [root / "snips" / label / f"train_{label}_full.json" for label in SNIPS_LABELS]
    if source == "atis":
        return [root / "atis_train.json"]
    if source == "massive":
        return [root / "massive_en-US.jsonl"]
    if source == "hwu":
        return [root / "hwu64_orig.csv", root / "hwu64_valid_raw", root / "hwu64_test_raw"]
    return [root / "clinc_raw_ner_spacy_lg_topk30.log", clinc_splits]


def load_clinc_helpers(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_augment_slots", "_txt_proc", "_pattern_proc"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Missing original CLINC helper functions")
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - execute only named AST helpers from hash-verified source
    return namespace


def clinc_annotations(log_path, splits_path):
    content = json.loads(splits_path.read_text(encoding="utf-8"))
    expected = collections.Counter(row[0] for name in ("train", "oos_train") for row in content.get(name, []))
    rows, pending = [], None
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Text : "):
                if pending is not None:
                    raise ValueError("CLINC text has no matching annotation")
                pending = re.sub(r" - \d+$", "", line.rstrip("\n")[7:])
            elif line.startswith("Ptn : "):
                if pending is None:
                    raise ValueError("CLINC annotation has no matching text")
                rows.append((pending, line.rstrip("\n")[6:]))
                pending = None
    if pending is not None or not expected or collections.Counter(text for text, _ in rows) != expected:
        raise ValueError("CLINC cached text multiset differs from official train + oos_train")
    return rows


def replay_clinc(rows, helpers, output):
    entities = collections.defaultdict(list)
    for _, annotation in rows:
        for match in re.findall(r"\[[^\[\]]+]", annotation):
            parts = match.strip("[] ").split(" : ", 1)
            if len(parts) != 2:
                # The original slot book comes from doc.ents, not literal brackets
                # in input text (one historical example contains '[ country ]').
                continue
            entities[parts[0]].append(parts[1])
    # Preserve original defaultdict behavior: unknown literal bracket placeholders
    # produce no substitutions, after which the original utterance is retained.
    book = collections.defaultdict(list, {
        key: collections.Counter(values).most_common(5) for key, values in entities.items()})
    with open(output, "w") as handle:
        for utterance, annotation in rows:
            candidates = set(helpers["_augment_slots"](annotation, book) + [utterance])
            pattern = helpers["_pattern_proc"](annotation)
            handle.writelines(f"{helpers['_txt_proc'](candidate)},{helpers['_txt_proc'](pattern)},{pattern.count('{SLOT}')}\n" for candidate in candidates)


@contextlib.contextmanager
def working_directory(path):
    before = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(before)


def prepare(output, root, generator, extractor, clinc_splits, sources):
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Launch the interpreter with PYTHONHASHSEED=42 for reproducible legacy set iteration")
    if not sources or len(set(sources)) != len(sources) or any(source not in TOP_K for source in sources):
        raise ValueError("Sources must be distinct known datasets")
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; legacy artifacts are preserved")
    output, root, generator, extractor, clinc_splits = [path.resolve() for path in (output, root, generator, extractor, clinc_splits)]
    if sha256(generator) != GENERATOR_HASH:
        raise ValueError("Recovered generator does not match the verified 2023 source hash")
    if "clinc" in sources and sha256(extractor) != EXTRACTOR_HASH:
        raise ValueError("CLINC extractor does not match the verified original source hash")
    all_inputs = {path for source in sources for path in source_inputs(source, root, clinc_splits)} | {generator}
    if "clinc" in sources:
        all_inputs.add(extractor)
    before = {path: sha256(path) for path in all_inputs}
    module = runpy.run_path(str(generator), run_name="recovered_augmentation")
    output.mkdir(parents=True)
    manifest = {"schema_version": 1, "recipe": "recovered 2023 generator main, unchanged; cached CLINC annotations with original helpers",
                "original_generator": {"path": str(generator), "sha256": GENERATOR_HASH, "git_commit": "48d5f94"},
                "pythonhashseed": "42", "columns": HEADER, "source_order": sources, "sources": {},
                "limitations": ["SNIPS overwrites its output for each intent; only final SearchScreeningEvent rows survive.",
                                "Legacy raw bytes are preserved; CSV reserialization captures complete write records including embedded newlines.",
                                "No additional held-out filtering, cap, truncation, or annotation inference is applied.",
                                "CLINC uses cached historical annotations; NER runtime and model are not rerun.",
                                "Table 1 counts and Table 2 scores are not established by successful replay."]}
    for source in sources:
        raw, proper = output / f"{source}_legacy.csv", output / f"{source}_train.csv"
        capture = WriteCapture(raw, proper, source)
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), working_directory(root), patch("builtins.open", capture.open):
            if source == "clinc":
                rows = clinc_annotations(root / "clinc_raw_ner_spacy_lg_topk30.log", clinc_splits)
                replay_clinc(rows, load_clinc_helpers(extractor), raw)
            else:
                data_path = root / "snips" if source == "snips" else root / "massive_en-US.jsonl" if source == "massive" else root / "hwu64_orig.csv" if source == "hwu" else root
                module["main"](str(data_path), "train", TOP_K[source], str(raw), source)
        with raw.open("rb") as handle:
            physical_lines = sum(1 for _ in handle)
        manifest["sources"][source] = {"top_k": TOP_K[source], "write_attempts": capture.write_attempts,
            "retained_rows": capture.retained_rows, "rows_per_output_open": capture.rows_per_open,
            "output_opens": capture.output_opens, "raw_physical_lines": physical_lines,
            "raw": {"output": raw.name, "sha256": sha256(raw)}, "trainable": {"output": proper.name, "sha256": sha256(proper)},
            "inputs": [{"path": str(path), "sha256": before[path]} for path in source_inputs(source, root, clinc_splits)]}
    combined = output / "combined_train.csv"
    total = 0
    with combined.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for source in sources:
            count = 0
            with (output / manifest["sources"][source]["trainable"]["output"]).open(encoding="utf-8", newline="") as rows:
                reader = csv.reader(rows)
                if next(reader) != HEADER:
                    raise ValueError("Invalid captured CSV header")
                for row in reader:
                    if len(row) != 3:
                        raise ValueError("Captured CSV field count differs")
                    writer.writerow(row)
                    count += 1
            if count != manifest["sources"][source]["retained_rows"]:
                raise ValueError("Captured CSV count differs from retained write records")
            total += count
    for path, expected in before.items():
        if sha256(path) != expected:
            raise ValueError(f"Original input changed during replay: {path}")
    manifest["input_integrity_verified"] = True
    manifest["code_and_inputs"] = [{"path": str(path), "sha256": value} for path, value in sorted(before.items())]
    manifest["combined"] = {"output": combined.name, "rows": total, "sha256": sha256(combined)}
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "retained_rows": total,
                      "write_attempts": sum(record["write_attempts"] for record in manifest["sources"].values())}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--clinc-splits-json", required=True, type=Path)
    parser.add_argument("--original-generator", type=Path, default=ROOT / "legacy/augment_data.py")
    parser.add_argument("--clinc-extractor", type=Path, default=ROOT / "data/massive/extract_ptn_clinc.py")
    parser.add_argument("--sources", nargs="+", choices=list(TOP_K), default=list(TOP_K))
    args = parser.parse_args()
    prepare(args.output_dir, args.data_root, args.original_generator, args.clinc_extractor, args.clinc_splits_json, args.sources)


if __name__ == "__main__":
    main()
