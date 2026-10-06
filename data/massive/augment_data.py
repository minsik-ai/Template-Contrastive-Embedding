"""Reconstruct training-only TaDSE slot augmentation with auditable provenance.

Uses only the standard library. Outputs headered utterance,pattern CSVs and a
manifest. This is a reconstruction, not a claim of bit-identical paper artifacts.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

TOP_K = {"snips": 5, "atis": 2, "massive": 3, "hwu": 3, "clinc": 5}
ANNOTATION = re.compile(r"\[([^\[\]:]+?)\s+:\s+([^\[\]]*?)\]")


def text_key(text):
    """Normalize comparisons only; emitted utterances keep their original text."""
    return " ".join(text.casefold().split())


@dataclass
class Example:
    text: str
    parts: list  # (literal text, None) or (slot value, typed slot name)

    @property
    def pattern(self):
        return "".join("{SLOT}" if kind else value for value, kind in self.parts)


def annotated(text, annotation):
    parts = []
    end = 0
    for match in ANNOTATION.finditer(annotation):
        parts.append((annotation[end:match.start()], None))
        parts.append((match.group(2), match.group(1).strip()))
        end = match.end()
    parts.append((annotation[end:], None))
    return Example(text, parts)


def offsets(row):
    text = row["text"]
    parts, end = [], 0
    for entity in sorted(row["entities"], key=lambda entity: entity["start"]):
        start, stop = entity["start"], entity["end"]
        if not 0 <= end <= start < stop <= len(text):
            raise ValueError(f"Invalid/overlapping entity offsets in {text!r}")
        parts.extend([(text[end:start], None), (text[start:stop], entity["entity"])])
        end = stop
    parts.append((text[end:], None))
    return Example(text, parts)


def read_json(path):
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8")), "utf-8"
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1")), "latin-1"


def snips_rows(path):
    content, encoding = read_json(path)
    examples = []
    for label in sorted(content):
        for row in content[label]:
            parts = [(part["text"], part.get("entity")) for part in row["data"]]
            examples.append(Example("".join(value for value, _ in parts), parts))
    return examples, encoding


def load_source(source, root, clinc_splits=None):
    """Return training examples, held-out strings, input files, and split notes."""
    files, notes, train, heldout = [], [], [], set()
    if source == "massive":
        files = [root / "massive_en-US.jsonl"]
        with files[0].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["partition"] == "train":
                    train.append(annotated(row["utt"], row["annot_utt"]))
                elif row["partition"] in {"dev", "valid", "validation", "test"}:
                    heldout.add(row["utt"])
                else:
                    raise ValueError(f"Unknown MASSIVE split: {row['partition']}")
    elif source == "snips":
        paths = sorted((root / "snips").glob("*/train_*_full.json"))
        if not paths:
            raise FileNotFoundError("No SNIPS train_*_full.json files")
        for path in paths:
            rows, encoding = snips_rows(path)
            train.extend(rows)
            files.append(path)
            notes.append(f"{path.name}: decoded {encoding}")
        for path in sorted((root / "snips").glob("*/validate_*.json")):
            rows, _ = snips_rows(path)
            heldout.update(row.text for row in rows)
            files.append(path)
        notes.append("Only bundled validation is available; separate paper test split is missing.")
        notes.append("Bundled full train has 13,784 rows versus paper 13.1k; Cartesian k=5 yields about 4.66M rows versus paper 163k. Exact historical filtering/augmentation implementation is missing; do not interpret this as count-faithful SNIPS replication.")
    elif source == "atis":
        for split in ("train", "test"):
            path = root / f"atis_{split}.json"
            content, _ = read_json(path)
            files.append(path)
            rows = content["rasa_nlu_data"]["common_examples"]
            if split == "train":
                for index, row in enumerate(rows):
                    try:
                        train.append(offsets(row))
                    except ValueError:
                        # One bundled row has 1-based offsets for the entire utterance.
                        # Repair only when the sole annotated value equals all text.
                        entities = row["entities"]
                        if len(entities) != 1 or entities[0].get("value") != row["text"]:
                            raise
                        repaired = dict(row, entities=[dict(entities[0], start=0, end=len(row["text"]))])
                        train.append(offsets(repaired))
                        notes.append(f"Repaired ATIS row {index} whole-utterance offsets: {entities[0]['start']}:{entities[0]['end']} -> 0:{len(row['text'])}; value equals full text.")
            else:
                heldout.update(row["text"] for row in rows)
        notes.append("No separate ATIS validation split is bundled.")
    elif source == "hwu":
        files = [root / "hwu64_orig.csv", root / "hwu64_valid_raw", root / "hwu64_test_raw"]
        for path in files[1:]:
            heldout.update(path.read_text(encoding="utf-8").splitlines())
        with files[0].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                if None in row or row["answer_annotation"] is None:
                    raise ValueError("Malformed HWU semicolon CSV row")
                text = row["answer_normalised"]
                if text and text not in heldout:
                    train.append(annotated(text, row["answer_annotation"]))
        notes.append("Training reconstructed as nonempty answer_normalised minus bundled valid/test exact strings; official train membership not supplied.")
    elif source == "clinc":
        if clinc_splits is None:
            raise FileNotFoundError("CLINC requires --clinc-splits-json: upstream split membership is unproven from log alone")
        files = [root / "clinc_raw_ner_spacy_lg_topk30.log", clinc_splits]
        content, _ = read_json(clinc_splits)
        allowed = {row[0] for name in ("train", "oos_train") for row in content.get(name, [])}
        heldout = {row[0] for name in ("val", "test", "oos_val", "oos_test") for row in content.get(name, [])}
        if not allowed or not heldout:
            raise ValueError("CLINC split JSON must include nonempty train and held-out splits")
        pending = None
        with files[0].open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Text : "):
                    if pending is not None:
                        raise ValueError("Unpaired CLINC Text record")
                    pending = re.sub(r" - \d+$", "", line.rstrip("\n")[7:])
                elif line.startswith("Ptn : "):
                    if pending is None:
                        raise ValueError("Unpaired CLINC Ptn record")
                    if pending not in allowed:
                        raise ValueError(f"CLINC log text not in verified train split: {pending!r}")
                    train.append(annotated(pending, line.rstrip("\n")[6:]))
                    pending = None
        if pending is not None or not train:
            raise ValueError("Incomplete/empty CLINC log")
        if collections.Counter(row.text for row in train) != collections.Counter(
            row[0] for name in ("train", "oos_train") for row in content.get(name, [])
        ):
            raise ValueError("CLINC log does not match full upstream training multiset")
        notes.append("Recovered existing spaCy annotations, verified against upstream train+oos_train multiset; original spaCy version unknown.")
    else:
        raise ValueError(source)
    heldout = {text_key(text) for text in heldout}
    before = len(train)
    train = [example for example in train if text_key(example.text) not in heldout]
    notes.append(f"Removed {before - len(train)} training rows overlapping casefolded, whitespace-normalized available held-out text before counting slot values.")
    return train, heldout, files, notes


def slot_vocabulary(examples, k):
    counts = collections.defaultdict(collections.Counter)
    for example in examples:
        for value, kind in example.parts:
            if kind:
                counts[kind][value] += 1
    # Counter.most_common preserves first occurrence on ties, matching legacy code.
    return {kind: counter.most_common(k) for kind, counter in sorted(counts.items())}


def variants(example, vocabulary, cap=None):
    choices = [[value for value, _ in vocabulary[kind]] if kind else [value]
               for value, kind in example.parts]
    products = itertools.product(*choices)
    if cap is not None:
        products = itertools.islice(products, cap)
    seen = {example.text}
    yield example.text
    for product in products:
        text = "".join(product)
        if text not in seen:
            seen.add(text)
            yield text


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, root, output, k=None, cap=None, clinc_splits=None, extra_heldout=None, loaded=None):
    k = TOP_K[source] if k is None else k
    if k < 1 or (cap is not None and cap < 1):
        raise ValueError("top-k and max-combinations must be positive")
    path = output / f"{source}_train.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh output directory")
    examples, heldout, files, notes = loaded if loaded is not None else load_source(source, root, clinc_splits)
    if extra_heldout is not None:
        heldout = {text_key(text) for text in heldout | extra_heldout}
        before = len(examples)
        examples = [example for example in examples if text_key(example.text) not in heldout]
        notes = notes + [f"Removed {before - len(examples)} further original rows overlapping other selected sources' held-out text before vocabulary construction."]
    vocabulary = slot_vocabulary(examples, k)
    output.mkdir(parents=True, exist_ok=True)
    count = excluded = capped = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance", "pattern"])
        for example in examples:
            combinations = 1
            for _, kind in example.parts:
                if kind:
                    combinations *= len(vocabulary[kind])
            capped += cap is not None and combinations > cap
            for text in variants(example, vocabulary, cap):
                if text_key(text) in heldout:
                    excluded += 1
                    continue
                writer.writerow([text, example.pattern])
                count += 1
    return {"status": "prepared", "source": source, "top_k": k,
            "train_examples": len(examples), "output_rows": count,
            "heldout_normalized_keys": len(heldout), "generated_heldout_matches_removed": excluded,
            "max_combinations_per_example": cap, "examples_truncated_by_cap": capped,
            "output": path.name, "sha256": sha256(path), "slot_vocabulary": vocabulary,
            "inputs": [{"path": str(p), "sha256": sha256(p)} for p in files], "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/augmentation"))
    parser.add_argument("--sources", nargs="+", choices=list(TOP_K), default=list(TOP_K))
    parser.add_argument("--top-k", type=int, help="Explicit override for every selected source")
    parser.add_argument("--max-combinations", type=int, help="Optional explicit smoke-test cap per example; never implicit")
    parser.add_argument("--clinc-splits-json", type=Path)
    args = parser.parse_args()
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.max_combinations is not None and args.max_combinations < 1:
        parser.error("--max-combinations must be positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be absent or empty; existing artifacts are never overwritten")
    manifest = {"schema_version": 1, "algorithm": "typed-slot training-frequency top-k Cartesian substitution",
                "tie_break": "first appearance in deterministic input order",
                "deduplication": "per original example only; original first",
                "format": "UTF-8 CSV with utterance,pattern header; commas preserved",
                "limitations": ["Reconstructed artifacts; historical exact output unavailable.",
                                "Casefolded, whitespace-normalized held-out matches excluded from vocabulary sources and output; semantic overlap cannot be guaranteed absent.",
                                "Missing held-out splits cannot be checked.",
                                "Normalization differs from historical comma-to-space replacement."],
                "sources": {}}
    if len(set(args.sources)) != len(args.sources):
        parser.error("--sources cannot contain duplicates")
    loaded_sources = {}
    global_heldout = set()
    for source in args.sources:
        try:
            loaded_sources[source] = load_source(source, args.data_root, args.clinc_splits_json)
            global_heldout.update(loaded_sources[source][1])
        except FileNotFoundError as error:
            manifest["sources"][source] = {"status": "missing", "reason": str(error)}
    manifest["heldout_filter"] = "Union of casefolded, whitespace-normalized held-out comparison keys from every available selected source, before vocabulary construction and after augmentation; differs from historical paper recipe."
    manifest["global_heldout_normalized_keys"] = len(global_heldout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "combined_train.csv"
    combined_count = 0
    with combined_path.open("x", encoding="utf-8", newline="") as handle:
        combined = csv.writer(handle)
        combined.writerow(["utterance", "pattern"])
        for source in args.sources:
            if source in loaded_sources:
                result = prepare(source, args.data_root, args.output_dir, args.top_k,
                                 args.max_combinations, args.clinc_splits_json,
                                 extra_heldout=global_heldout, loaded=loaded_sources[source])
                manifest["sources"][source] = result
                with (args.output_dir / result["output"]).open(encoding="utf-8", newline="") as source_handle:
                    reader = csv.reader(source_handle)
                    next(reader)
                    for row in reader:
                        combined.writerow(row)
                        combined_count += 1
            result = manifest["sources"][source]
            print(source, result["status"], result.get("output_rows", result.get("reason")), flush=True)
    manifest["combined"] = {"output": combined_path.name, "output_rows": combined_count,
                            "sha256": sha256(combined_path),
                            "source_order": [source for source in args.sources if source in loaded_sources]}
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
