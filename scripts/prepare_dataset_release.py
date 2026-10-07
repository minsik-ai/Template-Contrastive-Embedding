#!/usr/bin/env python3
"""Audit a reconstruction against rounded paper Table 1 and package it offline.

Never uploads or creates Hub repositories. Exit 2 means the private-upload gate
failed; verification.json and a draft README.md are still written. An existing
output directory is never reused. Publication requires a separate authorized
private-only upload, using a freshly successful audit and the packaged files.

Optional manifest evidence per source:
  original_examples: {"output": "snips_original.jsonl", "sha256": "..."}
Each JSONL row must contain nonempty utterance and generic masked template strings and
slots: [{"type": "city", "value": "Paris"}, ...]. These must be the originals
actually used by generation, before top-k substitution. Missing evidence blocks
the gate; top-k vocabulary cannot establish the full original value count.
"""
import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

PAPER_URL = "https://arxiv.org/html/2305.14299v3#S2.T1"
PAPER = {
    "snips": {"top_k": 5, "slots": "39", "values": "11.9K", "templates": "7.4K", "originals": "13.1K", "utterances": "163K"},
    "atis": {"top_k": 2, "slots": "41", "values": "0.6K", "templates": "3.3K", "originals": "4.5K", "utterances": "239K"},
    "massive": {"top_k": 3, "slots": "55", "values": "4.0K", "templates": "10.3K", "originals": "11.5K", "utterances": "78K"},
    "hwu": {"top_k": 3, "slots": "56", "values": "6.0K", "templates": "16.6K", "originals": "19.2K", "utterances": "133K"},
    "clinc": {"top_k": 5, "slots": "17", "values": "1.7K", "templates": "15.3K", "originals": "15.3K", "utterances": "220K"},
}
TOTAL = {"slots": "208", "values": "24.2K", "templates": "52.9K", "originals": "63.6K", "utterances": "834K"}
DEFINITIONS = {
    "originals": "Number of original evidence rows, retaining original multiplicity.",
    "slots": "Distinct typed slot names in original evidence.",
    "values": "Distinct exact slot value strings across all slot types before top-k selection.",
    "templates": "Distinct exact generic masked template strings in original evidence.",
    "utterances": "CSV data rows, retaining the generation policy's multiplicity.",
    "totals": "Sum of per-source counts, not a cross-source deduplication.",
}


def rounded_interval(display):
    """Explicit nearest-unit half-up interpretation; upper boundary is excluded."""
    if not display.endswith("K"):
        n = int(display)
        return n, n + 1
    number = Decimal(display[:-1])
    unit = Decimal(10) ** number.as_tuple().exponent * 1000
    center = number * 1000
    return int(center - unit / 2), int(center + unit / 2)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Artifact output must be a nonempty relative path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact output escapes manifest directory")
    if not path.is_file():
        raise ValueError(f"Missing artifact: {name}")
    return path


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["utterance", "pattern"]:
            raise ValueError(f"{path.name}: expected exactly utterance,pattern columns")
        for line, row in enumerate(reader, 2):
            if None in row or any(not (row.get(key) or "").strip() for key in ("utterance", "pattern")):
                raise ValueError(f"{path.name}:{line}: malformed or empty CSV row")
            yield row


def feed_digest(digest, row):
    digest.update(json.dumps([row["utterance"], row["pattern"]], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8") + b"\n")


def original_statistics(path):
    slots, values, templates = set(), set(), set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            row = json.loads(text)
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                               for k in ("utterance", "template")):
                raise ValueError(f"Original evidence line {line}: empty utterance/template")
            if not isinstance(row.get("slots"), list):
                raise ValueError(f"Original evidence line {line}: slots must be a list")  # noqa: TRY004 - malformed external evidence is a validation error
            for slot in row["slots"]:
                if not isinstance(slot, dict) or any(not isinstance(slot.get(k), str) or not slot[k].strip()
                                                     for k in ("type", "value")):
                    raise ValueError(f"Original evidence line {line}: invalid slot")
                slots.add(slot["type"])
                values.add(slot["value"])
            templates.add(row["template"])
            count += 1
    return {"originals": count, "slots": len(slots), "values": len(values), "templates": len(templates)}


def add_check(report, name, passed, **evidence):
    report["checks"].append(dict(name=name, passed=bool(passed), **evidence))


def count_check(report, name, actual, display):
    low, high = rounded_interval(display)
    add_check(report, name, type(actual) is int and low <= actual < high,
              actual=actual, paper_display=display, lower_inclusive=low, upper_exclusive=high)


def verified_file(report, root, record, name):
    try:
        if not isinstance(record, dict):
            raise ValueError("Missing artifact evidence record")  # noqa: TRY004 - malformed external evidence is a validation error
        path = relative_file(root, record.get("output"))
        actual = sha256(path)
        add_check(report, name + ".sha256", actual == record.get("sha256"),
                  actual=actual, expected=record.get("sha256"), path=str(path))
        return path
    except (OSError, ValueError) as error:
        add_check(report, name + ".available", False, error=str(error))
        return None


def audit(manifest_path):
    report = {"schema_version": 1, "eligible_for_private_upload": False, "packaged": False,
                  "publication_scope": "private only; this script never uploads",
                  "paper": PAPER_URL, "rounding_policy": "Nearest displayed unit, half-up; [lower, upper). Exact paper counts and rounding convention are unavailable.",
                  "statistic_definitions": DEFINITIONS, "checks": [], "observed": {}, "dataset_files": []}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), dict):
            raise ValueError("Manifest requires a sources object")  # noqa: TRY004 - malformed external evidence is a validation error
    except (OSError, ValueError) as error:
        add_check(report, "manifest.readable", False, error=str(error))
        return report
    report["manifest"] = {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}
    root = manifest_path.resolve().parent
    sources = manifest["sources"]
    add_check(report, "sources.exact_membership", set(sources) == set(PAPER), actual=sorted(sources), expected=list(PAPER))
    combined = manifest.get("combined", {})
    order = combined.get("source_order", []) if isinstance(combined, dict) else []
    order_ok = isinstance(order, list) and all(isinstance(x, str) for x in order) and len(order) == len(PAPER) and set(order) == set(PAPER)
    add_check(report, "combined.source_order", order_ok, actual=order)
    paths = {}
    for source, targets in PAPER.items():
        record = sources.get(source)
        if not isinstance(record, dict):
            add_check(report, source + ".present", False)
            continue
        add_check(report, source + ".prepared", record.get("status") == "prepared", actual=record.get("status"))
        add_check(report, source + ".top_k", type(record.get("top_k")) is int and record["top_k"] == targets["top_k"],
                  actual=record.get("top_k"), expected=targets["top_k"])
        add_check(report, source + ".uncapped", "max_combinations_per_example" in record and
                  record["max_combinations_per_example"] is None and record.get("examples_truncated_by_cap") == 0,
                  cap=record.get("max_combinations_per_example"), truncated=record.get("examples_truncated_by_cap"))
        inputs = record.get("inputs")
        add_check(report, source + ".inputs.present", isinstance(inputs, list) and bool(inputs))
        for index, item in enumerate(inputs if isinstance(inputs, list) else []):
            try:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ValueError("Invalid input provenance record")  # noqa: TRY004 - malformed external evidence is a validation error
                input_path = Path(item["path"])
                if not input_path.is_absolute():
                    input_path = root / input_path
                actual = sha256(input_path)
                add_check(report, f"{source}.input.{index}.sha256", actual == item.get("sha256"),
                          path=str(input_path), actual=actual, expected=item.get("sha256"))
            except (OSError, ValueError) as error:
                add_check(report, f"{source}.input.{index}.available", False, error=str(error))
        path = verified_file(report, root, record, source + ".csv")
        observed = {}
        if path is not None:
            try:
                count = sum(1 for _ in csv_rows(path))
                observed["utterances"] = count
                add_check(report, source + ".declared_rows", type(record.get("output_rows")) is int and count == record["output_rows"],
                          actual=count, expected=record.get("output_rows"))
                paths[source] = path
                report["dataset_files"].append({"source": source, "path": str(path), "rows": count, "sha256": sha256(path)})
            except (OSError, ValueError, csv.Error) as error:
                add_check(report, source + ".csv.valid", False, error=str(error))
        original_path = verified_file(report, root, record.get("original_examples"), source + ".original_examples")
        if original_path is not None:
            try:
                observed.update(original_statistics(original_path))
                add_check(report, source + ".declared_originals", type(record.get("train_examples")) is int and
                          observed["originals"] == record["train_examples"], actual=observed["originals"], expected=record.get("train_examples"))
            except (OSError, ValueError) as error:
                add_check(report, source + ".original_examples.valid", False, error=str(error))
        else:
            report.setdefault("manifest_only_original_counts", {})[source] = record.get("train_examples")
        for metric, display in targets.items():
            if metric != "top_k":
                count_check(report, source + ".paper." + metric, observed.get(metric), display)
        report["observed"][source] = observed
    combined_path = verified_file(report, root, combined, "combined.csv")
    if combined_path is not None:
        try:
            count, actual_digest = 0, hashlib.sha256()
            for row in csv_rows(combined_path):
                count += 1
                feed_digest(actual_digest, row)
            count_check(report, "combined.paper.utterances", count, TOTAL["utterances"])
            add_check(report, "combined.declared_rows", type(combined.get("output_rows")) is int and count == combined["output_rows"],
                      actual=count, expected=combined.get("output_rows"))
            if order_ok and len(paths) == len(PAPER):
                expected_digest = hashlib.sha256()
                for source in order:
                    for row in csv_rows(paths[source]):
                        feed_digest(expected_digest, row)
                add_check(report, "combined.matches_sources", actual_digest.digest() == expected_digest.digest(),
                          actual=actual_digest.hexdigest(), expected=expected_digest.hexdigest())
            else:
                add_check(report, "combined.matches_sources", False, error="Missing sources or invalid source order")
        except (OSError, ValueError, csv.Error) as error:
            add_check(report, "combined.csv.valid", False, error=str(error))
    for metric, display in TOTAL.items():
        counts = [report["observed"].get(source, {}).get(metric) for source in PAPER]
        actual = sum(counts) if all(type(value) is int for value in counts) else None
        count_check(report, "total.paper." + metric, actual, display)
    report["eligible_for_private_upload"] = bool(report["checks"]) and all(c["passed"] for c in report["checks"])
    return report


def package(report, output):
    """Stream independently audited rows into one Parquet file per source."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = output / "data"
    data.mkdir()
    schema = pa.schema([(key, pa.string()) for key in ("utterance", "pattern", "source")])
    report["parquet_files"] = []
    for record in report["dataset_files"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Source changed after audit: {path}")
        target = data / (record["source"] + "-train.parquet")
        count, batch = 0, []
        with pq.ParquetWriter(target, schema, compression="zstd") as writer:
            for row in csv_rows(path):
                batch.append(dict(row, source=record["source"]))
                count += 1
                if len(batch) >= 10000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        if count != record["rows"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Source changed during packaging: {path}")
        actual = pq.read_metadata(target).num_rows
        if actual != count:
            raise ValueError(f"Parquet row mismatch: {target}")
        report["parquet_files"].append({"path": str(target.relative_to(output)), "rows": count, "sha256": sha256(target)})
    report["packaged"] = True


def write_card(report, output):
    eligible = report["eligible_for_private_upload"] and report["packaged"]
    lines = ["---", "language:", "- en", "tags:", "- synthetic", "- sentence-similarity"]
    if eligible:
        lines += ["configs:", "- config_name: combined", "  default: true", "  data_files:", "  - split: train", "    path: data/*-train.parquet"]
        for source in PAPER:
            lines += [f"- config_name: {source}", "  data_files:", "  - split: train", f"    path: data/{source}-train.parquet"]
    lines += ["---", "", "# TaDSE reconstructed augmentation — private release draft", "",
              "Status: " + ("rounded Table 1 checks passed; packaged for a separately authorized private upload." if eligible else
                            "UNVERIFIED / BLOCKED. Do not upload this draft or claim paper replication."), "",
              f"Reference: [paper Table 1]({PAPER_URL}).", "",
              ("Counts matching rounded published intervals do not prove identical examples, historical recipe, or model accuracy. "
              "The paper's exact counts and rounding rule have not been recovered. The audit uses explicit nearest-unit half-up intervals."), "",
              ("The CSV schema is utterance/pattern; packaged Parquet adds source. Generated multiplicity is preserved. "
              "No truncation, resampling, or altered top-k is applied to fit targets."), "",
              ("Source datasets: SNIPS, ATIS, MASSIVE, HWU64, CLINC150. Original source licenses and redistribution terms "
              "have not been resolved by this packaging step; no blanket license is asserted. This draft is restricted to private use."), "",
              "## Audit", "", "See verification.json for hashes, actual counts, counting definitions, intervals and every failed check.", ""]
    for source in PAPER:
        count = report["observed"].get(source, {}).get("utterances")
        lines.append(f"- {source}: generated rows {count if count is not None else 'unavailable'}; paper {PAPER[source]['utterances']}.")
    lines += ["", "## Limitations", "",
              ("Reconstruction provenance and split filtering follow the generation manifest. Full original-example evidence "
              "is required to audit original row counts, slots, full values, and generic masked templates. Top-k vocabularies alone are insufficient."),
              ("Generic masked-template and unique value-string counting are explicit reconstruction definitions; numerical agreement "
              "does not recover the authors' undocumented counting conventions."), ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory must be new; existing audit/release artifacts are preserved")
    args.output_dir.mkdir(parents=True)
    report = audit(args.manifest)
    if report["eligible_for_private_upload"]:
        try:
            package(report, args.output_dir)
        except (ImportError, OSError, ValueError) as error:
            add_check(report, "packaging.completed", False, error=str(error))
            report["eligible_for_private_upload"] = False
    report["failures"] = [check for check in report["checks"] if not check["passed"]]
    write_card(report, args.output_dir)
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_for_private_upload": report["eligible_for_private_upload"],
                         "packaged": report["packaged"], "failed_checks": len(report["failures"]),
                         "verification": str(args.output_dir / "verification.json")}, indent=2))
    return 0 if report["eligible_for_private_upload"] and report["packaged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
