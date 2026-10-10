#!/usr/bin/env python3
"""Prepare immutable utterance/intent splits for TaDSE Table 2 cosine 1NN.

Preserves full split counts and reports duplicate text rather than filtering it.
Recovered inputs use bundled SNIPS full/Rasa ATIS; the canonical profile uses
JointBERT instead. MASSIVE JSONL, HWU profile and CLINC-plus are explicit.
Fetching pins GitHub revisions and content hashes in lock files.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEST_COUNTS = {"snips": 700, "atis": 893, "massive": 2974, "hwu": 1076, "clinc": 5500}
PINNED_REVISIONS = {
    "monologg/JointBERT": "00324f60dda4a491b2913bc3065ae93966caea3f",
    "clinc/oos-eval": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
    "xliuhw/NLU-Evaluation-Data": "f6071b496b17d71e6eb43f543af0707f4ff30557",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text(text):
    return " ".join(text.casefold().split())


def row(text, intent, identity):
    if not isinstance(text, str) or not text.strip() or not str(intent).strip():
        raise ValueError(f"Empty utterance or intent: {identity}")
    return {"id": str(identity), "utterance": text, "intent": str(intent)}


def jointbert(root, task):
    splits, paths = {}, []
    for source, target in (("train", "train"), ("dev", "valid"), ("test", "test")):
        files = [root / task / source / name for name in ("seq.in", "seq.out", "label")]
        texts, slots, labels = [path.read_text().splitlines() for path in files]
        if len(texts) != len(slots) or len(texts) != len(labels):
            raise ValueError(f"Unaligned {task}/{source} utterance/slot/intent rows")
        if any(len(text.split()) != len(slot.split()) for text, slot in zip(texts, slots)):
            raise ValueError(f"Unaligned {task}/{source} BIO tokens")
        splits[target] = [row(text, label, f"{task}/{source}/{index}")
                          for index, (text, label) in enumerate(zip(texts, labels))]
        paths.extend(files)
    return splits, paths


def massive(path):
    splits = {name: [] for name in ("train", "valid", "test")}
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            split = {"dev": "valid"}.get(item["partition"], item["partition"])
            splits[split].append(row(item["utt"], item["intent"], f"massive/{item['id']}"))
    return splits, [path]


def clinc(path):
    data = json.loads(path.read_text())
    splits = {}
    for source, target in (("train", "train"), ("val", "valid"), ("test", "test")):
        splits[target] = [row(text, label, f"clinc/{key}/{index}")
                          for key in (source, "oos_" + source)
                          for index, (text, label) in enumerate(data[key])]
    return splits, [path]


def recovered_snips(root):
    splits = {"train": [], "valid": [], "test": []}
    paths = []
    for intent in ("AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic",
                   "RateBook", "SearchCreativeWork", "SearchScreeningEvent"):
        for split, name in (("train", f"train_{intent}_full.json"), ("test", f"validate_{intent}.json")):
            path = root / intent / name
            paths.append(path)
            records = json.loads(path.read_text(encoding="latin-1"))[intent]
            splits[split].extend(row("".join(part["text"] for part in item["data"]), intent,
                                     f"snips/{split}/{intent}/{index}") for index, item in enumerate(records))
    return splits, paths


def recovered_atis(root):
    splits = {"train": [], "valid": [], "test": []}
    paths = []
    for split in ("train", "test"):
        path = root / f"atis_{split}.json"
        paths.append(path)
        records = json.loads(path.read_text())["rasa_nlu_data"]["common_examples"]
        splits[split] = [row(item["text"], item["intent"], f"atis/{split}/{index}")
                         for index, item in enumerate(records)]
    return splits, paths


def hwu_csv(directory):
    splits, paths = {}, []
    for source, target in (("train", "train"), ("val", "valid"), ("test", "test")):
        path = directory / f"{source}.csv"
        if source == "val" and not path.exists():
            splits[target] = []
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if {"text", "category"} <= fields:
                text_key, intent_key = "text", "category"
            elif {"utterance", "intent"} <= fields:
                text_key, intent_key = "utterance", "intent"
            else:
                raise ValueError(f"Expected text/category or utterance/intent columns: {path}")
            splits[target] = [row(item[text_key], item[intent_key], f"hwu/{source}/{index}")
                              for index, item in enumerate(reader)]
        paths.append(path)
    return splits, paths


def hwu_original(original, valid, test):
    """Recovered raw text membership and bare intent; diagnostic counts differ."""
    with original.open(newline="") as handle:
        records = [item for item in csv.DictReader(handle, delimiter=";") if item["answer_normalised"].strip()]
    valid_texts, test_texts = set(valid.read_text().splitlines()), set(test.read_text().splitlines())
    splits = {"train": [], "valid": [], "test": []}
    for index, item in enumerate(records):
        text = item["answer_normalised"]
        split = "valid" if text in valid_texts else "test" if text in test_texts else "train"
        splits[split].append(row(text, item["intent"], f"hwu/original/{index}"))
    return splits, [original, valid, test]


def audit(splits):
    keys = {split: Counter(canonical_text(item["utterance"]) for item in items) for split, items in splits.items()}
    return {"counts": {split: len(items) for split, items in splits.items()},
            "intent_counts": {split: len({item["intent"] for item in items}) for split, items in splits.items()},
            "duplicate_rows": {split: sum(count - 1 for count in texts.values()) for split, texts in keys.items()},
            "train_test_overlap_queries": sum(count for text, count in keys["test"].items() if text in keys["train"]),
            "train_valid_overlap_queries": sum(count for text, count in keys["valid"].items() if text in keys["train"]),
            "test_intents_absent_from_train": sorted({item["intent"] for item in splits["test"]}
                                                     - {item["intent"] for item in splits["train"]})}


def fetch_canonical(directory):
    """Fetch only public canonical JointBERT/CLINC data, never execute remote code."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "source_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        for item in lock["files"]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("Cached canonical input hash differs from lock")
        return lock
    lock = {"repositories": {}, "files": []}
    for repo, files in (("monologg/JointBERT", [f"data/{task}/{split}/{name}" for task in ("atis", "snips")
                                             for split in ("train", "dev", "test") for name in ("seq.in", "seq.out", "label")]),
                        ("clinc/oos-eval", ["data/data_oos_plus.json"])):
        revision = PINNED_REVISIONS[repo]
        lock["repositories"][repo] = revision
        for source in files:
            relative = Path("jointbert" if repo.startswith("monologg") else "clinc") / Path(source).relative_to("data")
            target = directory / relative
            if target.exists():
                raise ValueError(f"Untracked preexisting fetch destination: {target}")
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{source}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            lock["files"].append({"path": str(relative), "url": url, "sha256": sha256(target)})
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def fetch_hwu(directory):
    """Recover the public first fold used by the official DialoGLUE downloader.

    Table 2 uses the entire canonical fold's training reference set. Validation
    is not used; unavailable historical validation row indices are not inferred.
    """
    repo = "xliuhw/NLU-Evaluation-Data"
    lock_path = directory / "hwu_source_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        for item in lock["files"]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("HWU cached source hash mismatch")
        return lock
    revision = PINNED_REVISIONS[repo]
    with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1", timeout=60) as response:
        tree = json.load(response)["tree"]
    prefix = "CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1/"
    sources = sorted(item["path"] for item in tree if item["path"].startswith(prefix) and item["path"].endswith(".csv"))
    if len(sources) != 128:
        raise ValueError("Expected 64 train and 64 test HWU category files")

    def retrieve(source):
        relative = Path("hwu_raw") / Path(source).relative_to(prefix)
        target = directory / relative
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{source}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing HWU source differs: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"path": str(relative), "url": url, "sha256": sha256(target)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(retrieve, sources))
    splits = {"train": [], "test": []}
    for item in files:
        split = "train" if "/trainset/" in item["path"] else "test"
        with (directory / item["path"]).open(newline="") as handle:
            for value in csv.DictReader(handle, delimiter=";"):
                splits[split].append((value["answer_from_anno"], value["scenario"] + "_" + value["intent"]))
    for split, values in splits.items():
        path = directory / "hwu" / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("text", "category"))
            writer.writerows(values)
        files.append({"path": str(path.relative_to(directory)), "sha256": sha256(path)})
    lock = {"repository": repo, "revision": revision, "files": files,
                "protocol_note": "Canonical KFold_1 full train9960/test1076; validation unused and unavailable. Distinct from recovered original-CSV raw-match protocol."}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jointbert-root", type=Path)
    p.add_argument("--profile", choices=["recovered", "canonical"], default="recovered")
    p.add_argument("--snips-root", type=Path)
    p.add_argument("--atis-root", type=Path)
    p.add_argument("--hwu-profile", choices=["canonical-fold", "raw-match"], default="raw-match")
    p.add_argument("--massive-jsonl", required=True, type=Path)
    p.add_argument("--clinc-json", type=Path)
    p.add_argument("--hwu-dir", type=Path)
    p.add_argument("--hwu-original", type=Path)
    p.add_argument("--hwu-valid", type=Path)
    p.add_argument("--hwu-test", type=Path)
    p.add_argument("--fetch-canonical-dir", type=Path)
    p.add_argument("--source-lock", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output exists; choose a new benchmark artifact")
    lock = None
    if args.fetch_canonical_dir:
        lock = fetch_canonical(args.fetch_canonical_dir)
        args.jointbert_root = args.jointbert_root or args.fetch_canonical_dir / "jointbert"
        args.clinc_json = args.clinc_json or args.fetch_canonical_dir / "clinc/data_oos_plus.json"
        if not args.hwu_dir and args.hwu_profile == "canonical-fold":
            lock["hwu"] = fetch_hwu(args.fetch_canonical_dir)
            args.hwu_dir = args.fetch_canonical_dir / "hwu"
    if args.source_lock:
        lock = json.loads(args.source_lock.read_text())
    if not args.clinc_json:
        raise ValueError("Provide CLINC JSON or fetch canonical data")
    if args.profile == "canonical":
        if not args.jointbert_root:
            raise ValueError("Canonical profile requires JointBERT root")
        tasks = {task: jointbert(args.jointbert_root, task) for task in ("snips", "atis")}
    else:
        if not args.snips_root or not args.atis_root:
            raise ValueError("Recovered profile requires --snips-root and --atis-root")
        tasks = {"snips": recovered_snips(args.snips_root), "atis": recovered_atis(args.atis_root)}
    tasks["massive"] = massive(args.massive_jsonl)
    tasks["clinc"] = clinc(args.clinc_json)
    if args.hwu_profile == "canonical-fold" and args.hwu_dir:
        tasks["hwu"] = hwu_csv(args.hwu_dir)
    elif args.hwu_profile == "raw-match" and all((args.hwu_original, args.hwu_valid, args.hwu_test)):
        tasks["hwu"] = hwu_original(args.hwu_original, args.hwu_valid, args.hwu_test)
    else:
        raise ValueError("Provide labeled --hwu-dir or all three original HWU paths")
    result = {"protocol": "Table 2: full original training references, full test, cosine 1NN, no compression",
              "profile": args.profile, "hwu_profile": args.hwu_profile,
              "paper": "https://arxiv.org/html/2305.14299v3#A3", "source_lock": lock, "tasks": {}}
    for task, (splits, paths) in tasks.items():
        diagnostic = task == "hwu" and args.hwu_profile == "raw-match"
        if not splits["train"] or not splits["test"] or (len(splits["test"]) != TEST_COUNTS[task] and not diagnostic):
            raise ValueError(f"{task} split counts disagree with paper: {audit(splits)['counts']}")
        result["tasks"][task] = {"splits": splits, "audit": audit(splits),
                                    "paper_test_count_matches": len(splits["test"]) == TEST_COUNTS[task],
                                    "sources": {str(path.resolve()): sha256(path) for path in paths}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({task: value["audit"] for task, value in result["tasks"].items()}, indent=2))
    return result


if __name__ == "__main__":
    main()
