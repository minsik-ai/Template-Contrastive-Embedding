#!/usr/bin/env python3
"""Publish a freshly verified package to a NEW, private Hugging Face dataset."""
import argparse
import json
from pathlib import Path

import prepare_dataset_release as gate


def publish(release_dir, repo_id, api=None):
    report = json.loads((release_dir / "verification.json").read_text())
    if report.get("eligible_for_private_upload") is not True or report.get("packaged") is not True:
        raise ValueError("Upload blocked: paper-count verification and packaging must both pass")
    if report.get("failures"):
        raise ValueError("Upload blocked: verification contains failed checks")
    manifest = Path(report["manifest"]["path"])
    if gate.sha256(manifest) != report["manifest"]["sha256"]:
        raise ValueError("Upload blocked: generation manifest changed after verification")
    # Re-read the actual originals and CSVs; a saved success flag is insufficient.
    fresh = gate.audit(manifest)
    if fresh.get("eligible_for_private_upload") is not True:
        raise ValueError("Upload blocked: fresh paper-count and content audit failed")
    expected = {f"data/{source}-train.parquet" for source in gate.PAPER}
    records = report.get("parquet_files", [])
    if len(records) != len(expected) or {item["path"] for item in records} != expected:
        raise ValueError("Upload blocked: expected one verified Parquet file per dataset")
    for item in records:
        path = gate.relative_file(release_dir, item["path"])
        if gate.sha256(path) != item["sha256"]:
            raise ValueError(f"Upload blocked: packaged file changed: {item['path']}")
    gate.relative_file(release_dir, "README.md")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if repo_id.split("/")[0] != api.whoami()["name"]:
        raise ValueError("Use the authenticated user's namespace for this private review")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
    if api.dataset_info(repo_id).private is not True:
        raise RuntimeError("Refusing upload: destination is not confirmed private")
    allowed = sorted(expected | {"README.md", "verification.json"})
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(release_dir),
        allow_patterns=allowed,
        commit_message="Enable private review of verified TaDSE data",
    )
    info = api.dataset_info(repo_id, revision=commit.oid)
    actual = set(api.list_repo_files(repo_id, repo_type="dataset", revision=commit.oid))
    if info.private is not True or not set(allowed) <= actual:
        raise RuntimeError("Uploaded dataset failed privacy/file-list verification")
    receipt = {"repo_id": repo_id, "private": True, "commit": commit.oid,
               "url": f"https://huggingface.co/datasets/{repo_id}"}
    (release_dir / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.release_dir, args.repo_id), indent=2))


if __name__ == "__main__":
    main()
