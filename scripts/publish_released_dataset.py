#!/usr/bin/env python3
"""Publish the verified released-model corpus to a new private Hub dataset."""
import argparse
import hashlib
import json
from pathlib import Path

from package_released_dataset import (
    COMBINED_SHA256,
    COUNTS,
    SOURCE_ORDER,
    feed_digest,
    sha256,
)


def publish(release_dir, repo_id):
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    release_dir = Path(release_dir).resolve()
    report = json.loads((release_dir/'verification.json').read_text())
    if (report.get('verified') is not True or report.get('scope') != 'released-model-top2-training-pairs'
            or report.get('total_rows') != sum(COUNTS.values())
            or report.get('combined_sha256') != COMBINED_SHA256 or report.get('source_order') != SOURCE_ORDER):
        raise ValueError('Package is not the verified released-model top2 corpus')
    if sha256(report['manifest']['path']) != report['manifest']['sha256']:
        raise ValueError('Generation manifest changed')
    for item in report['input_files']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Generation input changed: ' + item['path'])
    expected = {'README.md', 'provenance.json'} | {f'data/{s}-train.parquet' for s in SOURCE_ORDER}
    if set(report['files_sha256']) != expected:
        raise ValueError('Unexpected packaged files')
    hashes = dict(report['files_sha256'])
    hashes['verification.json'] = sha256(release_dir/'verification.json')
    for name, digest in hashes.items():
        path = (release_dir/name).resolve()
        if not path.is_relative_to(release_dir) or sha256(path) != digest:
            raise ValueError('Packaged file changed: ' + name)
    api = HfApi()
    if repo_id.split('/')[0] != api.whoami()['name']:
        raise ValueError('Use the authenticated user namespace')
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=False)
    if api.dataset_info(repo_id).private is not True:
        raise RuntimeError('Destination is not private')
    commit = api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=str(release_dir),
        allow_patterns=sorted(hashes), commit_message='Release the exact merged-model training corpus for private review')
    if api.dataset_info(repo_id, revision=commit.oid).private is not True:
        raise RuntimeError('Private visibility verification failed')
    for name, digest in hashes.items():
        fetched = hf_hub_download(repo_id, name, repo_type='dataset', revision=commit.oid)
        if sha256(fetched) != digest:
            raise RuntimeError('Hub file checksum mismatch: ' + name)
    configs = get_dataset_config_names(repo_id, revision=commit.oid, token=True)
    if set(configs) != {'combined', *SOURCE_ORDER}:
        raise RuntimeError('Hub configurations do not match the release')
    dataset = load_dataset(repo_id, 'combined', split='train', revision=commit.oid, token=True)
    if len(dataset) != sum(COUNTS.values()):
        raise RuntimeError('Hub-loaded combined row count differs')
    if dataset.column_names != ['utterance', 'pattern', 'slot_count', 'source', 'source_row']:
        raise RuntimeError('Hub-loaded schema differs')
    digest = hashlib.sha256()
    source_index = 0
    source_row = 0
    for batch in dataset.iter(batch_size=8192):
        for values in zip(*(batch[name] for name in dataset.column_names)):
            row = dict(zip(dataset.column_names, values))
            source = SOURCE_ORDER[source_index]
            if row['source'] != source or row['source_row'] != source_row:
                raise RuntimeError('Hub-loaded source ordering differs')
            feed_digest(digest, row)
            source_row += 1
            if source_row == COUNTS[source]:
                source_index += 1
                source_row = 0
    if digest.hexdigest() != report['combined_content_sha256'] or source_index != len(SOURCE_ORDER):
        raise RuntimeError('Hub-loaded ordered content digest differs')
    receipt = {'repo_id':repo_id, 'url':f'https://huggingface.co/datasets/{repo_id}',
        'revision':commit.oid, 'private_verified':True, 'total_rows':len(dataset),
        'source_counts':COUNTS, 'configs':configs, 'files_sha256':hashes,
        'download_hashes_verified':True, 'hub_loaded_content_verified':True,
        'combined_csv_sha256':COMBINED_SHA256, 'combined_content_sha256':digest.hexdigest(),
        'scope':report['scope']}
    (release_dir.parent/'publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', required=True, type=Path)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    publish(args.release_dir, args.repo_id)
