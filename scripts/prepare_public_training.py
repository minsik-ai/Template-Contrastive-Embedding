#!/usr/bin/env python3
"""Download pinned public inputs and recover the released model's exact training CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'minsik-oh/template-dialogue-pairs'
DATASET_REVISION = '587ae7e976c82e6ee6b44dec17dc1ecb06b45d58'
MODEL = 'princeton-nlp/unsup-simcse-bert-base-uncased'
MODEL_REVISION = '218baf6bc48c463a53f5b4e1c568299537fab124'
CSV_SHA256 = '9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4'
SOURCES = ('snips', 'atis', 'massive', 'hwu', 'clinc')
COLUMNS = ('utterance', 'pattern', 'slot_count')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_csv(rows, output, expected_sha256=CSV_SHA256):
    """Keep record order, whitespace, duplicates, and embedded newlines intact."""
    with Path(output).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in COLUMNS])
    if sha256(output) != expected_sha256:
        raise ValueError('Downloaded rows do not reproduce the released training CSV')


def prepare(output):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    manifest = json.loads((ROOT / 'releases/tadse-top2-dialogue-pairs/manifest.json').read_text())
    if manifest['repo_id'] != DATASET or manifest['combined_csv_sha256'] != CSV_SHA256:
        raise ValueError('Release manifest differs from the pinned training recipe')
    output.mkdir(parents=True)
    paths = []
    hashes = {}
    for source in SOURCES:
        name = 'data/' + source + '-train.parquet'
        path = Path(hf_hub_download(DATASET, name, repo_type='dataset',
                                   revision=DATASET_REVISION, local_dir=str(output / 'dataset')))
        hashes[name] = sha256(path)
        if hashes[name] != manifest['files_sha256'][name]:
            raise ValueError('Released Parquet hash mismatch: ' + name)
        paths.append(path)

    def rows():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=list(COLUMNS)):
                yield from batch.to_pylist()

    write_training_csv(rows(), output / 'combined_train.csv')
    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=str(output / 'initial-model'),
                      allow_patterns=['config.json', 'pytorch_model.bin', 'vocab.txt',
                                      'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])
    for name in ('config.json', 'pytorch_model.bin', 'vocab.txt'):
        if not (output / 'initial-model' / name).is_file():
            raise FileNotFoundError('Initial checkpoint is missing ' + name)
    audit = {'dataset': DATASET, 'dataset_revision': DATASET_REVISION,
             'parquet_sha256': hashes, 'combined_csv_sha256': CSV_SHA256,
             'initial_model': MODEL, 'initial_model_revision': MODEL_REVISION,
             'source_order': list(SOURCES), 'columns': list(COLUMNS)}
    (output / 'preparation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Prepared training inputs in ' + str(output))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)


if __name__ == '__main__':
    main()
