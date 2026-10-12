#!/usr/bin/env python3
"""Package the exact released BERT merge's top-2 training pairs, without uploading.

This is a model-corpus fidelity check, not the separate historical paper-count gate.
A successful verification.json is written only after every round-trip check passes.
"""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path

SOURCE_ORDER = ['snips', 'atis', 'massive', 'hwu', 'clinc']
COUNTS = dict(zip(SOURCE_ORDER, [115378, 238847, 78289, 133176, 220180]))
TOP_K = dict(zip(SOURCE_ORDER, [2, 2, 3, 3, 5]))
SNIPS_COUNTS = dict(zip(
    ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook',
     'SearchCreativeWork', 'SearchScreeningEvent'],
    [15975, 28663, 13644, 13276, 22412, 8792, 12616]))
COMBINED_SHA256 = '9b8404042073814afbf0a63670e8ef9919d722cf8bee1b7aea208ecdad696dd4'
HEADER = ['utterance', 'pattern', 'slot_count']
BATCH_SIZE = 8192


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path):
    """Yield typed records without stripping, normalizing, or deduplicating text."""
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError(f'{path}: invalid CSV header')
        for number, fields in enumerate(reader, 1):
            if len(fields) != 3:
                raise ValueError(f'{path}: row {number}: expected three fields')
            utterance, pattern, count = fields
            if not count.isascii() or not count.isdecimal():
                raise ValueError(f'{path}: row {number}: invalid integer slot_count')
            count = int(count)
            if count != pattern.count('{SLOT}'):
                raise ValueError(f'{path}: row {number}: slot_count mismatch')
            yield {'utterance': utterance, 'pattern': pattern, 'slot_count': count}


def feed_digest(digest, row):
    """Unambiguous, ordered UTF-8 content digest shared by CSV and Parquet."""
    digest.update(json.dumps([row[key] for key in HEADER], ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8') + b'\n')


def artifact(root, record):
    name = record['output']
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Artifact path must remain inside manifest directory')
    if sha256(path) != record['sha256']:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return path


def inspect_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    before = {manifest_path: sha256(manifest_path)}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['source_order'] != SOURCE_ORDER or set(manifest['sources']) != set(SOURCE_ORDER)
            or manifest['columns'] != HEADER or manifest.get('input_integrity_verified') is not True):
        raise ValueError('Manifest source order, columns, or generation verification mismatch')
    if not manifest.get('code_and_inputs'):
        raise ValueError('Missing generation code/input hashes')
    for record in manifest['code_and_inputs']:
        path = Path(record['path'])
        if not path.is_absolute() or sha256(path) != record['sha256']:
            raise ValueError(f'Generation input hash mismatch: {path}')
        if path in before and before[path] != record['sha256']:
            raise ValueError(f'Conflicting input hashes: {path}')
        before[path] = record['sha256']
    generator = manifest['original_generator']
    if before.get(Path(generator['path'])) != generator['sha256']:
        raise ValueError('Original generator is missing from verified inputs')
    paths = {}
    for source in SOURCE_ORDER:
        record = manifest['sources'][source]
        if record['retained_rows'] != COUNTS[source] or record['top_k'] != TOP_K[source]:
            raise ValueError(f'{source}: release count/top-k mismatch')
        for item in record['inputs']:
            if before.get(Path(item['path'])) != item['sha256']:
                raise ValueError(f'{source}: input not covered by verified generation hashes')
        paths[source] = artifact(manifest_path.parent, record['trainable'])
        before[paths[source]] = record['trainable']['sha256']
    snips = manifest['sources']['snips']
    if snips['trainable_retention'] != 'all_intents' or snips['rows_per_intent'] != SNIPS_COUNTS:
        raise ValueError('SNIPS all-intent provenance mismatch')
    combined = artifact(manifest_path.parent, manifest['combined'])
    if manifest['combined']['rows'] != sum(COUNTS.values()) or sha256(combined) != COMBINED_SHA256:
        raise ValueError('Combined corpus differs from released model training corpus')
    before[combined] = COMBINED_SHA256
    return manifest, paths, combined, before


def package(manifest_path, output_dir, card_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest_path, output_dir, card_path = (Path(p).resolve() for p in (manifest_path, output_dir, card_path))
    if output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    _manifest, paths, combined, before = inspect_manifest(manifest_path)
    before[card_path] = sha256(card_path)
    schema = pa.schema([('utterance', pa.string()), ('pattern', pa.string()),
                        ('slot_count', pa.int64()), ('source', pa.string()), ('source_row', pa.int64())])
    (output_dir / 'data').mkdir(parents=True)
    combined_rows = csv_rows(combined)
    combined_digest = hashlib.sha256()
    reports = {}
    try:
        for source in SOURCE_ORDER:
            target = output_dir / 'data' / f'{source}-train.parquet'
            digest = hashlib.sha256()
            count, batch = 0, []
            with pq.ParquetWriter(target, schema, compression='zstd') as writer:
                for row in csv_rows(paths[source]):
                    if next(combined_rows, None) != row:
                        raise ValueError(f'{source}: combined CSV content/order mismatch at row {count}')
                    feed_digest(digest, row)
                    feed_digest(combined_digest, row)
                    batch.append(dict(row, source=source, source_row=count))
                    count += 1
                    if len(batch) == BATCH_SIZE:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            if count != COUNTS[source]:
                raise ValueError(f'{source}: actual row count mismatch')
            parquet = pq.ParquetFile(target)
            if parquet.schema_arrow != schema:
                raise ValueError(f'{source}: Parquet schema mismatch')
            roundtrip = (row for chunk in parquet.iter_batches(batch_size=BATCH_SIZE)
                         for row in chunk.to_pylist())
            recovered_digest = hashlib.sha256()
            for index, (original, restored) in enumerate(itertools.zip_longest(csv_rows(paths[source]), roundtrip)):
                expected = dict(original, source=source, source_row=index) if original is not None else None
                if restored != expected:
                    raise ValueError(f'{source}: Parquet roundtrip mismatch at row {index}')
                feed_digest(recovered_digest, restored)
            if recovered_digest.digest() != digest.digest():
                raise ValueError(f'{source}: content digest mismatch')
            reports[source] = {'count': count, 'top_k': TOP_K[source], 'csv_sha256': before[paths[source]],
                                   'parquet_sha256': sha256(target), 'content_sha256': digest.hexdigest()}
        if next(combined_rows, None) is not None:
            raise ValueError('Combined CSV contains extra rows')
    finally:
        combined_rows.close()
    shutil.copyfile(card_path, output_dir / 'README.md')
    shutil.copyfile(manifest_path, output_dir / 'provenance.json')
    for path, expected in before.items():
        if sha256(path) != expected:
            raise ValueError(f'Input changed while packaging: {path}')
    files = {str(path.relative_to(output_dir)): sha256(path)
             for path in sorted(output_dir.rglob('*')) if path.is_file()}
    result = {'verified': True, 'scope': 'released-model-top2-training-pairs',
                  'total_rows': sum(COUNTS.values()), 'source_order': SOURCE_ORDER, 'sources': reports,
                  'combined_sha256': COMBINED_SHA256, 'combined_content_sha256': combined_digest.hexdigest(),
                  'manifest': {'path': str(manifest_path), 'sha256': before[manifest_path]},
                  'input_files': [{'path': str(path), 'sha256': value} for path, value in sorted(before.items())],
                  'files_sha256': files}
    (output_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--card', type=Path, required=True)
    args = parser.parse_args()
    result = package(args.manifest, args.output_dir, args.card)
    print(json.dumps({'verified': result['verified'], 'total_rows': result['total_rows'], 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
