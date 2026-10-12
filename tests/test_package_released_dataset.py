"""Fidelity checks for packaging the released model's corpus."""
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('package_released_dataset', Path(__file__).parents[1] / 'scripts/package_released_dataset.py')
pkg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pkg)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(pkg.HEADER)
            writer.writerows(rows)
        return path

    def fixture(self):
        rows = [('  café\nhello\r\n世界  ', '  {SLOT}\n hi ', 1), ('duplicate', 'duplicate', 0), ('duplicate', 'duplicate', 0)]
        source = self.write_csv('snips.csv', rows)
        combined = self.write_csv('combined.csv', rows)
        code = self.root / 'generator.py'
        code.write_text('# original\n')
        input_record = {'path': str(code), 'sha256': pkg.sha256(code)}
        manifest = {'source_order': ['snips'], 'columns': pkg.HEADER, 'input_integrity_verified': True,
                        'code_and_inputs': [input_record], 'original_generator': input_record,
                        'sources': {'snips': {'retained_rows': 3, 'top_k': 2, 'trainable_retention': 'all_intents',
                                               'rows_per_intent': {'intent': 3}, 'inputs': [input_record],
                                               'trainable': {'output': source.name, 'sha256': pkg.sha256(source)}}},
                        'combined': {'output': combined.name, 'rows': 3, 'sha256': pkg.sha256(combined)}}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        for name, value in {'SOURCE_ORDER': ['snips'], 'COUNTS': {'snips': 3}, 'TOP_K': {'snips': 2},
                                'SNIPS_COUNTS': {'intent': 3}, 'COMBINED_SHA256': pkg.sha256(combined), 'BATCH_SIZE': 2}.items():
            patcher = patch.object(pkg, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return path, manifest, rows

    def test_csv_preserves_text_order_duplicates_and_digest(self):
        path, _, expected = self.fixture()
        rows = list(pkg.csv_rows(path.parent / 'snips.csv'))
        self.assertEqual([list(row.values()) for row in rows], [list(row) for row in expected])
        digest = hashlib.sha256()
        for row in rows:
            pkg.feed_digest(digest, row)
        reverse = hashlib.sha256()
        for row in reversed(rows):
            pkg.feed_digest(reverse, row)
        self.assertNotEqual(digest.digest(), reverse.digest())

    def test_rejects_slot_corruption_and_bad_field_count(self):
        for rows in [[('hello', '{SLOT}', 0)], [('hello', 'hi', '1.0')], [('hello', 'hi')]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                list(pkg.csv_rows(self.write_csv('bad.csv', rows)))

    def test_rejects_count_and_hash_corruption(self):
        path, manifest, _ = self.fixture()
        pkg.inspect_manifest(path)
        manifest['sources']['snips']['retained_rows'] = 4
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'count/top-k'):
            pkg.inspect_manifest(path)
        manifest['sources']['snips']['retained_rows'] = 3
        path.write_text(json.dumps(manifest))
        (self.root / 'snips.csv').write_text('corruption')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            pkg.inspect_manifest(path)

    def test_rejects_generation_input_corruption(self):
        path, _, _ = self.fixture()
        (self.root / 'generator.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Generation input hash mismatch'):
            pkg.inspect_manifest(path)

    @unittest.skipUnless(os.environ.get('TADSE_RUN_PARQUET_TESTS') == '1', 'opt-in PyArrow integration')
    def test_roundtrip_batches_and_fail_closed_on_order_mismatch(self):
        import pyarrow.parquet as pq
        path, manifest, rows = self.fixture()
        card = self.root / 'card.md'
        card.write_text('Private model training corpus')
        output = self.root / 'package'
        result = pkg.package(path, output, card)
        restored = pq.read_table(output / 'data/snips-train.parquet').to_pylist()
        self.assertEqual([r['utterance'] for r in restored], [r[0] for r in rows])
        self.assertEqual([r['source_row'] for r in restored], [0, 1, 2])
        self.assertTrue(result['verified'])
        self.assertEqual(result['combined_content_sha256'], result['sources']['snips']['content_sha256'])
        self.assertEqual(set(result['files_sha256']), {'README.md', 'provenance.json', 'data/snips-train.parquet'})
        self.assertEqual((output / 'provenance.json').read_bytes(), path.read_bytes())
        combined = self.write_csv('combined.csv', list(reversed(rows)))
        manifest['combined']['sha256'] = pkg.sha256(combined)
        path.write_text(json.dumps(manifest))
        with (patch.object(pkg, 'COMBINED_SHA256', pkg.sha256(combined)),
              self.assertRaisesRegex(ValueError, 'content/order mismatch')):
            pkg.package(path, self.root / 'bad-package', card)
        self.assertFalse((self.root / 'bad-package/verification.json').exists())


if __name__ == '__main__':
    unittest.main()
