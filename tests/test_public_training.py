import csv
import hashlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_public_training', ROOT / 'scripts/prepare_public_training.py')
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class PublicTrainingTests(unittest.TestCase):
    def test_csv_keeps_order_duplicates_unicode_and_newlines(self):
        rows = [{'utterance': ' café\nhello ', 'pattern': '{SLOT}\nhello ', 'slot_count': 1},
                {'utterance': ' café\nhello ', 'pattern': '{SLOT}\nhello ', 'slot_count': 1},
                {'utterance': 'a,b', 'pattern': 'a,b', 'slot_count': 0}]
        expected = io.StringIO(newline='')
        writer = csv.writer(expected)
        writer.writerow(preparation.COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in preparation.COLUMNS])
        data = expected.getvalue().encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            preparation.write_training_csv(iter(rows), path, hashlib.sha256(data).hexdigest())
            self.assertEqual(path.read_bytes(), data)
            with self.assertRaises(FileExistsError):
                preparation.write_training_csv(rows, path)

    def test_csv_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'do not reproduce'):
            preparation.write_training_csv([], Path(directory) / 'train.csv')

    def test_pinned_order_and_release_digest(self):
        self.assertEqual(preparation.SOURCES, ('snips', 'atis', 'massive', 'hwu', 'clinc'))
        self.assertEqual(preparation.DATASET_REVISION, '587ae7e976c82e6ee6b44dec17dc1ecb06b45d58')
        self.assertEqual(preparation.MODEL_REVISION, '218baf6bc48c463a53f5b4e1c568299537fab124')

    def test_runner_dispatches_original_training_export_and_fixed_merge(self):
        # A stub runtime captures the actual shell argument boundaries without training.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / '.venv-legacy/bin').mkdir(parents=True)
            (root / 'artifacts/legacy-recovery-20261011/SentEval/senteval').mkdir(parents=True)
            runner = root / 'scripts/train_released_bert.sh'
            runner.write_text((ROOT / 'scripts/train_released_bert.sh').read_text())
            log = root / 'commands.jsonl'
            runtime = root / '.venv-legacy/bin/python'
            runtime.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
if args[0] == '-c':
    print(pathlib.Path(args[2]).resolve())
elif args[0] != '-':
    with open(str(pathlib.Path(__file__).resolve().parents[2] / 'commands.jsonl'), 'a') as f:
        f.write(json.dumps(args) + '\\n')
''')
            runtime.chmod(0o755)
            subprocess.run(['bash', str(runner), 'data.csv', 'initial-model', str(root / 'runs')], check=True)
            import json
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 5)
            for index in (0, 2):
                command = commands[index]
                self.assertEqual(command[0], 'scripts/run_simcse_head_ablation.py')
                for flag, value in (('--num_train_epochs', '2'), ('--per_device_train_batch_size', '16'),
                                    ('--learning_rate', '1e-8'), ('--max_seq_length', '32'),
                                    ('--seed', '42'), ('--pairwise_loss_ratio', '0.5'),
                                    ('--pairwise_neg_type', 'utterance')):
                    self.assertEqual(command[command.index(flag) + 1], value)
            self.assertNotIn('--apply_ptn_transform', commands[0])
            self.assertIn('--apply_ptn_transform', commands[2])
            self.assertNotIn('--apply-ptn-transform', commands[1])
            self.assertIn('--apply-ptn-transform', commands[3])
            self.assertEqual(commands[4][0], 'scripts/merge_exported_models.py')
            with patch('subprocess.run', wraps=subprocess.run) as run:
                result = run(['bash', str(runner), 'data.csv', 'initial-model', str(root / 'runs')],
                             capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(log.read_text().splitlines()), 5)


if __name__ == '__main__':
    unittest.main()
