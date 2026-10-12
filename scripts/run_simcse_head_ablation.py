#!/usr/bin/env python3
"""Original training with one explicit initialization intervention; Python 3.8."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'legacy/training/model'


def tensor_hashes(model):
    return {name: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, t in model.state_dict().items()}


def initialize_from_simcse(model):
    import torch
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    before = tensor_hashes(model)
    with torch.no_grad():
        model.mlp.dense.weight.copy_(model.bert.pooler.dense.weight)
        model.mlp.dense.bias.copy_(model.bert.pooler.dense.bias)
    after = tensor_hashes(model)
    changed = sorted(name for name in before if before[name] != after[name])
    allowed = {'mlp.dense.weight', 'mlp.dense.bias'}
    if set(before) != set(after) or not set(changed).issubset(allowed):
        raise RuntimeError('Initialization changed tensors outside the contrastive head')
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError('Initialization consumed CPU RNG')
    if cuda_rng is not None and any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise RuntimeError('Initialization consumed CUDA RNG')
    for suffix in ('weight', 'bias'):
        if after['mlp.dense.' + suffix] != before['bert.pooler.dense.' + suffix]:
            raise RuntimeError('SimCSE pooler mapping is not exact')
    return {'mapping': {'bert.pooler.dense.weight': 'mlp.dense.weight',
                        'bert.pooler.dense.bias': 'mlp.dense.bias'},
            'changed_tensors': changed, 'all_other_tensors_unchanged': True,
            'pattern_transform_unchanged': True, 'rng_unchanged': True,
            'before_head_sha256': {k: before[k] for k in sorted(allowed)},
            'after_head_sha256': {k: after[k] for k in sorted(allowed)},
            'note': 'mlp is shared by utterance and template views; separate ptn_transform is unchanged'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('training_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    training_args = args.training_args
    if training_args and training_args[0] == '--':
        training_args = training_args[1:]
    if not training_args:
        p.error('Provide original training arguments after --')
    if args.audit.exists():
        raise FileExistsError('Preserving existing initialization audit')
    provenance = json.loads((ROOT / 'legacy/training/provenance.json').read_text())
    for entry in provenance['files']:
        file = ROOT / 'legacy/training' / entry['path']
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Recovered source changed: ' + entry['path'])
    sys.path.insert(0, str(LEGACY))
    from tadse.models import BertForPatternCL
    original = BertForPatternCL.from_pretrained
    initialized = False

    def load(cls, *a, **kw):
        nonlocal initialized
        model = original(*a, **kw)
        if not initialized:
            if isinstance(model, tuple):
                raise RuntimeError('Unexpected loading-info return in original entry point')
            audit = initialize_from_simcse(model)
            audit.update(source_checkpoint=str(a[0]), recovered_source_commit=provenance['source_commit'],
                         apply_ptn_transform=model.model_args.apply_ptn_transform)
            args.audit.parent.mkdir(parents=True, exist_ok=True)
            args.audit.write_text(json.dumps(audit, indent=2) + '\n')
            initialized = True
        return model

    entry = LEGACY / 'pairwise_pattern_train.py'
    previous_argv = sys.argv
    try:
        sys.argv = [str(entry)] + training_args
        with patch.object(BertForPatternCL, 'from_pretrained', classmethod(load)):
            runpy.run_path(str(entry), run_name='__main__')
        if not initialized:
            raise RuntimeError('Original BERT initializer was not called')
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
