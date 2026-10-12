#!/usr/bin/env python3
"""Fixed equal-weight merge of two compatible exported BERT models; Python 3.8."""
import argparse
import json
from pathlib import Path

from export_recovered_training import PROBES, directory_hashes, state_hashes


def merge(left, right, output):
    import torch
    from transformers import AutoModel, AutoTokenizer, BertModel
    left, right, output = [Path(p).resolve() for p in (left, right, output)]
    if left == right or output.exists() or any(p == output or p in output.parents for p in (left, right)):
        raise ValueError('Use distinct sources and a fresh output outside both sources')
    source_hashes = [directory_hashes(p) for p in (left, right)]
    models = [AutoModel.from_pretrained(str(p), local_files_only=True).cpu().eval() for p in (left, right)]
    if not all(isinstance(m, BertModel) for m in models):
        raise ValueError('Only standalone exported BERT models are supported')
    configs = [m.config.to_dict() for m in models]
    for config in configs:
        config.pop('_name_or_path', None)
        config.pop('name_or_path', None)
    if configs[0] != configs[1]:
        raise ValueError('Model configurations differ beyond source path metadata')
    tokenizers = [AutoTokenizer.from_pretrained(str(p), local_files_only=True) for p in (left, right)]
    a, b = tokenizers
    if type(a) is not type(b) or a.get_vocab() != b.get_vocab() or a.special_tokens_map != b.special_tokens_map:
        raise ValueError('Tokenizer classes, vocabularies or special tokens differ')
    for key in ('do_lower_case', 'model_max_length', 'padding_side', 'truncation_side'):
        if getattr(a, key, None) != getattr(b, key, None):
            raise ValueError('Tokenizer setting differs: ' + key)
    tokens = [t(PROBES, padding=True, truncation=True, max_length=32, return_tensors='pt') for t in tokenizers]
    if set(tokens[0]) != set(tokens[1]) or any(not torch.equal(tokens[0][k], tokens[1][k]) for k in tokens[0]):
        raise ValueError('Tokenizer probe outputs differ')
    states = [m.state_dict() for m in models]
    if set(states[0]) != set(states[1]):
        raise ValueError('State keys differ')
    float_count = integer_count = 0
    with torch.no_grad():
        for name, x in states[0].items():
            y = states[1][name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError('Tensor shape/dtype differs: ' + name)
            if x.is_floating_point():
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError('Non-finite source tensor: ' + name)
                x.copy_((x.float() * 0.5 + y.float() * 0.5).to(x.dtype))
                float_count += 1
            else:
                if not torch.equal(x, y):
                    raise ValueError('Non-floating buffers differ: ' + name)
                integer_count += 1
    expected_state = state_hashes(models[0])
    with torch.no_grad():
        expected_vectors = models[0](**tokens[0]).pooler_output
    if not torch.isfinite(expected_vectors).all():
        raise RuntimeError('Merged embeddings are non-finite')
    output.mkdir(parents=True)
    models[0].save_pretrained(str(output))
    tokenizers[0].save_pretrained(str(output))
    restored = AutoModel.from_pretrained(str(output), local_files_only=True).cpu().eval()
    if state_hashes(restored) != expected_state:
        raise RuntimeError('Merged checkpoint reload changed tensors')
    with torch.no_grad():
        actual = restored(**tokens[0]).pooler_output
    if not torch.equal(expected_vectors, actual):
        raise RuntimeError('Merged checkpoint reload changed probe embeddings')
    if [directory_hashes(p) for p in (left, right)] != source_hashes:
        raise RuntimeError('Source files changed')
    audit = {'sources': [str(left), str(right)], 'weights': [0.5, 0.5],
             'method': 'equal-weight parameter average, including exported pooler',
             'source_sha256': source_hashes, 'source_files_unchanged': True,
             'floating_tensors_averaged': float_count, 'identical_nonfloating_buffers': integer_count,
             'reload_max_abs_diff': float((expected_vectors-actual).abs().max()),
             'export_sha256': directory_hashes(output),
             'selection': 'Fixed 50/50 before evaluation; no ratio sweep or per-dataset selection'}
    (output/'merge-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'output':str(output),'floating_tensors_averaged':float_count,'reload_max_abs_diff':audit['reload_max_abs_diff']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', required=True, type=Path)
    p.add_argument('--right', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    merge(args.left, args.right, args.output_dir)
