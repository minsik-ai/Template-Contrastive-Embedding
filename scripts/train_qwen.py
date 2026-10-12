#!/usr/bin/env python3
"""Explicit Qwen TaDSE adaptation using the recovered custom training loop."""
import argparse
import ast
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))


def original_preprocessing(tokenizer, max_length, columns):
    import torch
    from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase,
    )
    path = ROOT/'legacy/training/model/pairwise_pattern_train.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != '211ab55a7a079d978cee43cdb6d0ea8278f4f461feaade820a4b1ceec6bfd51a':
        raise ValueError('Recovered preprocessing source changed')
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prepare = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_features')
    collator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OurDataCollatorWithPadding')
    ns = {'torch':torch, 'dataclass':dataclasses.dataclass, 'tokenizer':tokenizer,
          'data_args':SimpleNamespace(max_seq_length=max_length, pad_to_max_length=False,
                                     slot_file=None, one_slot_token=False),
          'utt_cname':columns[0], 'ptn_cname':columns[1], 'Dict':dict, 'List':list,
          'Optional':Optional, 'Union':Union, 'PaddingStrategy':PaddingStrategy,
          'PreTrainedTokenizerBase':PreTrainedTokenizerBase}
    exec(compile(ast.Module(body=[prepare, collator], type_ignores=[]), str(path), 'exec'), ns)  # noqa: S102 -- hash-pinned original code
    return ns['prepare_features'], ns['OurDataCollatorWithPadding'](tokenizer), {
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'prepare_ast_sha256': hashlib.sha256(ast.dump(prepare, include_attributes=False).encode()).hexdigest(),
        'collator_ast_sha256': hashlib.sha256(ast.dump(collator, include_attributes=False).encode()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--train-file', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--legacy-transformers-source', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--learning-rate', type=float, default=1e-8)
    p.add_argument('--max-length', type=int, default=32)
    p.add_argument('--attention-dropout', type=float, default=0.1)
    p.add_argument('--apply-ptn-transform', action='store_true')
    p.add_argument('--max-samples', type=int)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--save-steps', type=int, default=5000)
    p.add_argument('--logging-steps', type=int, default=200)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    import torch
    import transformers
    from datasets import load_dataset
    from qwen_trainer import QwenCLTrainer
    from tadse.qwen_models import Qwen3ForPatternCL
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError('Qwen tokenizer must define padding')
    model_args = SimpleNamespace(pooler_type='pad_left_no_pooler', mlp_only_train=False,
        apply_ptn_transform=args.apply_ptn_transform, simcse_utt_temp=0.05, simcse_ptn_temp=0.05,
        pairwise_temp=0.05, simcse_utt_loss_ratio=1.0, simcse_ptn_loss_ratio=1.0,
        pairwise_loss_ratio=0.5, pairwise_neg_type='utterance', mlm_loss_ratio=0.0, hidden_size=1024)
    model = Qwen3ForPatternCL.from_encoder_pretrained(str(args.checkpoint), model_args=model_args,
        attention_dropout=args.attention_dropout, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager')
    args.output_dir.mkdir(parents=True)
    dataset = load_dataset('csv', data_files={'train':str(args.train_file)},
        cache_dir=str(args.output_dir/'dataset-cache'))['train']
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    prepare, collator, preprocess_audit = original_preprocessing(tokenizer, args.max_length, dataset.column_names)
    dataset = dataset.map(prepare, batched=True, remove_columns=dataset.column_names)
    training_args = TrainingArguments(output_dir=str(args.output_dir/'model'),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate,
        seed=args.seed, report_to=[], eval_strategy='no', save_strategy='steps',
        save_steps=args.save_steps, save_total_limit=2, logging_steps=args.logging_steps,
        remove_unused_columns=False, dataloader_drop_last=False, dataloader_pin_memory=True)
    trainer = QwenCLTrainer(model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, processing_class=tokenizer,
        legacy_transformers_source=args.legacy_transformers_source)
    plan = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    plan.update(training_rows=len(dataset), model_args=vars(model.model_args),
        training_args=training_args.to_dict(), trainer_audit=trainer.legacy_audit,
        preprocessing=preprocess_audit, torch_version=torch.__version__,
        transformers_version=transformers.__version__, native_dimensions=model.config.hidden_size,
        evaluation_dimensions=768, train_sha256=hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        is_published_qwen_finetuning_recipe=False)
    (args.output_dir/'run-plan.json').write_text(json.dumps(plan,indent=2,default=str)+'\n')
    result = trainer.train()
    trainer.save_model(str(args.output_dir/'model'))
    tokenizer.save_pretrained(args.output_dir/'model')
    trainer.state.save_to_json(str(args.output_dir/'model/trainer_state.json'))
    model.eval()
    probes=tokenizer(['Book a table','Weather in New York tomorrow'],padding=True,return_tensors='pt').to(model.device)
    with torch.no_grad():
        expected=model(**probes,sent_emb=True).pooler_output.detach().cpu()
    export=args.output_dir/'encoder'
    model.model.save_pretrained(export)
    tokenizer.save_pretrained(export)
    from transformers import AutoModel
    restored=AutoModel.from_pretrained(export,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='eager').to(model.device).eval()
    with torch.no_grad():
        positions=probes['attention_mask'].long().cumsum(-1)-1
        positions=positions.masked_fill(probes['attention_mask']==0,0)
        actual=restored(**probes,position_ids=positions,use_cache=False).last_hidden_state[:,-1].cpu()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if not torch.isfinite(actual).all():
        raise RuntimeError('Non-finite exported embeddings')
    report={'training_loss':result.training_loss,'steps':trainer.state.global_step,
            'epoch':trainer.state.epoch,'export_max_abs_diff':float((actual-expected).abs().max()),
            'native_dimensions':actual.shape[-1],'encoder':str(export)}
    (args.output_dir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
