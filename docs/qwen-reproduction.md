# Qwen baseline and TaDSE adaptation

The paper's section5.5/Table6 evaluates Qwen3-Embedding-0.6B at768 dimensions
on SNIPS/ATIS (92.14/89.14 reported). It does not describe a TaDSE-fine-tuned
Qwen experiment. The user requested both the baseline and a new TaDSE adaptation.

Backbone: Qwen/Qwen3-Embedding-0.6B, immutable revision
97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3. Its native width is1024 and its
attention_dropout is0.0. Qwen requires Transformers>=4.51, so the BERT4.2.1
runtime cannot load this architecture directly.

## Baseline

Evaluate raw utterances with no instructions, left padding, last real token,
cumulative position IDs, maximum length128, first768 dimensions then L2
normalization, FP32/eager attention. These follow the paper's stated model/width
and recovered repository utterance evaluation convention; the exact historical
Qwen invocation was not recovered. All five recovered splits are reported, plus
the SNIPS/ATIS average for Table6. No NER or template inference is used.

## Fine-tuning experiment

Use the all-intent top2 corpus (785870 total pairs), seed42, two epochs, batch16,
length32, learning rate1e-8, losses1/1/0.5 with utterance negatives. First run
has no additional pattern MLP. Train the native1024-dimensional representations;
export a standard native Qwen encoder and evaluate its first768 dimensions.
There is no learned768 projection or BERT/SimCSE pooler-head mapping.

Attention dropout0.1 is an explicit new adaptation to supply stochastic positive
views. It is not a recovered paper setting. Baseline inference remains in eval
mode with its original config; dropout is disabled for fine-tuned inference too.

## Compatibility boundary

Original BERT sources and original objective functions are unchanged. The new
Qwen wrapper contains one encoder, fixes the old duplicate-backbone design,
filters BERT-only forward keywords, disables cache, and enforces left padding.
Pretrained encoder identity and exact native export are tested.

QwenCLTrainer binds unchanged original CLTrainer training, sampler, compute_loss
and checkpoint methods by AST. It also binds Transformers4.2.1 AdamW, optimizer
creation, training step, dataloader, logging and flow callback from the installed
legacy source. Modern Trainer supplies construction/serialization only. The
adapter explicitly supports one device, FP32, no accumulation or resume, and no
mid-training evaluation. Source and AST hashes are recorded. The original
preprocessing function and four-view collator are also executed by AST.

Tests cover finite four-view gradients, pattern projection gradients when
enabled, padding invariance, exact pretrained loading/export, and two complete
custom-trainer epochs including checkpoints. The full-model GPU pilot gates the
long run and provides a runtime estimate.

Sources: [paper](https://arxiv.org/html/2305.14299v3#S5.SS5),
[official Qwen model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Completed pilot

Job59252 completed the full baseline and a24-step training/export check.
Baseline scores: SNIPS93.4286, ATIS88.1299, MASSIVE83.4902, HWU85.0602,
CLINC78.0000; mean85.6218. SNIPS/ATIS mean90.7792 differs from the paper's90.64;
per-task historical parity is not claimed. No prompt/ratio choices were tuned
against those scores.

The pilot used64 rows solely for an execution check. It exercised the original
custom loop, loss-component logging, optimizer updates, checkpoints and native
encoder export with maximum output difference0.0. Pilot scores are not a model
quality result. The full job uses all785870 rows, not this pilot subset.
