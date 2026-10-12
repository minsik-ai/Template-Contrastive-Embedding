# Fixed top-2 checkpoint merge

Combine the exported original-stack, SimCSE-initialized top-2 non-MLP and MLP
seed42 models with one predetermined 50/50 parameter average. Both exports are
standalone BERT encoders. Average every floating tensor, including the trained
pooler; require identical non-floating buffers. No architecture mixing, new
training, ratio sweep or per-dataset model selection is performed.

The merge utility checks configurations, tokenizers, tensor names/shapes/dtypes,
source preservation and exact checkpoint reload. Evaluation uses the recovered
utterance-only protocol in the same legacy runtime as both parents. The merged
model is one ordinary BERT checkpoint with the existing inference interface.

This is a release candidate experiment, not a claim that merged performance
matches either parent. Parent models and the private Hub release remain unchanged.

Reference: Wortsman et al., [Model soups](https://proceedings.mlr.press/v162/wortsman22a.html).
