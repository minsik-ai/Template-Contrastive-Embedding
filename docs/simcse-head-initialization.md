# SimCSE head initialization ablation

Control: recovered original-stack TaDSE with pattern MLP, legacy data, seed42.
Its recovered-evaluation mean is83.1250 (HWU81.5663, CLINC71.4545).

Intervention: immediately after the original BERT checkpoint load, copy
`bert.pooler.dense.{weight,bias}` into `mlp.dense.{weight,bias}`. SimCSE's exported
pooler contains its trained contrastive projection. TaDSE's `mlp` is shared by
utterance and template views; its separate `ptn_transform` remains randomly
initialized exactly as in the control. This is not an inference-only head change.

`scripts/run_simcse_head_ablation.py` wraps the unchanged original entry point.
It intercepts only the first BERT load, applies the explicit copy, and records
all changed tensors plus CPU/CUDA RNG checks. Later checkpoint reloads are not
modified. Original source hashes are verified. No original files are edited.

The full runner's optional fourth argument `simcse-head` enables this ablation;
omitting it keeps original initialization. Submit array index1 only for the MLP
variant. Training data, seed, losses, two epochs, learning rate, batch size and
recovered evaluation are identical to the control. Use the same trained-head
export in both cases. No private release replacement or GitHub push occurs.

The fixed-seed comparison measures this initialization intervention. It does not
establish that the paper used it, nor demonstrate robustness across seeds.
