# All-intent SNIPS top-3, non-MLP TaDSE

Run seeds42,43,44 with the existing two-epoch TaDSE recipe, changing only SNIPS
from all-intent top2 to all-intent top3. Use the unchanged recovered augmentation
functions. No new NER, balancing, caps, deduplication, loss changes or sampling
changes. All other source CSV hashes must match the top2 corpus.

Expected counts: SNIPS440594; combined1111086. SNIPS counts in original intent
order are50770/188997/40437/39975/67786/16606/36023. The larger corpus means
138886 optimizer steps at two epochs and batch16 (top2:98234); this is an
equal-epoch recipe comparison, not a fixed-step ablation.

The array runner trains each seed, exports the trained head, then uses only
`scripts/evaluate_recovered.py`: utterance-only, original preprocessing and
reference selection, recovered HWU raw matching, no pattern inference.
Existing original-code provenance and source hash checks remain applicable.

Report all seeds, per-dataset mean/range and gains against the recovered SimCSE
baseline and matched top2/top5 seeds. Evaluate reliability per dataset rather
than using only the best overall seed. SimCSE HWU is81.0843; legacy top5 gains
0.1205/0.3614/0.4819 points, mean0.3213. Paper TaDSE HWU82.77 implies a1.69-point
gain over its reported SimCSE81.08. That gap remains unresolved; top3 is an
experiment, not an assumed fix.

SNIPS top2 itself was not weaker: three-seed mean97.5238 versus legacy top5
97.4286. Lower overall scores were mainly outside SNIPS. Legacy retains only
its final intent, so it is not a pure top5-versus-top2 comparison.

No GitHub push or replacement of the private Hub model is part of this run.
