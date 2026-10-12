> Historical reconstruction notes. For the public model and its exact training
> corpus, use the [released-model reproduction guide](released-model.md).
> Earlier runners and evaluation protocols below are retained as experiment history.

# Reconstructed dependency environment

Use **Python 3.11** in a new virtual environment. `requirements.txt` pins the
direct training, tabular-data, and retrieval dependencies; it is a reconstructed
baseline, not a recovered lockfile from the paper experiments. Transitive
dependencies remain resolver-selected. Save `pip freeze`, Python version, GPU
model, driver version, and the repository commit with each experiment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > environment.freeze.txt
```

On a GPU worker, choose a Torch wheel compatible with the available driver before
installing the remaining requirements. For example, CUDA 12.6:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

PyTorch lists CPU, CUDA 11.8, 12.6, and 12.8 packages for this release in its
[official installation table](https://pytorch.org/get-started/previous-versions/).
Do not run training workloads on a shared cluster login node.

## Why these versions

- The repository imports `Qwen3Model` unconditionally, including for BERT runs.
  The [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-4B) requires
  Transformers 4.51 or newer. Transformers 4.51.3 supplies this architecture
  while retaining imports used by the older scripts.
- Accelerate 1.6.0 supports the stock Transformers Trainer in this baseline.
- Datasets 3.5.1 and PyArrow 19.0.1 provide CSV/JSON/Parquet loading. Explicit
  NumPy 1.26.4 avoids changing numerical/ABI generations while reconstructing
  the experiments.
- No vision/audio stack, FAISS, Apex, DeepSpeed, or external LLM API is necessary
  for the basic embedding and deterministic slot-substitution workflow.

Published package metadata was checked for the listed versions, and `uv pip
compile requirements.txt --python-version 3.11 --python-platform
x86_64-unknown-linux-gnu` resolved successfully. Installation,
model forward/backward execution, checkpoint reload, and GPU execution are
separate checks; resolving packages alone does not establish paper reproduction.

## Legacy utilities and missing pieces

| Utility | Additional requirement or blocker |
| --- | --- |
| `data/massive/extract_ptn_clinc.py` | Requires spaCy and the English large NER pipeline. Optional compatible pair: `spacy==3.7.5` and `en_core_web_lg==3.7.1`. The script unconditionally calls `spacy.require_gpu()`; CPU use needs a code change and GPU use needs a matching CuPy installation. It also slices a set at the final summary, which raises `TypeError`. |
| `data/massive/accuracy_infer.py` | Imports `python-dotenv` and `google-genai`, then constructs a client from `GOOGLE_API_KEY` at import time. Even local evaluation therefore requires refactoring or valid API configuration. These optional cloud dependencies are excluded from the offline baseline. |
| `eval/calc_align_uniform.py`, `eval/repr_tsne.py` | Depend on PySpark and unavailable project-specific `alexa_dr_semantic_similarity.utils.spark_builder`. Installing PySpark alone does not repair these scripts. |
| Vendored `data/massive/simcse/trainers.py` | Depends on SentEval and external benchmark data; not required by the lightweight `simcse.tool` embedding helper. |
| Original `model/tadse/trainers.py` | Contains an old copied Trainer loop. Its epoch-end `_maybe_log_save_evaluate` call has a different signature from its step-end call, and evaluation references undefined `PATH_TO_DATA`. Dependency pinning cannot repair these code paths. Use the reconstructed stock-Trainer runner for smoke experiments. |

Optional CPU spaCy setup (the original extractor still needs its forced-GPU call
removed before CPU execution):

```bash
python -m pip install spacy==3.7.5
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
```

The reconstructed augmentation path uses bundled slot annotations and does not
require spaCy or an external API. Original NER-pipeline versions, full synthetic
training artifacts, exact training configurations, and original environment
locks must be recovered separately before claiming an exact reproduction.
