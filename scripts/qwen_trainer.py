"""Run the recovered CLTrainer loop on modern model implementations.

Only a single CPU/GPU in FP32 is supported. Modern Trainer supplies construction
and serialization; its train/training_step/optimizer/dataloader are never used.
The installed Transformers 4.2.1 source is parsed, not imported, so Qwen can use
modern Transformers without silently substituting its optimization behavior.
"""

# Imports below populate globals for the unchanged, AST-loaded legacy functions.
# ruff: noqa: F401
from __future__ import annotations

import ast
import collections
import hashlib
import logging
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from types import MethodType

import torch
from packaging import version
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import PreTrainedModel, Trainer, get_scheduler
from transformers.trainer_callback import (
    DefaultFlowCallback,
    TrainerCallback,
    TrainerState,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    IntervalStrategy,
    TrainOutput,
    set_seed,
    speed_metrics,
)

RECOVERED_SOURCE = (
    Path(__file__).resolve().parents[1] / "legacy/training/model/tadse/trainers.py"
)
RECOVERED_METHODS = ("train", "compute_loss", "_get_train_sampler", "_save_checkpoint")
SUPPORT_METHODS = (
    "create_optimizer_and_scheduler",
    "training_step",
    "get_train_dataloader",
    "_maybe_log_save_evaluate",
    "_prepare_inputs",
    "log",
    "store_flos",
    "_sorted_checkpoints",
    "_rotate_checkpoints",
)


def _load_nodes(path, class_name, names, namespace, audit):
    source = Path(path).read_text()
    tree = ast.parse(source)
    container = (
        tree.body
        if class_name is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    )
    nodes = [
        next(n for n in container if getattr(n, "name", None) == name) for name in names
    ]
    for node in nodes:
        key = f"{Path(path).name}:{class_name or 'module'}.{node.name}"
        audit["ast_sha256"][key] = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
    audit["source_sha256"][str(path)] = hashlib.sha256(source.encode()).hexdigest()
    # Postponed annotations allow old typing aliases without importing old modules.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 — trusted local source
    return {name: namespace[name] for name in names}


class QwenCLTrainer(Trainer):
    """Original TaDSE training loop with explicit version compatibility bridges."""

    def __init__(self, *args, legacy_transformers_source, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        invalid = []
        for name in (
            "fp16",
            "bf16",
            "deepspeed",
            "load_best_model_at_end",
            "gradient_checkpointing",
            "adafactor",
        ):
            if getattr(a, name, False):
                invalid.append(name)
        if a.world_size != 1 or a.n_gpu > 1:
            invalid.append("multiple devices")
        if a.gradient_accumulation_steps != 1:
            invalid.append("gradient_accumulation_steps != 1")
        if str(getattr(a.eval_strategy, "value", a.eval_strategy)) != "no":
            invalid.append("evaluation during training")
        if kwargs.get("optimizers", (None, None)) != (None, None):
            invalid.append("external optimizer")
        if getattr(a, "resume_from_checkpoint", False):
            invalid.append("resume")
        if any(
            p.dtype != torch.float32
            for p in self.model.parameters()
            if p.is_floating_point()
        ):
            invalid.append("non-FP32 model")
        if invalid:
            raise ValueError(
                "Recovered Qwen trainer does not support: " + ", ".join(invalid)
            )
        if (
            hashlib.sha256(RECOVERED_SOURCE.read_bytes()).hexdigest()
            != "f0db981279e20d9c8b095616a200125b85932bb3bbffff582bdf59ec1938848e"
        ):
            raise ValueError("Recovered CLTrainer source hash mismatch")
        root = Path(legacy_transformers_source)
        if not re.search(
            r"__version__\s*=\s*[\"\']4\.2\.1[\"\']", (root / "__init__.py").read_text()
        ):
            raise ValueError("legacy_transformers_source must be Transformers 4.2.1")
        self.legacy_audit = {
            "ast_sha256": {},
            "source_sha256": {},
            "compatibility_bridges": [
                "Modern Trainer construction and save_model only; original train is bound unchanged.",
                "Single-process local_rank normalized to -1 after modern device setup.",
                "Legacy args/AMP/sharded flags restored for supported FP32 branch.",
                "Original 4.2.1 AdamW, optimizer grouping, training step, dataloader, logging and flow callback loaded by AST.",
                "Modern linear scheduler API and unwrap identity used in single-device mode.",
            ],
        }
        namespace = dict(
            globals(),
            nn=torch.nn,
            Optimizer=torch.optim.Optimizer,
            logger=logging.getLogger(__name__),
            is_torch_tpu_available=lambda: False,
            _model_unwrap=lambda model: model,
            EvaluationStrategy=IntervalStrategy,
            reissue_pt_warnings=lambda caught: [
                warnings.warn(w.message, w.category) for w in caught
            ],
        )
        _load_nodes(
            root / "optimization.py", None, ("AdamW",), namespace, self.legacy_audit
        )
        support = _load_nodes(
            root / "trainer.py",
            "Trainer",
            SUPPORT_METHODS,
            namespace,
            self.legacy_audit,
        )
        recovered = _load_nodes(
            RECOVERED_SOURCE,
            "CLTrainer",
            RECOVERED_METHODS,
            namespace,
            self.legacy_audit,
        )
        callback = _load_nodes(
            root / "trainer_callback.py",
            None,
            ("DefaultFlowCallback",),
            namespace,
            self.legacy_audit,
        )["DefaultFlowCallback"]
        for name, method in {**support, **recovered}.items():
            setattr(self, name, MethodType(method, self))
        self.remove_callback(DefaultFlowCallback)
        self.add_callback(callback())
        a.local_rank = -1
        a.evaluation_strategy = IntervalStrategy.NO
        a.resume_from_checkpoint = False
        a.tpu_metrics_debug = False
        a.adafactor = False
        self.use_amp = self.use_apex = self.sharded_dpp = False
        self.deepspeed = None
