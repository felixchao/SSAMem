from __future__ import annotations

import argparse
from typing import Any


def build_training_args_from_config(config: dict[str, Any], args):
    training = dict(config.get("training") or {})
    runtime = dict(config.get("runtime") or {})
    kernel = dict(config.get("kernel") or {})
    lora = dict(training.get("lora") or {})
    alignment = dict(training.get("alignment") or training.get("ssa") or {})
    dpo = dict(training.get("dpo") or {})
    resolved = argparse.Namespace(**vars(args))
    resolved.runtime_mode = runtime.get("runtime_mode", "hf" if training.get("base_model") else args.runtime_mode)
    resolved.model_name_or_path = training.get("base_model") or runtime.get("model_name_or_path") or args.model_name_or_path
    resolved.trust_remote_code = bool(runtime.get("trust_remote_code", getattr(args, "trust_remote_code", False)))
    resolved.device = runtime.get("device", getattr(args, "device", "cpu"))
    resolved.torch_dtype = runtime.get(
        "torch_dtype",
        getattr(args, "torch_dtype", "float16" if resolved.runtime_mode == "hf" else "float32"),
    )
    resolved.load_in_4bit = bool(training.get("load_in_4bit", runtime.get("load_in_4bit", False)))
    resolved.storage_root = kernel.get("storage_root", getattr(args, "storage_root", None))
    resolved.cluster_assignment_threshold = float(
        kernel.get("cluster_assignment_threshold", getattr(args, "cluster_assignment_threshold", 0.95))
    )
    resolved.cluster_assignment_top_k = int(
        kernel.get("cluster_assignment_top_k", getattr(args, "cluster_assignment_top_k", 4))
    )
    resolved.lr = float(training.get("learning_rate", getattr(args, "lr", 1e-3)))
    resolved.epochs = int(training.get("epochs", getattr(args, "epochs", 1)))
    resolved.batch_size = int(training.get("batch_size", getattr(args, "batch_size", 1)))
    resolved.log_every = int(training.get("log_every", getattr(args, "log_every", 10)))
    resolved.grad_clip_norm = training.get("grad_clip_norm", None)
    resolved.history_path = training.get("history_path", getattr(args, "history_path", None))
    resolved.checkpoint = getattr(args, "checkpoint", None) or training.get("init_checkpoint")
    resolved.lora = lora
    resolved.alignment = alignment
    resolved.dpo = dpo
    return resolved
