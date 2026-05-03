from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ssamem.config import KernelConfig, MASConfig, PipelineConfig, RunConfig, RuntimeConfig


def configure_runtime(args) -> None:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s %(message)s")

def resolve_output_path(args) -> str | None:
    if getattr(args, "output_path", None):
        return args.output_path
    if not args.output_dir:
        return None
    run_name = args.run_name or str(args.command)
    return str(Path(args.output_dir) / f"{run_name}.json")

def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise SystemExit("YAML configs require `pyyaml`; use JSON or install pyyaml.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return dict(payload or {})

def deep_get(mapping: dict[str, Any], path: str, default=None):
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current

def summarize_metric_history(history: list[dict[str, float]], *, tail_count: int = 50) -> dict[str, Any]:
    if not history:
        return {"steps": 0, "metrics": {}}
    metric_names = sorted({name for row in history for name in row})
    metrics = {}
    for name in metric_names:
        values = [float(row[name]) for row in history if name in row]
        tail = values[-tail_count:]
        metrics[name] = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean_tail": sum(tail) / len(tail),
        }
    return {"steps": len(history), "tail_count": tail_count, "metrics": metrics}

def build_pipeline_from_args(args) -> PointerDrivenSSAMemPipeline:
    from ssamem.pipeline import PointerDrivenSSAMemPipeline

    config = PipelineConfig(
        runtime=RuntimeConfig(
            runtime_mode=args.runtime_mode,
            model_name_or_path=args.model_name_or_path,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            torch_dtype=getattr(args, "torch_dtype", None) or ("float16" if args.runtime_mode == "hf" else "float32"),
            device=args.device,
            load_in_4bit=getattr(args, "load_in_4bit", False),
        ),
        kernel=KernelConfig(
            top_k_prefetch=getattr(args, "top_k_prefetch", 1),
            storage_root=getattr(args, "storage_root", None),
        ),
        mas=MASConfig(
            architecture=getattr(args, "mas_style", "camel"),
            task_domain=getattr(args, "task_domain", None),
        ),
        run=RunConfig(
            seed=getattr(args, "seed", 7),
            run_name=getattr(args, "run_name", None),
            output_dir=getattr(args, "output_dir", None),
            log_level=getattr(args, "log_level", "INFO"),
        ),
        max_new_tokens=getattr(args, "max_new_tokens", 32),
    )
    return PointerDrivenSSAMemPipeline.from_config(config)

def build_synthetic_ssa_batches(hidden_size: int, steps: int) -> list[SSABatch]:
    import torch
    from ssamem.training.space import SSABatch

    batches = []
    for step in range(steps):
        latent_tensor = torch.randn(8, hidden_size)
        explicit_text = (
            f"Thought {step}: read the evidence, extract the answer span, "
            f"then provide the final short answer."
        )
        batches.append(
            SSABatch(
                latent_tensors=[latent_tensor],
                explicit_cot_texts=[explicit_text],
                student_prompts=["Decode the mounted latent memory into a compact reasoning state."],
            )
        )
    return batches

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

def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_ssa_distiller_for_eval(args):
    from ssamem.training.space import SSADistiller, ExplicitTeacher, LatentStudent

    pipeline = build_pipeline_from_args(args)
    alignment_cfg = getattr(args, "alignment", {})
    distiller = SSADistiller(
        student=LatentStudent(pipeline.userspace),
        teacher=ExplicitTeacher(pipeline.userspace),
        loss_type=alignment_cfg.get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(alignment_cfg.get("distill_loss_div_std", True)),
        answer_loss_weight=float(alignment_cfg.get("answer_loss_weight", 0.0)),
    )
    distiller.to(args.device)
    return pipeline, distiller
