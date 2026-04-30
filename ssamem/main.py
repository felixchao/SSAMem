from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ssamem.config import KernelConfig, MASConfig, PipelineConfig, RunConfig, RuntimeConfig


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Pointer-Driven SSAMem")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run an end-to-end standalone demo.")
    demo_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    demo_parser.add_argument("--model-name-or-path", default=None)
    demo_parser.add_argument("--trust-remote-code", action="store_true")
    demo_parser.add_argument("--device", default="cpu")
    demo_parser.add_argument("--top-k-prefetch", type=int, default=1)
    demo_parser.add_argument("--max-new-tokens", type=int, default=24)
    demo_parser.add_argument("--storage-root", default=None)
    add_shared_run_args(demo_parser)

    mas_parser = subparsers.add_parser("demo-mas", help="Run a MAS-style standalone demo.")
    mas_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    mas_parser.add_argument("--model-name-or-path", default=None)
    mas_parser.add_argument("--trust-remote-code", action="store_true")
    mas_parser.add_argument("--device", default="cpu")
    mas_parser.add_argument("--top-k-prefetch", type=int, default=1)
    mas_parser.add_argument("--max-new-tokens", type=int, default=24)
    mas_parser.add_argument("--storage-root", default=None)
    mas_parser.add_argument("--mas-style", default="camel", choices=["camel", "autogen", "debate", "macnet"])
    mas_parser.add_argument("--task-domain", default=None)
    add_shared_run_args(mas_parser)

    collect_traj_parser = subparsers.add_parser("collect-text-mas-trajectories", help="Run pure-text MAS and save trajectory JSONL.")
    collect_traj_parser.add_argument("--input", required=True, help="Input JSONL with task/context/target fields.")
    collect_traj_parser.add_argument("--output", required=True)
    collect_traj_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    collect_traj_parser.add_argument("--model-name-or-path", default=None)
    collect_traj_parser.add_argument("--trust-remote-code", action="store_true")
    collect_traj_parser.add_argument("--device", default="cpu")
    collect_traj_parser.add_argument("--torch-dtype", default=None, choices=["float16", "float32", "bfloat16"])
    collect_traj_parser.add_argument("--load-in-4bit", action="store_true")
    collect_traj_parser.add_argument("--limit", type=int, default=100)
    collect_traj_parser.add_argument("--mas-style", default="camel", choices=["camel", "autogen", "debate", "macnet"])
    collect_traj_parser.add_argument("--task-domain", default="popqa")
    collect_traj_parser.add_argument("--max-new-tokens", type=int, default=48)
    add_shared_run_args(collect_traj_parser)

    train_parser = subparsers.add_parser("train-ssa", help="Run a minimal SSA alignment demo.")
    train_parser.add_argument("--config", default=None)
    train_parser.add_argument("--manifest", default=None)
    train_parser.add_argument("--output-path", default=None)
    train_parser.add_argument("--checkpoint", default=None, help="Optional SSA checkpoint to initialize projector/composer.")
    train_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    train_parser.add_argument("--model-name-or-path", default=None)
    train_parser.add_argument("--trust-remote-code", action="store_true")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--steps", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--storage-root", default=None)
    train_parser.add_argument("--history-path", default=None)
    add_shared_run_args(train_parser)

    build_ssa_parser = subparsers.add_parser("build-ssa-data", help="Build SSA latent tensor shards and manifest.")
    build_ssa_parser.add_argument("--input", required=True)
    build_ssa_parser.add_argument("--output", required=True)
    build_ssa_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    build_ssa_parser.add_argument("--model-name-or-path", default=None)
    build_ssa_parser.add_argument("--trust-remote-code", action="store_true")
    build_ssa_parser.add_argument("--device", default="cpu")
    build_ssa_parser.add_argument("--torch-dtype", default=None, choices=["float16", "float32", "bfloat16"])
    build_ssa_parser.add_argument("--load-in-4bit", action="store_true")
    build_ssa_parser.add_argument("--latent-max-tokens", type=int, default=128)
    build_ssa_parser.add_argument("--no-latents", action="store_true")
    add_shared_run_args(build_ssa_parser)

    prepare_ssa_parser = subparsers.add_parser("prepare-ssa-traces", help="Prepare traces.jsonl for SSA data building.")
    prepare_ssa_parser.add_argument("--output", required=True)
    prepare_ssa_parser.add_argument(
        "--source",
        default="synthetic-api",
        choices=["synthetic-api", "hf", "popqa", "kodcode"],
        help="Use offline synthetic API records or convert a HuggingFace dataset.",
    )
    prepare_ssa_parser.add_argument("--limit", type=int, default=200)
    prepare_ssa_parser.add_argument("--dataset-name", default=None)
    prepare_ssa_parser.add_argument("--subset", default=None)
    prepare_ssa_parser.add_argument("--split", default="train")
    add_shared_run_args(prepare_ssa_parser)

    collect_dpo_parser = subparsers.add_parser("collect-dpo-prefs", help="Collect or synthesize pointer DPO preferences.")
    collect_dpo_parser.add_argument("--config", default=None)
    collect_dpo_parser.add_argument("--input", default=None)
    collect_dpo_parser.add_argument("--output", required=True)
    collect_dpo_parser.add_argument("--pointers", nargs="+", default=["<PTR_0x001>", "<PTR_0x002>"])
    collect_dpo_parser.add_argument("--positive-pointer", default=None)
    add_shared_run_args(collect_dpo_parser)

    dpo_parser = subparsers.add_parser("train-pointer-dpo", help="Train pointer router with TRL DPOTrainer.")
    dpo_parser.add_argument("--config", default=None)
    dpo_parser.add_argument("--preferences", default=None)
    dpo_parser.add_argument("--runtime-mode", default="hf", choices=["tiny-random", "hf"])
    dpo_parser.add_argument("--model-name-or-path", default=None)
    dpo_parser.add_argument("--trust-remote-code", action="store_true")
    dpo_parser.add_argument("--device", default="cuda")
    add_shared_run_args(dpo_parser)

    training_eval_parser = subparsers.add_parser("eval-training", help="Evaluate SSA latent distance and DPO data safety.")
    training_eval_parser.add_argument("--config", default=None)
    training_eval_parser.add_argument("--manifest", default=None)
    training_eval_parser.add_argument("--preferences", default=None)
    training_eval_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    training_eval_parser.add_argument("--model-name-or-path", default=None)
    training_eval_parser.add_argument("--trust-remote-code", action="store_true")
    training_eval_parser.add_argument("--device", default="cpu")
    training_eval_parser.add_argument("--checkpoint", default=None)
    training_eval_parser.add_argument("--limit", type=int, default=20)
    add_shared_run_args(training_eval_parser)

    split_ssa_parser = subparsers.add_parser("split-ssa-manifest", help="Split an SSA manifest into train/val/test manifests.")
    split_ssa_parser.add_argument("--manifest", required=True)
    split_ssa_parser.add_argument("--output", required=True)
    split_ssa_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_ssa_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_ssa_parser.add_argument("--test-ratio", type=float, default=0.1)
    add_shared_run_args(split_ssa_parser)

    kfold_ssa_parser = subparsers.add_parser("kfold-ssa-manifest", help="Create randomized K-fold train/val SSA manifests.")
    kfold_ssa_parser.add_argument("--manifest", required=True)
    kfold_ssa_parser.add_argument("--output", required=True)
    kfold_ssa_parser.add_argument("--folds", type=int, default=5)
    add_shared_run_args(kfold_ssa_parser)

    filter_ssa_parser = subparsers.add_parser("filter-ssa-manifest", help="Filter an SSA manifest by one metadata key/value.")
    filter_ssa_parser.add_argument("--manifest", required=True)
    filter_ssa_parser.add_argument("--output", required=True)
    filter_ssa_parser.add_argument("--metadata-key", default="source")
    filter_ssa_parser.add_argument("--metadata-value", required=True)
    add_shared_run_args(filter_ssa_parser)

    ssa_report_parser = subparsers.add_parser("eval-ssa-report", help="Compare SSA latent-distance metrics before and after training.")
    ssa_report_parser.add_argument("--config", default=None)
    ssa_report_parser.add_argument("--manifest", default=None)
    ssa_report_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    ssa_report_parser.add_argument("--model-name-or-path", default=None)
    ssa_report_parser.add_argument("--trust-remote-code", action="store_true")
    ssa_report_parser.add_argument("--device", default="cpu")
    ssa_report_parser.add_argument("--checkpoint", default=None)
    ssa_report_parser.add_argument("--limit", type=int, default=100)
    ssa_report_parser.add_argument("--output-path", default=None)
    add_shared_run_args(ssa_report_parser)

    ssa_probe_parser = subparsers.add_parser("probe-ssa-generation", help="Probe no-memory/text-memory/latent-memory generation quality.")
    ssa_probe_parser.add_argument("--config", default=None)
    ssa_probe_parser.add_argument("--manifest", default=None)
    ssa_probe_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    ssa_probe_parser.add_argument("--model-name-or-path", default=None)
    ssa_probe_parser.add_argument("--trust-remote-code", action="store_true")
    ssa_probe_parser.add_argument("--device", default="cpu")
    ssa_probe_parser.add_argument("--checkpoint", default=None)
    ssa_probe_parser.add_argument("--limit", type=int, default=10)
    ssa_probe_parser.add_argument("--max-new-tokens", type=int, default=48)
    ssa_probe_parser.add_argument("--max-keywords", type=int, default=8)
    ssa_probe_parser.add_argument("--output-path", default=None)
    add_shared_run_args(ssa_probe_parser)

    pointer_eval_parser = subparsers.add_parser("eval-pointer-routing", help="Evaluate pointer routing on preference data.")
    pointer_eval_parser.add_argument("--config", default=None)
    pointer_eval_parser.add_argument("--preferences", default=None)
    pointer_eval_parser.add_argument("--runtime-mode", default="hf", choices=["tiny-random", "hf"])
    pointer_eval_parser.add_argument("--model-name-or-path", default=None)
    pointer_eval_parser.add_argument("--trust-remote-code", action="store_true")
    pointer_eval_parser.add_argument("--device", default="cpu")
    pointer_eval_parser.add_argument("--limit", type=int, default=100)
    pointer_eval_parser.add_argument("--output-path", default=None)
    add_shared_run_args(pointer_eval_parser)

    suite_parser = subparsers.add_parser("run-ssa-suite", help="Split, train, eval, and probe SSA in one run.")
    suite_parser.add_argument("--config", default=None)
    suite_parser.add_argument("--manifest", default=None)
    suite_parser.add_argument("--suite-dir", required=True)
    suite_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    suite_parser.add_argument("--model-name-or-path", default=None)
    suite_parser.add_argument("--trust-remote-code", action="store_true")
    suite_parser.add_argument("--device", default="cpu")
    suite_parser.add_argument("--output-path", default=None)
    suite_parser.add_argument("--train-ratio", type=float, default=0.8)
    suite_parser.add_argument("--val-ratio", type=float, default=0.1)
    suite_parser.add_argument("--test-ratio", type=float, default=0.1)
    suite_parser.add_argument("--eval-limit", type=int, default=100)
    suite_parser.add_argument("--probe-limit", type=int, default=20)
    suite_parser.add_argument("--max-new-tokens", type=int, default=48)
    suite_parser.add_argument("--max-keywords", type=int, default=8)
    suite_parser.add_argument("--storage-root", default=None)
    add_shared_run_args(suite_parser)

    eval_parser = subparsers.add_parser("evaluate-triviaqa", help="Run a TriviaQA benchmark.")
    eval_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    eval_parser.add_argument("--model-name-or-path", default=None)
    eval_parser.add_argument("--trust-remote-code", action="store_true")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--top-k-prefetch", type=int, default=1)
    eval_parser.add_argument("--max-new-tokens", type=int, default=24)
    eval_parser.add_argument("--storage-root", default=None)
    eval_parser.add_argument("--split", default="validation")
    eval_parser.add_argument("--limit", type=int, default=20)
    eval_parser.add_argument("--max-snippets", type=int, default=3)
    eval_parser.add_argument("--max-evidence-chars", type=int, default=1600)
    eval_parser.add_argument("--latent-max-tokens", type=int, default=128)
    eval_parser.add_argument("--output-path", default=None)
    add_shared_run_args(eval_parser)

    return parser


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


def run_demo(args) -> None:
    import torch

    pipeline = build_pipeline_from_args(args)

    torch.manual_seed(args.seed)
    pointer_a = pipeline.register_memory_tensor(
        torch.randn(6, pipeline.userspace.hidden_size),
        metadata={"topic": "young-player-award", "source": "demo-memory-a"},
    )
    pointer_b = pipeline.register_memory_tensor(
        torch.randn(6, pipeline.userspace.hidden_size),
        metadata={"topic": "football-history", "source": "demo-memory-b"},
    )

    result = pipeline.run_message(
        sender_id="planner",
        receiver_id="userspace_role_1",
        content=f"請參考 {pointer_a} 回答問題：2006年第一位獲得最佳年輕球員獎的是誰？",
        top_k_prefetch=args.top_k_prefetch,
    )

    consolidation = pipeline.consolidate_episode(
        hidden_states=torch.randn(12, pipeline.userspace.hidden_size),
        global_reward=1.0,
        metadata={"episode": "demo"},
    )

    payload = {
        "registered_pointers": [pointer_a, pointer_b],
        "mounted_pointers": result.mounted_pointer_ids,
        "userspace_prompt": result.prompt,
        "userspace_output": result.output_text,
        "consolidated_pointer": consolidation.pointer,
        "prefetched_count": len(result.resolved_message.prefetched_hits),
        "storage_root": args.storage_root,
        "run_name": args.run_name or "demo",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "log_level": args.log_level,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_demo_mas_scenario(task_domain: str | None) -> tuple[str, str, dict]:
    if task_domain == "triviaqa":
        evidence_text = (
            "Lionel Messi won the FIFPro Best Young Player Award in 2006. "
            "He was recognized as the first winner of that award."
        )
        task_description = "Question: Who was the first winner of the 2006 Best Young Player Award?"
        metadata = {"topic": "triviaqa-demo", "question_type": "factoid"}
        return task_description, evidence_text, metadata

    if task_domain == "popqa":
        evidence_text = (
            "The capital city of Australia is Canberra. It is not Sydney or Melbourne."
        )
        task_description = "Question: What is the capital of Australia?"
        metadata = {"topic": "popqa-demo", "question_type": "factoid"}
        return task_description, evidence_text, metadata

    if task_domain == "pddl":
        evidence_text = (
            "Goal: move block A onto block B. Initial state: A is on the table, B is clear, "
            "the robot hand is empty."
        )
        task_description = "Plan a valid sequence of actions to move block A onto block B."
        metadata = {"topic": "pddl-demo", "task_type": "planning"}
        return task_description, evidence_text, metadata

    if task_domain == "kodcode":
        evidence_text = (
            "Implement a Python function that returns the first non-repeating character in a string. "
            "If no such character exists, return an empty string."
        )
        task_description = (
            "Write Python code for a function `first_unique_char(s: str) -> str` "
            "that returns the first non-repeating character."
        )
        metadata = {"topic": "kodcode-demo", "task_type": "coding"}
        return task_description, evidence_text, metadata

    if task_domain == "alfworld":
        evidence_text = (
            "You are in a kitchen. The mug is on the counter. The microwave is closed. "
            "The task is to heat the mug with the microwave."
        )
        task_description = "Choose the next valid ALFWorld action to make progress on heating the mug."
        metadata = {"topic": "alfworld-demo", "task_type": "action"}
        return task_description, evidence_text, metadata

    evidence_text = "This memory page contains a generic demo context for multi-agent coordination."
    task_description = "Please refer to the mounted memory, plan a response strategy, and provide the final answer."
    metadata = {"topic": "demo-mas", "task_type": "generic"}
    return task_description, evidence_text, metadata


def run_demo_mas(args) -> None:
    import torch

    pipeline = build_pipeline_from_args(args)
    torch.manual_seed(args.seed)
    task_description, evidence_text, metadata = build_demo_mas_scenario(args.task_domain)
    if args.task_domain:
        latent_tensor = pipeline.userspace.encode_text_as_latent(
            evidence_text,
            max_tokens=128,
            strategy="last_hidden",
        )
    else:
        latent_tensor = torch.randn(6, pipeline.userspace.hidden_size)
    pointer = pipeline.register_memory_tensor(latent_tensor, metadata=metadata)
    trace = pipeline.run_multi_agent_task(
        task_description=f"Please use {pointer} when helpful.\n\n{task_description}",
        mas_style=args.mas_style,
        task_domain=args.task_domain,
        top_k_prefetch=args.top_k_prefetch,
    )
    payload = {
        "architecture": trace.architecture,
        "task_domain": args.task_domain,
        "task_description": task_description,
        "registered_pointer": pointer,
        "turn_count": len(trace.turns),
        "final_output": trace.final_output,
        "roles": [turn.role for turn in trace.turns],
        "run_name": args.run_name or "demo-mas",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "log_level": args.log_level,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_synthetic_ssa_batches(hidden_size: int, steps: int) -> list[SSABatch]:
    import torch
    from ssamem.trainingspace import SSABatch

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


def run_train_ssa(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.trainingspace import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        load_alignment_checkpoint,
        ssa_collate,
        prepare_lora_for_training,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    pipeline = build_pipeline_from_args(args)
    lora_cfg = getattr(args, "lora", {})
    alignment_cfg = getattr(args, "alignment", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )
        if not bool(alignment_cfg.get("train_lora_backbone", False)):
            for param in pipeline.userspace.model.parameters():
                param.requires_grad = False

    student = LatentStudent(userspace=pipeline.userspace)
    teacher = ExplicitTeacher(userspace=pipeline.userspace)
    distiller = SSADistiller(
        student=student,
        teacher=teacher,
        loss_type=alignment_cfg.get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(alignment_cfg.get("distill_loss_div_std", True)),
        answer_loss_weight=float(alignment_cfg.get("answer_loss_weight", 0.0)),
        preference_loss_weight=float(alignment_cfg.get("preference_loss_weight", 0.0)),
        preference_beta=float(alignment_cfg.get("preference_beta", 0.1)),
    )
    if getattr(args, "checkpoint", None):
        load_alignment_checkpoint(distiller, args.checkpoint, map_location=args.device)

    optimizer = torch.optim.Adam((param for param in distiller.parameters() if param.requires_grad), lr=args.lr)
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if manifest_path:
        dataset = SSAManifestDataset(manifest_path, map_location=args.device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ssa_collate)
    else:
        dataloader = build_synthetic_ssa_batches(pipeline.userspace.hidden_size, args.steps)
    losses = distiller.fit(
        dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        grad_clip_norm=getattr(args, "grad_clip_norm", None),
        log_every=args.log_every,
    )
    output_path = args.output_path or deep_get(config, "training.output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_projection": distiller.student_projection.state_dict(),
                "memory_composer": distiller.student.memory_composer.state_dict(),
                "memory_projector": distiller.student.memory_projector.state_dict(),
                "history": losses,
                "config": config,
            },
            output_path,
        )
    history_path = getattr(args, "history_path", None)
    if history_path is None and output_path:
        history_path = str(Path(output_path).with_suffix(".history.json"))
    history_summary = summarize_metric_history(losses)
    if history_path:
        _write_json(history_path, {"history": losses, "summary": history_summary})
        _write_json(str(Path(history_path).with_suffix(".summary.json")), history_summary)
    if hasattr(pipeline.userspace.model, "save_pretrained") and deep_get(config, "training.adapter_output_dir"):
        pipeline.userspace.model.save_pretrained(deep_get(config, "training.adapter_output_dir"))
    print(
        json.dumps(
            {
                "steps": args.steps,
                "epochs": args.epochs,
                "history": losses,
                "history_path": history_path,
                "history_summary": history_summary,
                "checkpoint": output_path,
                "run_name": args.run_name or "train-ssa",
                "seed": args.seed,
                "output_dir": args.output_dir,
                "log_level": args.log_level,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_build_ssa_data(args) -> None:
    if not args.no_latents:
        import torch

        torch.manual_seed(args.seed)
        from ssamem.trainingspace import build_ssa_data_from_traces

        pipeline = build_pipeline_from_args(args)
        samples = build_ssa_data_from_traces(
            input_path=args.input,
            output_dir=args.output,
            userspace=pipeline.userspace,
            latent_max_tokens=args.latent_max_tokens,
        )
        sample_count = len(samples)
    else:
        from ssamem.ssa_data import build_ssa_manifest_without_latents

        sample_count = build_ssa_manifest_without_latents(input_path=args.input, output_dir=args.output)
    print(
        json.dumps(
            {
                "samples": sample_count,
                "manifest": str(Path(args.output) / "manifest.jsonl"),
                "latents_built": not args.no_latents,
                "run_name": args.run_name or "build-ssa-data",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_prepare_ssa_traces(args) -> None:
    from ssamem.ssa_data import (
        ssa_records_from_hf_dataset,
        ssa_records_from_kodcode,
        ssa_records_from_popqa,
        generate_synthetic_api_ssa_records,
        save_ssa_trace_records,
    )

    if args.source == "synthetic-api":
        records = generate_synthetic_api_ssa_records(limit=args.limit, seed=args.seed)
    elif args.source == "popqa":
        records = ssa_records_from_popqa(split=args.split, limit=args.limit)
    elif args.source == "kodcode":
        records = ssa_records_from_kodcode(split=args.split, limit=args.limit)
    else:
        if not args.dataset_name:
            raise SystemExit("--dataset-name is required when --source=hf.")
        records = ssa_records_from_hf_dataset(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            limit=args.limit,
        )
    save_ssa_trace_records(records, args.output)
    print(
        json.dumps(
            {
                "records": len(records),
                "output": args.output,
                "source": args.source,
                "dataset_name": args.dataset_name,
                "run_name": args.run_name or "prepare-ssa-traces",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _row_task_prompt(row: dict[str, Any], idx: int) -> str:
    return str(
        row.get("task_prompt")
        or row.get("task_description")
        or row.get("prompt")
        or row.get("question")
        or row.get("instruction")
        or f"Task {idx}"
    ).strip()


def _row_context_text(row: dict[str, Any]) -> str:
    context = row.get("context_text") or row.get("context") or row.get("memory")
    if context:
        return str(context).strip()
    answer = row.get("answer") or row.get("answers") or row.get("possible_answers")
    if answer:
        return f"Known answer aliases: {json.dumps(answer, ensure_ascii=False)}"
    return ""


def _row_target_text(row: dict[str, Any]) -> str | None:
    for key in ("target_text", "target_agent_output", "final_answer", "answer", "answers", "possible_answers"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value if value.strip() else None
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else None
        return str(value)
    return None


def _target_hit(target: str | None, output: str | None) -> bool:
    import re

    if not target or not output:
        return False
    target_norm = re.sub(r"[^a-z0-9 ]+", " ", str(target).lower()).strip()
    output_norm = re.sub(r"[^a-z0-9 ]+", " ", str(output).lower())
    return bool(target_norm and target_norm in output_norm)


def _latentmem_like_message_graph(trace, *, context_text: str) -> dict[str, Any]:
    nodes = []
    links = []
    for turn in trace.turns:
        nodes.append(
            {
                "id": turn.role,
                "message": {
                    "system_prompt_template": None,
                    "system_prompt_fields": {},
                    "user_prompt_template": turn.prompt,
                    "user_prompt_fields": {
                        "memory_content": context_text,
                        "task_description": trace.task_description,
                    },
                    "response": turn.response,
                    "state": {
                        "role": turn.role,
                        "architecture": trace.architecture,
                        "mounted_latent_count": turn.mounted_latent_count,
                        "mounted_latent_tokens": turn.mounted_latent_tokens,
                    },
                },
            }
        )
        for upstream_role in turn.upstream_roles:
            links.append({"source": upstream_role, "target": turn.role})
    return {
        "state": trace.task_description,
        "action": trace.final_output or "",
        "observation": "",
        "mas_message_graph_data": json.dumps(
            {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": nodes,
                "links": links,
            },
            ensure_ascii=False,
        ),
    }


def run_collect_text_mas_trajectories(args) -> None:
    import torch
    from transformers import GenerationConfig
    from ssamem.trainingspace import _jsonl_rows

    torch.manual_seed(args.seed)
    pipeline = build_pipeline_from_args(args)
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=pipeline.userspace.tokenizer.pad_token_id,
        eos_token_id=pipeline.userspace.tokenizer.eos_token_id,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    rewards: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(_jsonl_rows(args.input)):
            if idx >= args.limit:
                break
            task_prompt = _row_task_prompt(row, idx)
            context_text = _row_context_text(row)
            target_text = _row_target_text(row)
            trace = pipeline.run_multi_agent_task(
                task_description=task_prompt,
                mas_style=args.mas_style,
                task_domain=args.task_domain,
                memory_content=context_text,
                generation_config=generation_config,
                top_k_prefetch=0,
            )
            final_output = trace.final_output or ""
            reward = 1.0 if _target_hit(target_text, final_output) else 0.0
            rewards.append(reward)
            trajectory = [
                {
                    "agent": turn.role,
                    "prompt": turn.prompt,
                    "output": turn.response,
                    "upstream_roles": turn.upstream_roles,
                }
                for turn in trace.turns
            ]
            payload = {
                "task_id": row.get("trace_id") or row.get("id") or idx,
                "task_prompt": task_prompt,
                "context_text": context_text,
                "target_agent_output": target_text,
                "final_answer": final_output,
                "reward": reward,
                "mas_style": args.mas_style,
                "task_domain": args.task_domain,
                "trajectory": trajectory,
                "message_graph": _latentmem_like_message_graph(trace, context_text=context_text),
                "metadata": {
                    "source": "text_mas_rollout",
                    "input_index": idx,
                    "input_metadata": row.get("metadata") or {},
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            if count == 1 or count % 10 == 0:
                print(f"collect-text-mas-trajectories count={count} reward_mean={sum(rewards) / len(rewards):.4g}", flush=True)
    print(
        json.dumps(
            {
                "count": count,
                "output": str(output_path),
                "reward_mean": 0.0 if not rewards else sum(rewards) / len(rewards),
                "run_name": args.run_name or "collect-text-mas-trajectories",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_collect_dpo_prefs(args) -> None:
    from ssamem.ssa_data import (
        collect_synthetic_pointer_preferences,
        pointer_preferences_from_rollouts,
        save_pointer_preferences,
    )

    config = load_config_file(args.config) if args.config else {}
    prompts: list[str] = []
    input_path = args.input or deep_get(config, "data.prompts")
    rollout_path = deep_get(config, "data.rollouts")
    if input_path and str(input_path).endswith(".jsonl"):
        try:
            samples = pointer_preferences_from_rollouts(input_path)
        except ValueError:
            samples = []
        if samples:
            save_pointer_preferences(samples, args.output)
            print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))
            return
    if rollout_path:
        samples = pointer_preferences_from_rollouts(rollout_path)
        save_pointer_preferences(samples, args.output)
        print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))
        return
    if input_path:
        for row in Path(input_path).read_text(encoding="utf-8").splitlines():
            if row.strip():
                try:
                    payload = json.loads(row)
                    prompts.append(str(payload.get("prompt") or payload.get("task_prompt") or row))
                except json.JSONDecodeError:
                    prompts.append(row.strip())
    else:
        prompts = [
            "前端已完成，請實作稅率計算 API。",
            "Worker needs the API contract memory page before coding.",
        ]
    pointers = deep_get(config, "rollout.pointers", args.pointers)
    positive_pointer = args.positive_pointer or deep_get(config, "rollout.positive_pointer")
    samples = collect_synthetic_pointer_preferences(
        prompts=prompts,
        pointers=pointers,
        positive_pointer=positive_pointer,
    )
    save_pointer_preferences(samples, args.output)
    print(json.dumps({"preferences": len(samples), "output": args.output}, ensure_ascii=False, indent=2))


def run_train_pointer_dpo(args) -> None:
    import torch
    from ssamem.trainingspace import (
        initialize_pointer_vocabulary,
        load_pointer_preferences,
        prepare_lora_for_training,
        train_pointer_dpo,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    if not args.model_name_or_path and args.runtime_mode == "hf":
        raise SystemExit("train-pointer-dpo requires --model-name-or-path or training.base_model in config.")
    pipeline = build_pipeline_from_args(args)
    initialize_pointer_vocabulary(pipeline.userspace.tokenizer, pipeline.userspace.model)
    lora_cfg = getattr(args, "lora", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if not preferences_path:
        raise SystemExit("train-pointer-dpo requires --preferences or data.preferences in config.")
    output_dir = args.output_dir or deep_get(config, "training.dpo.output_dir") or "pointer_dpo_adapter"
    dpo_cfg = getattr(args, "dpo", {})
    result = train_pointer_dpo(
        model=pipeline.userspace.model,
        tokenizer=pipeline.userspace.tokenizer,
        preference_samples=load_pointer_preferences(preferences_path),
        output_dir=output_dir,
        beta=float(dpo_cfg.get("beta", 0.1)),
        max_prompt_length=int(dpo_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(dpo_cfg.get("max_completion_length", 16)),
        per_device_train_batch_size=int(dpo_cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(dpo_cfg.get("gradient_accumulation_steps", 32)),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
        num_train_epochs=float(dpo_cfg.get("num_train_epochs", 1.0)),
    )
    if hasattr(pipeline.userspace.model, "save_pretrained"):
        pipeline.userspace.model.save_pretrained(output_dir)
    print(json.dumps({"output_dir": output_dir, "train_result": str(result)}, ensure_ascii=False, indent=2))


def run_eval_training(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.trainingspace import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        ssa_collate,
        load_pointer_preferences,
    )

    torch.manual_seed(args.seed)
    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    payload: dict[str, Any] = {}
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if manifest_path:
        pipeline = build_pipeline_from_args(args)
        distiller = SSADistiller(
            student=LatentStudent(pipeline.userspace),
            teacher=ExplicitTeacher(pipeline.userspace),
        )
        distiller.to(args.device)
        checkpoint_path = getattr(args, "checkpoint", None)
        if checkpoint_path:
            load_alignment_checkpoint(distiller, checkpoint_path, map_location=args.device)
            payload["checkpoint"] = checkpoint_path
            payload["checkpoint_loaded"] = True
        else:
            payload["checkpoint_loaded"] = False
        dataset = SSAManifestDataset(manifest_path, map_location=args.device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ssa_collate)
        distances = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= args.limit:
                    break
                distances.append(float(distiller.compute_step(batch).latent_distance.cpu().item()))
        payload["latent_distance_mean"] = sum(distances) / max(len(distances), 1)
        payload["latent_distance_count"] = len(distances)
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if preferences_path:
        prefs = load_pointer_preferences(preferences_path)
        payload["pointer_preferences"] = len(prefs)
        payload["pointer_preferences_valid"] = True
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_ssa_distiller_for_eval(args):
    from ssamem.trainingspace import SSADistiller, ExplicitTeacher, LatentStudent

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


def run_split_ssa_manifest(args) -> None:
    from ssamem.trainingspace import split_ssa_manifest

    payload = split_ssa_manifest(
        args.manifest,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_kfold_ssa_manifest(args) -> None:
    from ssamem.trainingspace import kfold_ssa_manifest

    payload = kfold_ssa_manifest(
        args.manifest,
        output_dir=args.output,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_filter_ssa_manifest(args) -> None:
    from ssamem.trainingspace import filter_ssa_manifest

    payload = filter_ssa_manifest(
        args.manifest,
        output_path=args.output,
        metadata_key=args.metadata_key,
        metadata_value=args.metadata_value,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_eval_ssa_report(args) -> None:
    from ssamem.trainingspace import (
        SSAManifestDataset,
        evaluate_ssa_latent_distances,
        load_alignment_checkpoint,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("eval-ssa-report requires --manifest or data.manifest in config.")

    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    _, baseline_distiller = _build_ssa_distiller_for_eval(args)
    baseline = evaluate_ssa_latent_distances(distiller=baseline_distiller, dataset=dataset, limit=args.limit)

    payload: dict[str, Any] = {
        "manifest": manifest_path,
        "limit": args.limit,
        "baseline": baseline,
        "checkpoint_loaded": False,
    }
    checkpoint_path = getattr(args, "checkpoint", None) or deep_get(config, "training.output_path")
    if checkpoint_path:
        _, trained_distiller = _build_ssa_distiller_for_eval(args)
        load_alignment_checkpoint(trained_distiller, checkpoint_path, map_location=args.device)
        checkpoint_metrics = evaluate_ssa_latent_distances(distiller=trained_distiller, dataset=dataset, limit=args.limit)
        baseline_mean = float(baseline["summary"]["latent_distance"]["mean"])
        trained_mean = float(checkpoint_metrics["summary"]["latent_distance"]["mean"])
        improvement_absolute = baseline_mean - trained_mean
        improvement_relative = 0.0 if baseline_mean == 0.0 else improvement_absolute / baseline_mean
        payload.update(
            {
                "checkpoint": checkpoint_path,
                "checkpoint_loaded": True,
                "checkpoint_metrics": checkpoint_metrics,
                "improvement": {
                    "absolute": improvement_absolute,
                    "relative": improvement_relative,
                },
            }
        )
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_probe_ssa_generation(args) -> None:
    from ssamem.trainingspace import SSAManifestDataset, load_alignment_checkpoint, probe_ssa_generation

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("probe-ssa-generation requires --manifest or data.manifest in config.")
    checkpoint_path = getattr(args, "checkpoint", None) or deep_get(config, "training.output_path")
    pipeline, distiller = _build_ssa_distiller_for_eval(args)
    if checkpoint_path:
        load_alignment_checkpoint(distiller, checkpoint_path, map_location=args.device)
    dataset = SSAManifestDataset(manifest_path, map_location=args.device)
    payload = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=dataset,
        student=distiller.student if checkpoint_path else None,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_keywords=args.max_keywords,
    )
    payload["manifest"] = manifest_path
    payload["checkpoint"] = checkpoint_path
    payload["checkpoint_loaded"] = bool(checkpoint_path)
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_eval_pointer_routing(args) -> None:
    from ssamem.trainingspace import (
        evaluate_pointer_routing,
        initialize_pointer_vocabulary,
        load_pointer_preferences,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    preferences_path = args.preferences or deep_get(config, "data.preferences")
    if not preferences_path:
        raise SystemExit("eval-pointer-routing requires --preferences or data.preferences in config.")
    preferences = load_pointer_preferences(preferences_path)
    payload = {"preferences": preferences_path}
    if args.runtime_mode == "hf" and args.model_name_or_path:
        pipeline = build_pipeline_from_args(args)
        initialize_pointer_vocabulary(pipeline.userspace.tokenizer, pipeline.userspace.model)
        payload.update(
            evaluate_pointer_routing(
                preferences,
                model=pipeline.userspace.model,
                tokenizer=pipeline.userspace.tokenizer,
                device=args.device,
                seed=args.seed,
                limit=args.limit,
            )
        )
    else:
        payload.update(evaluate_pointer_routing(preferences, seed=args.seed, limit=args.limit))
    output_path = resolve_output_path(args)
    if output_path:
        _write_json(output_path, payload)
        payload["output_path"] = output_path
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_ssa_suite(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from ssamem.trainingspace import (
        SSADistiller,
        SSAManifestDataset,
        ExplicitTeacher,
        LatentStudent,
        ssa_collate,
        evaluate_ssa_latent_distances,
        load_alignment_checkpoint,
        prepare_lora_for_training,
        probe_ssa_generation,
        split_ssa_manifest,
    )

    config = load_config_file(args.config) if args.config else {}
    args = build_training_args_from_config(config, args) if config else args
    manifest_path = args.manifest or deep_get(config, "data.manifest")
    if not manifest_path:
        raise SystemExit("run-ssa-suite requires --manifest or data.manifest in config.")

    suite_root = Path(args.suite_dir)
    suite_root.mkdir(parents=True, exist_ok=True)
    splits_dir = suite_root / "splits"
    reports_dir = suite_root / "reports"
    checkpoints_dir = suite_root / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    split_payload = split_ssa_manifest(
        manifest_path,
        output_dir=splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    _write_json(reports_dir / "split_summary.json", split_payload)

    train_manifest = split_payload["splits"]["train"]["manifest"]
    val_manifest = split_payload["splits"]["val"]["manifest"]
    test_manifest = split_payload["splits"]["test"]["manifest"]

    torch.manual_seed(args.seed)
    pipeline = build_pipeline_from_args(args)
    lora_cfg = getattr(args, "lora", {})
    if lora_cfg and args.runtime_mode == "hf":
        pipeline.userspace._runtime.model = prepare_lora_for_training(
            pipeline.userspace.model,
            r=int(lora_cfg.get("r", 128)),
            alpha=int(lora_cfg.get("alpha", 32)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        )

    distiller = SSADistiller(
        student=LatentStudent(userspace=pipeline.userspace),
        teacher=ExplicitTeacher(userspace=pipeline.userspace),
        loss_type=getattr(args, "alignment", {}).get("loss_type", "smooth_l1"),
        distill_loss_div_std=bool(getattr(args, "alignment", {}).get("distill_loss_div_std", True)),
    )
    optimizer = torch.optim.Adam(distiller.parameters(), lr=args.lr)
    train_dataset = SSAManifestDataset(train_manifest, map_location=args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ssa_collate)
    history = distiller.fit(
        train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        log_every=args.log_every,
    )
    checkpoint_path = checkpoints_dir / "ssa_distiller.pt"
    torch.save(
        {
            "student_projection": distiller.student_projection.state_dict(),
            "memory_composer": distiller.student.memory_composer.state_dict(),
            "memory_projector": distiller.student.memory_projector.state_dict(),
            "history": history,
            "config": config,
        },
        checkpoint_path,
    )
    if hasattr(pipeline.userspace.model, "save_pretrained"):
        adapter_dir = checkpoints_dir / "adapter"
        pipeline.userspace.model.save_pretrained(adapter_dir)
    _write_json(
        reports_dir / "train_history.json",
        {
            "history": history,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint": str(checkpoint_path),
        },
    )

    def _paired_report(split_name: str, split_manifest_path: str) -> dict[str, Any]:
        dataset = SSAManifestDataset(split_manifest_path, map_location=args.device)
        _, baseline_distiller = _build_ssa_distiller_for_eval(args)
        baseline = evaluate_ssa_latent_distances(distiller=baseline_distiller, dataset=dataset, limit=args.eval_limit)
        _, trained_distiller = _build_ssa_distiller_for_eval(args)
        load_alignment_checkpoint(trained_distiller, checkpoint_path, map_location=args.device)
        trained = evaluate_ssa_latent_distances(distiller=trained_distiller, dataset=dataset, limit=args.eval_limit)
        baseline_mean = float(baseline["summary"]["mean"])
        trained_mean = float(trained["summary"]["mean"])
        return {
            "split": split_name,
            "manifest": split_manifest_path,
            "baseline": baseline,
            "checkpoint_metrics": trained,
            "improvement": {
                "absolute": baseline_mean - trained_mean,
                "relative": 0.0 if baseline_mean == 0.0 else (baseline_mean - trained_mean) / baseline_mean,
            },
        }

    val_report = _paired_report("val", val_manifest)
    test_report = _paired_report("test", test_manifest)
    _write_json(reports_dir / "val_report.json", val_report)
    _write_json(reports_dir / "test_report.json", test_report)

    val_probe = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=SSAManifestDataset(val_manifest, map_location=args.device),
        limit=args.probe_limit,
        max_new_tokens=getattr(args, "max_new_tokens", 48),
        max_keywords=args.max_keywords,
    )
    test_probe = probe_ssa_generation(
        userspace=pipeline.userspace,
        dataset=SSAManifestDataset(test_manifest, map_location=args.device),
        limit=args.probe_limit,
        max_new_tokens=getattr(args, "max_new_tokens", 48),
        max_keywords=args.max_keywords,
    )
    _write_json(reports_dir / "val_probe.json", val_probe)
    _write_json(reports_dir / "test_probe.json", test_probe)

    summary = {
        "suite_dir": str(suite_root),
        "checkpoint": str(checkpoint_path),
        "split_summary": split_payload,
        "val_improvement": val_report["improvement"],
        "test_improvement": test_report["improvement"],
        "val_probe_summary": {
            "no_memory": val_probe["no_memory"],
            "text_memory": val_probe["text_memory"],
            "latent_memory": val_probe["latent_memory"],
        },
        "test_probe_summary": {
            "no_memory": test_probe["no_memory"],
            "text_memory": test_probe["text_memory"],
            "latent_memory": test_probe["latent_memory"],
        },
    }
    summary_path = args.output_path or (suite_root / "suite_summary.json")
    _write_json(summary_path, summary)
    summary["output_path"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_evaluate_triviaqa(args) -> None:
    from ssamem.evaluate_triviaqa import evaluate_triviaqa

    pipeline = build_pipeline_from_args(args)
    summary = evaluate_triviaqa(
        pipeline=pipeline,
        split=args.split,
        limit=args.limit,
        max_snippets=args.max_snippets,
        max_evidence_chars=args.max_evidence_chars,
        top_k_prefetch=args.top_k_prefetch,
        latent_max_tokens=args.latent_max_tokens,
        output_path=resolve_output_path(args),
    )
    summary["run_name"] = args.run_name or "evaluate-triviaqa"
    summary["seed"] = args.seed
    summary["output_dir"] = args.output_dir
    summary["log_level"] = args.log_level
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch_free_commands = {"prepare-ssa-traces", "collect-dpo-prefs", "filter-ssa-manifest"}
    if args.command == "build-ssa-data" and getattr(args, "no_latents", False):
        torch_free_commands.add("build-ssa-data")
    if args.command not in torch_free_commands:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "This ssamem command requires `torch` and `transformers` in the active Python environment. "
                "Install dependencies from `ssamem/requirements.txt` first."
            ) from exc
    configure_runtime(args)
    if args.output_dir:
        RunConfig(output_dir=args.output_dir).ensure_output_dir()

    if args.command == "demo":
        run_demo(args)
    elif args.command == "demo-mas":
        run_demo_mas(args)
    elif args.command == "collect-text-mas-trajectories":
        run_collect_text_mas_trajectories(args)
    elif args.command == "build-ssa-data":
        run_build_ssa_data(args)
    elif args.command == "prepare-ssa-traces":
        run_prepare_ssa_traces(args)
    elif args.command == "train-ssa":
        run_train_ssa(args)
    elif args.command == "collect-dpo-prefs":
        run_collect_dpo_prefs(args)
    elif args.command == "train-pointer-dpo":
        run_train_pointer_dpo(args)
    elif args.command == "eval-training":
        run_eval_training(args)
    elif args.command == "split-ssa-manifest":
        run_split_ssa_manifest(args)
    elif args.command == "kfold-ssa-manifest":
        run_kfold_ssa_manifest(args)
    elif args.command == "filter-ssa-manifest":
        run_filter_ssa_manifest(args)
    elif args.command == "eval-ssa-report":
        run_eval_ssa_report(args)
    elif args.command == "probe-ssa-generation":
        run_probe_ssa_generation(args)
    elif args.command == "eval-pointer-routing":
        run_eval_pointer_routing(args)
    elif args.command == "run-ssa-suite":
        run_ssa_suite(args)
    elif args.command == "evaluate-triviaqa":
        run_evaluate_triviaqa(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
