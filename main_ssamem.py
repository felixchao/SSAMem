from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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

    train_parser = subparsers.add_parser("train-codi", help="Run a minimal CODi distillation demo.")
    train_parser.add_argument("--runtime-mode", default="tiny-random", choices=["tiny-random", "hf"])
    train_parser.add_argument("--model-name-or-path", default=None)
    train_parser.add_argument("--trust-remote-code", action="store_true")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--steps", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--storage-root", default=None)
    add_shared_run_args(train_parser)

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


def build_pipeline_from_args(args):
    from ssamem.pipeline import PointerDrivenSSAMemPipeline

    config = PipelineConfig(
        runtime=RuntimeConfig(
            runtime_mode=args.runtime_mode,
            model_name_or_path=args.model_name_or_path,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            device=args.device,
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


def build_synthetic_codi_batches(hidden_size: int, steps: int):
    import torch
    from ssamem.trainingspace import CODiBatch

    batches = []
    for step in range(steps):
        latent_tensor = torch.randn(8, hidden_size)
        explicit_text = (
            f"Thought {step}: read the evidence, extract the answer span, "
            f"then provide the final short answer."
        )
        batches.append(
            CODiBatch(
                latent_tensors=[latent_tensor],
                explicit_cot_texts=[explicit_text],
                student_prompts=["Decode the mounted latent memory into a compact reasoning state."],
            )
        )
    return batches


def run_train_codi(args) -> None:
    import torch
    from ssamem.trainingspace import CODiDistiller, ExplicitTeacher, LatentStudent

    torch.manual_seed(args.seed)
    pipeline = build_pipeline_from_args(args)
    student = LatentStudent(userspace=pipeline.userspace)
    teacher = ExplicitTeacher(userspace=pipeline.userspace)
    distiller = CODiDistiller(student=student, teacher=teacher)

    optimizer = torch.optim.Adam(distiller.parameters(), lr=args.lr)
    batches = build_synthetic_codi_batches(pipeline.userspace.hidden_size, args.steps)
    losses = distiller.fit(batches, optimizer=optimizer, epochs=1, device=args.device)
    print(
        json.dumps(
            {
                "steps": args.steps,
                "losses": losses,
                "run_name": args.run_name or "train-codi",
                "seed": args.seed,
                "output_dir": args.output_dir,
                "log_level": args.log_level,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ssamem requires `torch` and `transformers` in the active Python environment. "
            "Install dependencies from `ssamem/requirements.txt` first."
        ) from exc

    parser = build_parser()
    args = parser.parse_args()
    configure_runtime(args)
    if args.output_dir:
        RunConfig(output_dir=args.output_dir).ensure_output_dir()

    if args.command == "demo":
        run_demo(args)
    elif args.command == "demo-mas":
        run_demo_mas(args)
    elif args.command == "train-codi":
        run_train_codi(args)
    elif args.command == "evaluate-triviaqa":
        run_evaluate_triviaqa(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
