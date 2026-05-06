from __future__ import annotations

from ssamem.cli import build_parser
from ssamem.commands.common import configure_runtime
from ssamem.commands.data import (
    run_build_ssa_data,
    run_collect_dpo_prefs,
    run_filter_ssa_manifest,
    run_kfold_ssa_manifest,
    run_make_multimemory_ssa_manifest,
    run_prepare_ssa_traces,
    run_split_ssa_manifest,
)
from ssamem.commands.demo import run_collect_text_mas_trajectories, run_demo, run_demo_mas
from ssamem.commands.evaluation import (
    run_build_lmpo_pairs,
    run_collect_lmpo_rollouts,
    run_eval_pointer_routing,
    run_eval_ssa_report,
    run_eval_training,
    run_evaluate_triviaqa,
    run_probe_ssa_generation,
    run_ssa_suite,
)
from ssamem.commands.mas import (
    run_build_experience_bank,
    run_eval_mas_memory_agent_loop,
    run_eval_mas_latent_memory,
    run_eval_mas_memory_search,
    run_query_memory_agent,
)
from ssamem.commands.retrieval import run_eval_retrieval, run_train_retrieval
from ssamem.commands.training import run_train_lmpo_projector, run_train_pointer_dpo, run_train_ssa
from ssamem.config import RunConfig


TORCH_FREE_COMMANDS = {
    "prepare-ssa-traces",
    "collect-dpo-prefs",
    "filter-ssa-manifest",
    "make-multimemory-ssa-manifest",
    "build-lmpo-pairs",
}


COMMAND_HANDLERS = {
    "demo": run_demo,
    "demo-mas": run_demo_mas,
    "collect-text-mas-trajectories": run_collect_text_mas_trajectories,
    "build-ssa-data": run_build_ssa_data,
    "prepare-ssa-traces": run_prepare_ssa_traces,
    "train-ssa": run_train_ssa,
    "train-retrieval": run_train_retrieval,
    "eval-retrieval": run_eval_retrieval,
    "collect-dpo-prefs": run_collect_dpo_prefs,
    "train-pointer-dpo": run_train_pointer_dpo,
    "eval-training": run_eval_training,
    "split-ssa-manifest": run_split_ssa_manifest,
    "kfold-ssa-manifest": run_kfold_ssa_manifest,
    "filter-ssa-manifest": run_filter_ssa_manifest,
    "make-multimemory-ssa-manifest": run_make_multimemory_ssa_manifest,
    "eval-ssa-report": run_eval_ssa_report,
    "probe-ssa-generation": run_probe_ssa_generation,
    "eval-mas-latent-memory": run_eval_mas_latent_memory,
    "build-experience-bank": run_build_experience_bank,
    "eval-mas-memory-search": run_eval_mas_memory_search,
    "eval-mas-memory-agent-loop": run_eval_mas_memory_agent_loop,
    "query-memory-agent": run_query_memory_agent,
    "collect-lmpo-rollouts": run_collect_lmpo_rollouts,
    "build-lmpo-pairs": run_build_lmpo_pairs,
    "train-lmpo-projector": run_train_lmpo_projector,
    "eval-pointer-routing": run_eval_pointer_routing,
    "run-ssa-suite": run_ssa_suite,
    "evaluate-triviaqa": run_evaluate_triviaqa,
}


def _requires_torch(args) -> bool:
    if args.command == "build-ssa-data" and getattr(args, "no_latents", False):
        return False
    return args.command not in TORCH_FREE_COMMANDS


def _ensure_runtime_dependencies(args) -> None:
    if not _requires_torch(args):
        return
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This ssamem command requires `torch` and `transformers` in the active Python environment. "
            "Install dependencies from `ssamem/requirements.txt` first."
        ) from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ensure_runtime_dependencies(args)
    configure_runtime(args)
    if args.output_dir:
        RunConfig(output_dir=args.output_dir).ensure_output_dir()
    COMMAND_HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
