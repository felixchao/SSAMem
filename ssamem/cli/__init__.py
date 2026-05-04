from __future__ import annotations

"""Command-line parser construction for `python -m ssamem`."""

import argparse


RUNTIME_CHOICES = ["tiny-random", "hf"]
MAS_CHOICES = ["camel", "autogen", "debate", "macnet"]
LOG_LEVEL_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR"]


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVEL_CHOICES)


def add_runtime_args(parser: argparse.ArgumentParser, *, default_device: str = "cpu") -> None:
    parser.add_argument("--runtime-mode", default="tiny-random", choices=RUNTIME_CHOICES)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default=default_device)


def add_hf_runtime_args(parser: argparse.ArgumentParser, *, default_device: str = "cpu") -> None:
    add_runtime_args(parser, default_device=default_device)
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--load-in-4bit", action="store_true")


def add_config_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)


def add_mas_eval_args(parser: argparse.ArgumentParser, *, default_tokens: int = 32) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=default_tokens)
    parser.add_argument("--mas-style", default="camel", choices=MAS_CHOICES)
    parser.add_argument("--task-domain", default="popqa")
    parser.add_argument("--output-path", default=None)


def add_experience_clustering_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cluster-method",
        default="online",
        choices=["online", "offline-kmeans"],
        help="How to group projected experience memories into pointer clusters.",
    )
    parser.add_argument("--kmeans-clusters", type=int, default=50)
    parser.add_argument("--kmeans-max-iter", type=int, default=50)


def add_checkpoint_eval_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--checkpoint", required=required, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-path", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Pointer-Driven SSAMem")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run an end-to-end standalone demo.")
    add_runtime_args(demo_parser)
    demo_parser.add_argument("--top-k-prefetch", type=int, default=1)
    demo_parser.add_argument("--max-new-tokens", type=int, default=24)
    demo_parser.add_argument("--storage-root", default=None)
    add_shared_run_args(demo_parser)

    mas_parser = subparsers.add_parser("demo-mas", help="Run a MAS-style standalone demo.")
    add_runtime_args(mas_parser)
    mas_parser.add_argument("--top-k-prefetch", type=int, default=1)
    mas_parser.add_argument("--max-new-tokens", type=int, default=24)
    mas_parser.add_argument("--storage-root", default=None)
    mas_parser.add_argument("--mas-style", default="camel", choices=MAS_CHOICES)
    mas_parser.add_argument("--task-domain", default=None)
    add_shared_run_args(mas_parser)

    collect_traj_parser = subparsers.add_parser("collect-text-mas-trajectories", help="Run pure-text MAS and save trajectory JSONL.")
    collect_traj_parser.add_argument("--input", required=True, help="Input JSONL with task/context/target fields.")
    collect_traj_parser.add_argument("--output", required=True)
    add_hf_runtime_args(collect_traj_parser)
    collect_traj_parser.add_argument("--limit", type=int, default=100)
    collect_traj_parser.add_argument("--mas-style", default="camel", choices=MAS_CHOICES)
    collect_traj_parser.add_argument("--task-domain", default="popqa")
    collect_traj_parser.add_argument("--max-new-tokens", type=int, default=48)
    add_shared_run_args(collect_traj_parser)

    train_parser = subparsers.add_parser("train-ssa", help="Run a minimal SSA alignment demo.")
    add_config_manifest_args(train_parser)
    train_parser.add_argument("--output-path", default=None)
    train_parser.add_argument("--checkpoint", default=None, help="Optional SSA checkpoint to initialize projector/composer.")
    add_runtime_args(train_parser)
    train_parser.add_argument("--steps", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--beta", type=float, default=0.1)
    train_parser.add_argument("--answer-ce-weight", type=float, default=0.0)
    train_parser.add_argument("--max-pairs", type=int, default=None)
    train_parser.add_argument("--grad-clip-norm", type=float, default=None)
    train_parser.add_argument("--storage-root", default=None)
    train_parser.add_argument("--history-path", default=None)
    add_shared_run_args(train_parser)

    build_ssa_parser = subparsers.add_parser("build-ssa-data", help="Build SSA latent tensor shards and manifest.")
    build_ssa_parser.add_argument("--input", required=True)
    build_ssa_parser.add_argument("--output", required=True)
    add_hf_runtime_args(build_ssa_parser)
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
    add_runtime_args(dpo_parser, default_device="cuda")
    dpo_parser.set_defaults(runtime_mode="hf")
    add_shared_run_args(dpo_parser)

    training_eval_parser = subparsers.add_parser("eval-training", help="Evaluate SSA latent distance and DPO data safety.")
    add_config_manifest_args(training_eval_parser)
    training_eval_parser.add_argument("--preferences", default=None)
    add_runtime_args(training_eval_parser)
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

    multimemory_ssa_parser = subparsers.add_parser(
        "make-multimemory-ssa-manifest",
        help="Create an SSA manifest where each sample mounts one positive latent plus distractor latents.",
    )
    multimemory_ssa_parser.add_argument("--manifest", required=True)
    multimemory_ssa_parser.add_argument("--output", required=True)
    multimemory_ssa_parser.add_argument("--distractors", type=int, default=2)
    multimemory_ssa_parser.add_argument("--allow-same-target", action="store_true")
    multimemory_ssa_parser.add_argument("--shuffle-memories", action="store_true")
    add_shared_run_args(multimemory_ssa_parser)

    ssa_report_parser = subparsers.add_parser("eval-ssa-report", help="Compare SSA latent-distance metrics before and after training.")
    add_config_manifest_args(ssa_report_parser)
    add_runtime_args(ssa_report_parser)
    ssa_report_parser.add_argument("--checkpoint", default=None)
    ssa_report_parser.add_argument("--limit", type=int, default=100)
    ssa_report_parser.add_argument("--output-path", default=None)
    add_shared_run_args(ssa_report_parser)

    ssa_probe_parser = subparsers.add_parser("probe-ssa-generation", help="Probe no-memory/text-memory/latent-memory generation quality.")
    add_config_manifest_args(ssa_probe_parser)
    add_runtime_args(ssa_probe_parser)
    ssa_probe_parser.add_argument("--checkpoint", default=None)
    ssa_probe_parser.add_argument("--limit", type=int, default=10)
    ssa_probe_parser.add_argument("--max-new-tokens", type=int, default=48)
    ssa_probe_parser.add_argument("--max-keywords", type=int, default=8)
    ssa_probe_parser.add_argument("--output-path", default=None)
    add_shared_run_args(ssa_probe_parser)

    mas_latent_eval_parser = subparsers.add_parser(
        "eval-mas-latent-memory",
        help="Evaluate no-memory/text-memory/latent-memory in the MAS pipeline with a trained composer/projector.",
    )
    add_config_manifest_args(mas_latent_eval_parser)
    add_runtime_args(mas_latent_eval_parser)
    mas_latent_eval_parser.add_argument("--checkpoint", required=True)
    mas_latent_eval_parser.add_argument("--limit", type=int, default=20)
    add_mas_eval_args(mas_latent_eval_parser, default_tokens=64)
    add_shared_run_args(mas_latent_eval_parser)

    experience_bank_parser = subparsers.add_parser(
        "build-experience-bank",
        help="Preload projected trajectory latents into the MemoryAgent experience bank.",
    )
    add_config_manifest_args(experience_bank_parser)
    add_runtime_args(experience_bank_parser)
    experience_bank_parser.add_argument("--checkpoint", required=True)
    experience_bank_parser.add_argument("--limit", type=int, default=100)
    experience_bank_parser.add_argument("--output-path", default=None)
    experience_bank_parser.add_argument("--pointer-table-path", default=None)
    add_experience_clustering_args(experience_bank_parser)
    add_shared_run_args(experience_bank_parser)

    mas_search_eval_parser = subparsers.add_parser(
        "eval-mas-memory-search",
        help="Evaluate MAS using MemoryAgent SEARCH over a preloaded experience bank.",
    )
    mas_search_eval_parser.add_argument("--config", default=None)
    mas_search_eval_parser.add_argument("--bank-manifest", required=True)
    mas_search_eval_parser.add_argument("--eval-manifest", required=True)
    add_runtime_args(mas_search_eval_parser)
    mas_search_eval_parser.add_argument("--checkpoint", required=True)
    mas_search_eval_parser.add_argument("--limit", type=int, default=20)
    mas_search_eval_parser.add_argument("--bank-limit", type=int, default=200)
    mas_search_eval_parser.add_argument("--top-k-prefetch", type=int, default=1)
    add_experience_clustering_args(mas_search_eval_parser)
    add_mas_eval_args(mas_search_eval_parser, default_tokens=32)
    add_shared_run_args(mas_search_eval_parser)

    mas_loop_eval_parser = subparsers.add_parser(
        "eval-mas-memory-agent-loop",
        help="Evaluate MAS agents that explicitly request MemoryAgent SEARCH/GET before answering.",
    )
    mas_loop_eval_parser.add_argument("--config", default=None)
    mas_loop_eval_parser.add_argument("--bank-manifest", required=True)
    mas_loop_eval_parser.add_argument("--eval-manifest", required=True)
    add_runtime_args(mas_loop_eval_parser)
    mas_loop_eval_parser.add_argument("--checkpoint", required=True)
    mas_loop_eval_parser.add_argument("--limit", type=int, default=20)
    mas_loop_eval_parser.add_argument("--bank-limit", type=int, default=200)
    mas_loop_eval_parser.add_argument("--top-k", type=int, default=1)
    add_experience_clustering_args(mas_loop_eval_parser)
    mas_loop_eval_parser.add_argument(
        "--memory-request-policy",
        default="require-search",
        choices=["auto", "require-search"],
        help="Use auto to let agents choose, or require-search to force each role to query MemoryAgent.",
    )
    add_mas_eval_args(mas_loop_eval_parser, default_tokens=32)
    add_shared_run_args(mas_loop_eval_parser)

    query_memory_parser = subparsers.add_parser(
        "query-memory-agent",
        help="Preload an experience bank, then run explicit MemoryAgent SEARCH or GET.",
    )
    add_config_manifest_args(query_memory_parser)
    add_runtime_args(query_memory_parser)
    query_memory_parser.add_argument("--checkpoint", required=True)
    query_memory_parser.add_argument("--query", default=None, help="SEARCH query text.")
    query_memory_parser.add_argument("--address", default=None, help="Exact GET address, e.g. <PTR_0x001>:0000.")
    query_memory_parser.add_argument("--limit", type=int, default=100, help="Number of memories to preload.")
    query_memory_parser.add_argument("--top-k", type=int, default=3)
    query_memory_parser.add_argument("--output-path", default=None)
    add_experience_clustering_args(query_memory_parser)
    add_shared_run_args(query_memory_parser)

    lmpo_rollout_parser = subparsers.add_parser(
        "collect-lmpo-rollouts",
        help="Collect sampled single-latent responses with memory-utility rewards.",
    )
    add_config_manifest_args(lmpo_rollout_parser)
    add_runtime_args(lmpo_rollout_parser)
    lmpo_rollout_parser.add_argument("--checkpoint", default=None)
    lmpo_rollout_parser.add_argument("--limit", type=int, default=100)
    lmpo_rollout_parser.add_argument("--rollouts-per-sample", type=int, default=4)
    lmpo_rollout_parser.add_argument("--max-new-tokens", type=int, default=64)
    lmpo_rollout_parser.add_argument("--temperature", type=float, default=0.7)
    lmpo_rollout_parser.add_argument("--top-p", type=float, default=0.9)
    lmpo_rollout_parser.add_argument("--output-path", required=True)
    add_shared_run_args(lmpo_rollout_parser)

    lmpo_pair_parser = subparsers.add_parser(
        "build-lmpo-pairs",
        help="Build chosen/rejected LMPO preference pairs from rollout rewards.",
    )
    lmpo_pair_parser.add_argument("--rollouts", required=True)
    lmpo_pair_parser.add_argument("--output-path", required=True)
    lmpo_pair_parser.add_argument("--margin", type=float, default=0.5)
    lmpo_pair_parser.add_argument("--max-pairs-per-sample", type=int, default=1)
    lmpo_pair_parser.add_argument("--require-chosen-target-hit", action="store_true")
    lmpo_pair_parser.add_argument("--allow-invalid-chosen", action="store_true")
    add_shared_run_args(lmpo_pair_parser)

    lmpo_train_parser = subparsers.add_parser(
        "train-lmpo-projector",
        help="LMPO-style preference tuning for the single-latent composer/projector.",
    )
    lmpo_train_parser.add_argument("--config", default=None)
    lmpo_train_parser.add_argument("--pairs", required=True)
    lmpo_train_parser.add_argument("--output-path", required=True)
    lmpo_train_parser.add_argument("--checkpoint", required=True)
    add_runtime_args(lmpo_train_parser)
    lmpo_train_parser.add_argument("--lr", type=float, default=1e-5)
    lmpo_train_parser.add_argument("--epochs", type=int, default=1)
    lmpo_train_parser.add_argument("--beta", type=float, default=0.1)
    lmpo_train_parser.add_argument("--answer-ce-weight", type=float, default=0.05)
    lmpo_train_parser.add_argument("--hidden-anchor-weight", type=float, default=0.0)
    lmpo_train_parser.add_argument("--max-pairs", type=int, default=None)
    lmpo_train_parser.add_argument("--log-every", type=int, default=10)
    lmpo_train_parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    add_shared_run_args(lmpo_train_parser)

    pointer_eval_parser = subparsers.add_parser("eval-pointer-routing", help="Evaluate pointer routing on preference data.")
    pointer_eval_parser.add_argument("--config", default=None)
    pointer_eval_parser.add_argument("--preferences", default=None)
    add_runtime_args(pointer_eval_parser)
    pointer_eval_parser.set_defaults(runtime_mode="hf")
    pointer_eval_parser.add_argument("--limit", type=int, default=100)
    pointer_eval_parser.add_argument("--output-path", default=None)
    add_shared_run_args(pointer_eval_parser)

    suite_parser = subparsers.add_parser("run-ssa-suite", help="Split, train, eval, and probe SSA in one run.")
    add_config_manifest_args(suite_parser)
    suite_parser.add_argument("--suite-dir", required=True)
    add_runtime_args(suite_parser)
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
    add_runtime_args(eval_parser)
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
