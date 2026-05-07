from __future__ import annotations

import argparse

from ssamem.cli.common import (
    add_config_manifest_args,
    add_experience_clustering_args,
    add_hf_runtime_args,
    add_mas_eval_args,
    add_runtime_args,
    add_shared_run_args,
)


def register_evaluation_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    retrieval_eval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate retrieval-key alignment with Recall@K and MRR.",
    )
    add_config_manifest_args(retrieval_eval_parser)
    add_hf_runtime_args(retrieval_eval_parser, default_device="cuda")
    retrieval_eval_parser.add_argument("--candidate-manifest", default=None)
    retrieval_eval_parser.add_argument("--checkpoint", required=True)
    retrieval_eval_parser.add_argument("--output-path", default=None)
    retrieval_eval_parser.add_argument("--retrieval-key-dim", type=int, default=0)
    retrieval_eval_parser.add_argument("--query-max-tokens", type=int, default=96)
    retrieval_eval_parser.add_argument("--query-field", default="task_prompt", choices=["task_prompt", "student_text", "explicit_text"])
    retrieval_eval_parser.add_argument("--temperature", type=float, default=0.07)
    retrieval_eval_parser.add_argument("--batch-size", type=int, default=16)
    add_shared_run_args(retrieval_eval_parser)

    training_eval_parser = subparsers.add_parser("eval-training", help="Evaluate SSA latent distance and DPO data safety.")
    add_config_manifest_args(training_eval_parser)
    training_eval_parser.add_argument("--preferences", default=None)
    add_runtime_args(training_eval_parser)
    training_eval_parser.add_argument("--checkpoint", default=None)
    training_eval_parser.add_argument("--limit", type=int, default=20)
    add_shared_run_args(training_eval_parser)

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

    mas_search_eval_parser = subparsers.add_parser(
        "eval-mas-memory-search",
        help="Evaluate MAS using MemoryAgent SEARCH over a preloaded experience bank.",
    )
    mas_search_eval_parser.add_argument("--config", default=None)
    mas_search_eval_parser.add_argument("--bank-manifest", required=True)
    mas_search_eval_parser.add_argument("--eval-manifest", required=True)
    add_runtime_args(mas_search_eval_parser)
    mas_search_eval_parser.add_argument("--checkpoint", required=True)
    mas_search_eval_parser.add_argument("--retrieval-checkpoint", default=None)
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
    mas_loop_eval_parser.add_argument("--retrieval-checkpoint", default=None)
    mas_loop_eval_parser.add_argument("--limit", type=int, default=20)
    mas_loop_eval_parser.add_argument("--bank-limit", type=int, default=200)
    mas_loop_eval_parser.add_argument("--top-k", type=int, default=1)
    add_experience_clustering_args(mas_loop_eval_parser)
    mas_loop_eval_parser.add_argument(
        "--memory-request-policy",
        default="require-search",
        choices=["auto", "require-search", "search-only"],
        help="Use auto to let agents choose, or require-search/search-only to force SEARCH-only MemoryAgent access.",
    )
    add_mas_eval_args(mas_loop_eval_parser, default_tokens=32)
    add_shared_run_args(mas_loop_eval_parser)

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
