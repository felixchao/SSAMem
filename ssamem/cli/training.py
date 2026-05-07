from __future__ import annotations

import argparse

from ssamem.cli.common import add_config_manifest_args, add_hf_runtime_args, add_runtime_args, add_shared_run_args


def register_training_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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

    retrieval_train_parser = subparsers.add_parser(
        "train-retrieval",
        help="Train QueryLatentEncoder and MemoryKeyEncoder with InfoNCE retrieval alignment.",
    )
    add_config_manifest_args(retrieval_train_parser)
    add_hf_runtime_args(retrieval_train_parser, default_device="cuda")
    retrieval_train_parser.add_argument("--valid-manifest", default=None)
    retrieval_train_parser.add_argument("--checkpoint", default=None, help="Optional retrieval checkpoint to initialize from.")
    retrieval_train_parser.add_argument("--output-path", default=None)
    retrieval_train_parser.add_argument("--history-path", default=None)
    retrieval_train_parser.add_argument("--retrieval-key-dim", type=int, default=0)
    retrieval_train_parser.add_argument("--query-max-tokens", type=int, default=96)
    retrieval_train_parser.add_argument("--query-field", default="task_prompt", choices=["task_prompt", "student_text", "explicit_text"])
    retrieval_train_parser.add_argument("--temperature", type=float, default=0.07)
    retrieval_train_parser.add_argument("--lr", type=float, default=1e-4)
    retrieval_train_parser.add_argument("--epochs", type=int, default=1)
    retrieval_train_parser.add_argument("--batch-size", type=int, default=8)
    retrieval_train_parser.add_argument("--eval-batch-size", type=int, default=16)
    retrieval_train_parser.add_argument("--log-every", type=int, default=10)
    retrieval_train_parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    add_shared_run_args(retrieval_train_parser)

    dpo_parser = subparsers.add_parser("train-pointer-dpo", help="Train pointer router with TRL DPOTrainer.")
    dpo_parser.add_argument("--config", default=None)
    dpo_parser.add_argument("--preferences", default=None)
    add_runtime_args(dpo_parser, default_device="cuda")
    dpo_parser.set_defaults(runtime_mode="hf")
    add_shared_run_args(dpo_parser)

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
