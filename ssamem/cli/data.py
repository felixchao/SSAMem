from __future__ import annotations

import argparse

from ssamem.cli.common import add_hf_runtime_args, add_shared_run_args


def register_data_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
