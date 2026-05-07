from __future__ import annotations

"""Shared CLI constants and argument helpers for `python -m ssamem`."""

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
