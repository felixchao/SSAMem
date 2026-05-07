from __future__ import annotations

"""Register CLI commands required to run the main SSAMem architecture."""

import argparse

from ssamem.cli.common import (
    add_config_manifest_args,
    add_experience_clustering_args,
    add_runtime_args,
    add_shared_run_args,
)


def register_core_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    experience_bank_parser = subparsers.add_parser(
        "build-experience-bank",
        help="Preload projected trajectory latents into the MemoryAgent experience bank.",
    )
    add_config_manifest_args(experience_bank_parser)
    add_runtime_args(experience_bank_parser)
    experience_bank_parser.add_argument("--checkpoint", required=True)
    experience_bank_parser.add_argument(
        "--retrieval-checkpoint",
        default=None,
        help="Optional trained query/memory retrieval-key checkpoint for pointer-table SEARCH.",
    )
    experience_bank_parser.add_argument("--limit", type=int, default=100)
    experience_bank_parser.add_argument("--output-path", default=None)
    experience_bank_parser.add_argument("--pointer-table-path", default=None)
    add_experience_clustering_args(experience_bank_parser)
    add_shared_run_args(experience_bank_parser)

    query_memory_parser = subparsers.add_parser(
        "query-memory-agent",
        help="Preload an experience bank, then run explicit MemoryAgent SEARCH or GET.",
    )
    add_config_manifest_args(query_memory_parser)
    add_runtime_args(query_memory_parser)
    query_memory_parser.add_argument("--checkpoint", required=True)
    query_memory_parser.add_argument(
        "--retrieval-checkpoint",
        default=None,
        help="Optional trained query/memory retrieval-key checkpoint for pointer-table SEARCH.",
    )
    query_memory_parser.add_argument("--query", default=None, help="SEARCH query text.")
    query_memory_parser.add_argument("--address", default=None, help="Exact GET address, e.g. <PTR_0x001>:0000.")
    query_memory_parser.add_argument("--limit", type=int, default=100, help="Number of memories to preload.")
    query_memory_parser.add_argument("--top-k", type=int, default=3)
    query_memory_parser.add_argument("--output-path", default=None)
    add_experience_clustering_args(query_memory_parser)
    add_shared_run_args(query_memory_parser)
