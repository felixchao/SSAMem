from __future__ import annotations

import argparse

from ssamem.cli.common import MAS_CHOICES, add_hf_runtime_args, add_runtime_args, add_shared_run_args


def register_demo_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
