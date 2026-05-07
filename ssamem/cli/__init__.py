from __future__ import annotations

"""Command-line parser construction for `python -m ssamem`."""

import argparse

from ssamem.cli.core import register_core_commands
from ssamem.cli.data import register_data_commands
from ssamem.cli.demo import register_demo_commands
from ssamem.cli.evaluation import register_evaluation_commands
from ssamem.cli.training import register_training_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Pointer-Driven SSAMem")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_core_commands(subparsers)
    register_demo_commands(subparsers)
    register_data_commands(subparsers)
    register_training_commands(subparsers)
    register_evaluation_commands(subparsers)
    return parser
