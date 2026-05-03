from __future__ import annotations

import json

from ssamem.commands.common import build_pipeline_from_args


def run_demo(args) -> None:
    from ssamem.workflows.demo import run_demo as _run_demo

    _run_demo(args, build_pipeline_from_args=build_pipeline_from_args)

def run_demo_mas(args) -> None:
    from ssamem.workflows.demo import run_demo_mas as _run_demo_mas

    _run_demo_mas(args, build_pipeline_from_args=build_pipeline_from_args)

def run_collect_text_mas_trajectories(args) -> None:
    from ssamem.workflows.trajectory import collect_text_mas_trajectories

    payload = collect_text_mas_trajectories(args, build_pipeline_from_args=build_pipeline_from_args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
