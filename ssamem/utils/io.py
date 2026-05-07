from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_output_path(args) -> str | None:
    if getattr(args, "output_path", None):
        return args.output_path
    if not args.output_dir:
        return None
    run_name = args.run_name or str(args.command)
    return str(Path(args.output_dir) / f"{run_name}.json")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
