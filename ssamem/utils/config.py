from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise SystemExit("YAML configs require `pyyaml`; use JSON or install pyyaml.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return dict(payload or {})


def deep_get(mapping: dict[str, Any], path: str, default=None):
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
