from __future__ import annotations

from typing import Any


def summarize_metric_history(history: list[dict[str, float]], *, tail_count: int = 50) -> dict[str, Any]:
    if not history:
        return {"steps": 0, "metrics": {}}
    metric_names = sorted({name for row in history for name in row})
    metrics = {}
    for name in metric_names:
        values = [float(row[name]) for row in history if name in row]
        tail = values[-tail_count:]
        metrics[name] = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean_tail": sum(tail) / len(tail),
        }
    return {"steps": len(history), "tail_count": tail_count, "metrics": metrics}
