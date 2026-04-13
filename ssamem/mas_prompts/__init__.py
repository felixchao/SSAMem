from __future__ import annotations

from importlib import import_module

_DOMAIN_MODULES = {
    "alfworld": "ssamem.mas_prompts.alfworld",
    "kodcode": "ssamem.mas_prompts.kodcode",
    "pddl": "ssamem.mas_prompts.pddl",
    "popqa": "ssamem.mas_prompts.popqa",
    "triviaqa": "ssamem.mas_prompts.triviaqa",
}


def get_domain_prompts(task_domain: str) -> dict[str, dict[str, str]]:
    normalized = task_domain.strip().lower()
    if normalized not in _DOMAIN_MODULES:
        raise ValueError(f"Unsupported MAS task domain: {task_domain}")
    module = import_module(_DOMAIN_MODULES[normalized])
    return getattr(module, "PROMPTS")

