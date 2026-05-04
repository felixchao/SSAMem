from __future__ import annotations

"""Parsing and formatting helpers for explicit MemoryAgent actions."""

import re

from ssamem.data_models import MemoryRequest, PointerSearchHit

_ACTION_PATTERN = re.compile(r"^\s*(SEARCH|GET|NONE)\b\s*:?\s*(.*)", re.IGNORECASE)
_TOP_K_PATTERN = re.compile(r"\btop[_ -]?k\s*=\s*(\d+)", re.IGNORECASE)


def parse_memory_request(
    text: str,
    *,
    default_query: str = "",
    default_top_k: int = 1,
    policy: str = "auto",
) -> MemoryRequest:
    raw = (text or "").strip()
    normalized_policy = policy.strip().lower()
    match = None
    for line in raw.splitlines():
        match = _ACTION_PATTERN.match(line.strip())
        if match is not None:
            break
    if normalized_policy == "require-search" and match is None:
        return MemoryRequest(mode="SEARCH", query=default_query, top_k=default_top_k, raw_text=raw)
    if match is None and raw.lower().startswith("no memory"):
        return MemoryRequest(mode="NONE", query=default_query, top_k=default_top_k, raw_text=raw)
    if match is None:
        return MemoryRequest(mode="NONE", query=default_query, top_k=default_top_k, raw_text=raw)

    mode = match.group(1).upper()
    payload = match.group(2).strip()
    top_k = default_top_k
    top_k_match = _TOP_K_PATTERN.search(payload)
    if top_k_match:
        top_k = max(int(top_k_match.group(1)), 1)
        payload = _TOP_K_PATTERN.sub("", payload).strip(" ;,\n")

    if mode == "GET":
        address_match = re.search(r"<PTR_0x[0-9A-Fa-f]+>(?::\d+)?", payload)
        return MemoryRequest(
            mode="GET",
            address=address_match.group(0) if address_match else payload or None,
            top_k=1,
            raw_text=raw,
        )
    if mode == "SEARCH":
        return MemoryRequest(mode="SEARCH", query=payload or default_query, top_k=top_k, raw_text=raw)
    return MemoryRequest(mode="NONE", query=default_query, top_k=default_top_k, raw_text=raw)


def build_memory_request_prompt(
    role_prompt: str,
    pointer_table_context: str,
    *,
    task_query: str = "",
    default_top_k: int = 1,
    policy: str = "auto",
) -> str:
    if policy == "require-search":
        task_block = f"[Current Task]\n{task_query or role_prompt}\n\n"
        instruction = (
            "You must request memory before answering. Return exactly one line:\n"
            f"SEARCH: <brief query about the current task>; top_k={default_top_k}"
        )
        return (
            f"{task_block}"
            f"{pointer_table_context}\n\n"
            "[Memory Access Decision]\n"
            f"{instruction}\n"
        ).strip()
    else:
        instruction = (
            "Before answering, decide whether you need latent memory from the MemoryAgent.\n"
            "Return exactly one line in one of these formats:\n"
            f"SEARCH: <brief query>; top_k={default_top_k}\n"
            "GET: <PTR_0x001>:0000\n"
            "NONE:"
        )
    return (
        f"{role_prompt}\n\n"
        f"{pointer_table_context}\n\n"
        "[Memory Access Decision]\n"
        f"{instruction}\n"
    ).strip()


def format_memory_observation(hits: list[PointerSearchHit]) -> str:
    if not hits:
        return "[MemoryAgent Observation]\nNo memory was retrieved."
    lines = ["[MemoryAgent Observation]"]
    for rank, hit in enumerate(hits, start=1):
        summary = hit.latent.metadata.get("summary") or hit.latent.metadata.get("topic") or ""
        target = hit.latent.metadata.get("target_text")
        lines.append(
            f"{rank}. address={hit.exact_address} score={hit.score:.4g} "
            f"summary={summary!s} target={target!s}"
        )
    return "\n".join(lines)


def build_answer_prompt(role_prompt: str, memory_observation: str) -> str:
    return (
        f"{role_prompt}\n\n"
        f"{memory_observation}\n\n"
        "[Output Requirement]\n"
        "Now answer the original role task using the mounted latent memory when useful."
    ).strip()
