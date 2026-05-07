from __future__ import annotations

"""Core runtime command registry for the main SSAMem architecture."""

from ssamem.commands.mas import run_build_experience_bank, run_query_memory_agent


CORE_COMMAND_HANDLERS = {
    "build-experience-bank": run_build_experience_bank,
    "query-memory-agent": run_query_memory_agent,
}
