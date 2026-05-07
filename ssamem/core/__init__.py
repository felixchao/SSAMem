from __future__ import annotations

"""Core runtime modules for the SSAMem architecture."""

from ssamem.core.mas import MASPromptTemplate, MASRoleSpec, MASTopologySpec
from ssamem.core.kernelspace import IPCBus, MemoryAgent, OSKernel
from ssamem.core.memory_actions import parse_memory_request
from ssamem.core.pipeline import PointerDrivenSSAMemPipeline
from ssamem.core.userspace import SoftPromptBatch, UserSpaceGenerationOutput, UserSpaceMAS

__all__ = [
    "IPCBus",
    "MASPromptTemplate",
    "MASRoleSpec",
    "MASTopologySpec",
    "MemoryAgent",
    "OSKernel",
    "PointerDrivenSSAMemPipeline",
    "SoftPromptBatch",
    "UserSpaceGenerationOutput",
    "UserSpaceMAS",
    "parse_memory_request",
]
