from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "AgentMessage",
    "CODiBatch",
    "CODiDistiller",
    "ConsolidationResult",
    "DiskLatentStore",
    "DiskPageTableStore",
    "ExplicitTeacher",
    "HashMIPSRetriever",
    "HashEmbeddingEncoder",
    "IPCBus",
    "KernelConfig",
    "MASConfig",
    "MASExecutionTrace",
    "MASPromptTemplate",
    "MASRoleSpec",
    "MASTopologySpec",
    "MASTurn",
    "LatentStudent",
    "LatentTensor",
    "MemoryAgent",
    "OSKernel",
    "PageTable",
    "PipelineConfig",
    "PipelineRunResult",
    "PointerDrivenSSAMemPipeline",
    "PersistentMemoryBackend",
    "PointerSearchHit",
    "ResolvedIPCMessage",
    "RunConfig",
    "BaseRetriever",
    "RuntimeConfig",
    "SoftPromptBatch",
    "UserSpaceGenerationOutput",
    "UserSpaceMAS",
]

_LAZY_IMPORTS = {
    "RuntimeConfig": ("ssamem.config", "RuntimeConfig"),
    "KernelConfig": ("ssamem.config", "KernelConfig"),
    "MASConfig": ("ssamem.config", "MASConfig"),
    "RunConfig": ("ssamem.config", "RunConfig"),
    "PipelineConfig": ("ssamem.config", "PipelineConfig"),
    "AgentMessage": ("ssamem.data_models", "AgentMessage"),
    "ConsolidationResult": ("ssamem.data_models", "ConsolidationResult"),
    "DiskLatentStore": ("ssamem.storage", "DiskLatentStore"),
    "DiskPageTableStore": ("ssamem.storage", "DiskPageTableStore"),
    "LatentTensor": ("ssamem.data_models", "LatentTensor"),
    "MASExecutionTrace": ("ssamem.data_models", "MASExecutionTrace"),
    "MASTurn": ("ssamem.data_models", "MASTurn"),
    "PageTable": ("ssamem.data_models", "PageTable"),
    "PipelineRunResult": ("ssamem.data_models", "PipelineRunResult"),
    "PointerSearchHit": ("ssamem.data_models", "PointerSearchHit"),
    "ResolvedIPCMessage": ("ssamem.data_models", "ResolvedIPCMessage"),
    "BaseRetriever": ("ssamem.retrieval", "BaseRetriever"),
    "HashEmbeddingEncoder": ("ssamem.retrieval", "HashEmbeddingEncoder"),
    "HashMIPSRetriever": ("ssamem.retrieval", "HashMIPSRetriever"),
    "IPCBus": ("ssamem.kernelspace", "IPCBus"),
    "MemoryAgent": ("ssamem.kernelspace", "MemoryAgent"),
    "MASPromptTemplate": ("ssamem.mas", "MASPromptTemplate"),
    "MASRoleSpec": ("ssamem.mas", "MASRoleSpec"),
    "MASTopologySpec": ("ssamem.mas", "MASTopologySpec"),
    "OSKernel": ("ssamem.kernelspace", "OSKernel"),
    "PointerDrivenSSAMemPipeline": ("ssamem.pipeline", "PointerDrivenSSAMemPipeline"),
    "PersistentMemoryBackend": ("ssamem.storage", "PersistentMemoryBackend"),
    "CODiBatch": ("ssamem.trainingspace", "CODiBatch"),
    "CODiDistiller": ("ssamem.trainingspace", "CODiDistiller"),
    "ExplicitTeacher": ("ssamem.trainingspace", "ExplicitTeacher"),
    "LatentStudent": ("ssamem.trainingspace", "LatentStudent"),
    "SoftPromptBatch": ("ssamem.userspace", "SoftPromptBatch"),
    "UserSpaceMAS": ("ssamem.userspace", "UserSpaceMAS"),
    "UserSpaceGenerationOutput": ("ssamem.userspace", "UserSpaceGenerationOutput"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'ssamem' has no attribute '{name}'")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
