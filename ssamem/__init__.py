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
    "PointerDrivenLatentOSPipeline",
    "PersistentMemoryBackend",
    "PointerSearchHit",
    "ResolvedIPCMessage",
    "RuntimeConfig",
    "SoftPromptBatch",
    "UserSpaceGenerationOutput",
    "UserSpaceMAS",
]

_LAZY_IMPORTS = {
    "RuntimeConfig": ("latent_os.config", "RuntimeConfig"),
    "KernelConfig": ("latent_os.config", "KernelConfig"),
    "MASConfig": ("latent_os.config", "MASConfig"),
    "PipelineConfig": ("latent_os.config", "PipelineConfig"),
    "AgentMessage": ("latent_os.data_models", "AgentMessage"),
    "ConsolidationResult": ("latent_os.data_models", "ConsolidationResult"),
    "DiskLatentStore": ("latent_os.storage", "DiskLatentStore"),
    "DiskPageTableStore": ("latent_os.storage", "DiskPageTableStore"),
    "LatentTensor": ("latent_os.data_models", "LatentTensor"),
    "MASExecutionTrace": ("latent_os.data_models", "MASExecutionTrace"),
    "MASTurn": ("latent_os.data_models", "MASTurn"),
    "PageTable": ("latent_os.data_models", "PageTable"),
    "PipelineRunResult": ("latent_os.data_models", "PipelineRunResult"),
    "PointerSearchHit": ("latent_os.data_models", "PointerSearchHit"),
    "ResolvedIPCMessage": ("latent_os.data_models", "ResolvedIPCMessage"),
    "HashEmbeddingEncoder": ("latent_os.kernelspace", "HashEmbeddingEncoder"),
    "IPCBus": ("latent_os.kernelspace", "IPCBus"),
    "MemoryAgent": ("latent_os.kernelspace", "MemoryAgent"),
    "MASPromptTemplate": ("latent_os.mas", "MASPromptTemplate"),
    "MASRoleSpec": ("latent_os.mas", "MASRoleSpec"),
    "MASTopologySpec": ("latent_os.mas", "MASTopologySpec"),
    "OSKernel": ("latent_os.kernelspace", "OSKernel"),
    "PointerDrivenLatentOSPipeline": ("latent_os.pipeline", "PointerDrivenLatentOSPipeline"),
    "PersistentMemoryBackend": ("latent_os.storage", "PersistentMemoryBackend"),
    "CODiBatch": ("latent_os.trainingspace", "CODiBatch"),
    "CODiDistiller": ("latent_os.trainingspace", "CODiDistiller"),
    "ExplicitTeacher": ("latent_os.trainingspace", "ExplicitTeacher"),
    "LatentStudent": ("latent_os.trainingspace", "LatentStudent"),
    "SoftPromptBatch": ("latent_os.userspace", "SoftPromptBatch"),
    "UserSpaceMAS": ("latent_os.userspace", "UserSpaceMAS"),
    "UserSpaceGenerationOutput": ("latent_os.userspace", "UserSpaceGenerationOutput"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'latent_os' has no attribute '{name}'")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
