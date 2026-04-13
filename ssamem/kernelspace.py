from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ssamem.data_models import (
    AgentMessage,
    ConsolidationResult,
    LatentTensor,
    PageTable,
    PointerSearchHit,
    ResolvedIPCMessage,
)
from ssamem.retrieval import BaseRetriever
from ssamem.storage import PersistentMemoryBackend


class MemoryAgent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: Optional[int] = None,
        *,
        page_table: Optional[PageTable] = None,
        reward_threshold: float = 0.0,
        latent_window: int = 8,
        persistence: Optional[PersistentMemoryBackend] = None,
        autosave: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim or hidden_size
        self.page_table = page_table or PageTable()
        self.reward_threshold = reward_threshold
        self.latent_window = latent_window
        self.memory_store: dict[str, LatentTensor] = {}
        self.persistence = persistence
        self.autosave = autosave
        self.key_projector = (
            nn.Identity() if self.hidden_size == self.key_dim else nn.Linear(self.hidden_size, self.key_dim)
        )

    def build_key_vector(self, tensor_data: torch.Tensor) -> torch.Tensor:
        pooled = tensor_data.mean(dim=0)
        key_vector = self.key_projector(pooled)
        return key_vector.detach().float()

    def register_latent(self, latent: LatentTensor) -> str:
        pointer = self.page_table.allocate(latent.storage_id)
        latent.pointer = pointer
        self.memory_store[latent.storage_id] = latent
        self._autosave_latent(latent)
        return pointer

    def _autosave_latent(self, latent: LatentTensor) -> None:
        if self.persistence is None or not self.autosave:
            return
        self.persistence.save_page_table(self.page_table)
        self.persistence.save_latent(latent)

    def add_tensor(
        self,
        tensor_data: torch.Tensor,
        *,
        utility_score: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> str:
        if tensor_data.dim() == 3:
            if tensor_data.size(0) != 1:
                raise ValueError("`tensor_data` must be [K, D] or [1, K, D].")
            tensor_data = tensor_data.squeeze(0)
        key_vector = self.build_key_vector(tensor_data)
        latent = LatentTensor(
            tensor_data=tensor_data.detach(),
            key_vector=key_vector,
            utility_score=utility_score,
            metadata=dict(metadata or {}),
        )
        return self.register_latent(latent)

    def save_to_disk(self) -> None:
        if self.persistence is None:
            raise ValueError("Persistence backend is not configured for this MemoryAgent.")
        self.persistence.save(self.page_table, self.memory_store)

    def load_from_disk(
        self,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> None:
        if self.persistence is None:
            raise ValueError("Persistence backend is not configured for this MemoryAgent.")
        page_table, memory_store = self.persistence.load(map_location=map_location)
        self.page_table = page_table
        self.memory_store = memory_store

    def get_latent_by_pointer(self, pointer: str) -> LatentTensor:
        storage_id = self.page_table.resolve(pointer)
        if storage_id not in self.memory_store:
            raise KeyError(f"Pointer {pointer} maps to missing storage id {storage_id}.")
        return self.memory_store[storage_id]

    def resolve_explicit_pointers(self, pointers: Sequence[str]) -> list[PointerSearchHit]:
        hits = []
        for pointer in pointers:
            latent = self.get_latent_by_pointer(pointer)
            hits.append(PointerSearchHit(pointer=pointer, score=float("inf"), latent=latent))
        return hits

    def consolidate_episode(
        self,
        hidden_states: torch.Tensor,
        global_reward: float,
        *,
        metadata: Optional[dict] = None,
        utility_decay: float = 1.0,
    ) -> ConsolidationResult:
        reward_value = float(global_reward)
        if reward_value < self.reward_threshold:
            return ConsolidationResult(
                stored=False,
                global_reward=reward_value,
                reason=f"Reward {reward_value:.4f} below threshold {self.reward_threshold:.4f}.",
            )

        if hidden_states.dim() == 3:
            if hidden_states.size(0) != 1:
                raise ValueError("`hidden_states` must be [seq, hidden] or [1, seq, hidden].")
            hidden_states = hidden_states.squeeze(0)
        if hidden_states.dim() != 2:
            raise ValueError(
                "`hidden_states` must have shape [seq, hidden] after squeeze, "
                f"but received {tuple(hidden_states.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                "Hidden size mismatch during consolidation. "
                f"Expected {self.hidden_size}, got {hidden_states.size(-1)}."
            )

        latent_slice = hidden_states[-self.latent_window :].detach()
        pointer = self.add_tensor(
            latent_slice,
            utility_score=reward_value * utility_decay,
            metadata=metadata,
        )
        return ConsolidationResult(
            stored=True,
            pointer=pointer,
            latent=self.get_latent_by_pointer(pointer),
            global_reward=reward_value,
        )


class IPCBus:
    def __init__(self, memory_agent: MemoryAgent, retriever: BaseRetriever) -> None:
        self.memory_agent = memory_agent
        self.retriever = retriever

    def intercept(self, message: AgentMessage, *, top_k_prefetch: int = 0) -> ResolvedIPCMessage:
        explicit_pointers = message.extract_pointers()
        explicit_hits = self.memory_agent.resolve_explicit_pointers(explicit_pointers)
        prefetched_hits = self.retriever.search(
            intent_text=message.content,
            memory_store=self.memory_agent.memory_store,
            top_k=top_k_prefetch,
            exclude_pointers=explicit_pointers,
        )
        return ResolvedIPCMessage(
            message=message,
            explicit_hits=explicit_hits,
            prefetched_hits=prefetched_hits,
        )


class OSKernel(nn.Module):
    def __init__(
        self,
        memory_agent: MemoryAgent,
        retriever: BaseRetriever,
        ipc_bus: Optional[IPCBus] = None,
    ) -> None:
        super().__init__()
        self.memory_agent = memory_agent
        self.retriever = retriever
        self.ipc_bus = ipc_bus or IPCBus(memory_agent=memory_agent, retriever=retriever)

    def handle_ipc(self, message: AgentMessage, *, top_k_prefetch: int = 0) -> ResolvedIPCMessage:
        return self.ipc_bus.intercept(message, top_k_prefetch=top_k_prefetch)

    def consolidate(
        self,
        hidden_states: torch.Tensor,
        global_reward: float,
        *,
        metadata: Optional[dict] = None,
    ) -> ConsolidationResult:
        return self.memory_agent.consolidate_episode(
            hidden_states=hidden_states,
            global_reward=global_reward,
            metadata=metadata,
        )
