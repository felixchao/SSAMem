from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

import torch
from torch import nn

from latent_os.data_models import (
    AgentMessage,
    ConsolidationResult,
    LatentTensor,
    PageTable,
    PointerSearchHit,
    ResolvedIPCMessage,
)
from latent_os.storage import PersistentMemoryBackend


class HashEmbeddingEncoder(nn.Module):
    def __init__(self, key_dim: int, vocab_size: int = 4096) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, key_dim)

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.vocab_size

    def tokenize(self, text: str) -> list[int]:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        if not tokens:
            tokens = ["<empty>"]
        return [self._token_to_id(token) for token in tokens]

    def forward(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        pooled_vectors = []
        for text in texts:
            token_ids = torch.tensor(self.tokenize(text), device=self.embedding.weight.device)
            token_embeds = self.embedding(token_ids)
            pooled_vectors.append(token_embeds.mean(dim=0))
        return torch.stack(pooled_vectors, dim=0)


class MemoryAgent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: Optional[int] = None,
        *,
        page_table: Optional[PageTable] = None,
        query_encoder: Optional[nn.Module] = None,
        reward_threshold: float = 0.0,
        latent_window: int = 8,
        persistence: Optional[PersistentMemoryBackend] = None,
        autosave: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim or hidden_size
        self.page_table = page_table or PageTable()
        self.query_encoder = query_encoder or HashEmbeddingEncoder(key_dim=self.key_dim)
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

    def maximum_inner_product_search(
        self,
        intent_text: str,
        *,
        top_k: int = 1,
        exclude_pointers: Optional[Sequence[str]] = None,
    ) -> list[PointerSearchHit]:
        if top_k <= 0 or not self.memory_store:
            return []

        exclude = set(exclude_pointers or [])
        valid_memories = [latent for latent in self.memory_store.values() if latent.pointer not in exclude]
        if not valid_memories:
            return []

        query = self.query_encoder(intent_text).squeeze(0).float()
        key_bank = torch.stack([latent.key_vector.to(query.device, dtype=query.dtype) for latent in valid_memories], dim=0)
        scores = torch.matmul(key_bank, query)

        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.numel()), dim=0)
        hits = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            latent = valid_memories[index]
            hits.append(PointerSearchHit(pointer=latent.pointer or "", score=float(score), latent=latent))
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
    def __init__(self, memory_agent: MemoryAgent) -> None:
        self.memory_agent = memory_agent

    def intercept(self, message: AgentMessage, *, top_k_prefetch: int = 0) -> ResolvedIPCMessage:
        explicit_pointers = message.extract_pointers()
        explicit_hits = self.memory_agent.resolve_explicit_pointers(explicit_pointers)
        prefetched_hits = self.memory_agent.maximum_inner_product_search(
            message.content,
            top_k=top_k_prefetch,
            exclude_pointers=explicit_pointers,
        )
        return ResolvedIPCMessage(
            message=message,
            explicit_hits=explicit_hits,
            prefetched_hits=prefetched_hits,
        )


class OSKernel(nn.Module):
    def __init__(self, memory_agent: MemoryAgent, ipc_bus: Optional[IPCBus] = None) -> None:
        super().__init__()
        self.memory_agent = memory_agent
        self.ipc_bus = ipc_bus or IPCBus(memory_agent=memory_agent)

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
