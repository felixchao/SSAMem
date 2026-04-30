from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ssamem.data_models import (
    AgentMessage,
    ClusterMemory,
    ConsolidationResult,
    LatentTensor,
    MemoryCluster,
    PageTable,
    PageTableEntry,
    PointerAddress,
    PointerSearchHit,
    ResolvedIPCMessage,
)
from ssamem.retrieval import BaseRetriever, RandomHyperplaneLSHIndex
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
        cluster_assignment_threshold: float = 0.75,
        cluster_assignment_top_k: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim or hidden_size
        self.page_table = page_table or PageTable()
        self.reward_threshold = reward_threshold
        self.latent_window = latent_window
        self.memory_store: dict[str, LatentTensor] = {}
        self.memory_clusters: dict[str, MemoryCluster] = {}
        self.persistence = persistence
        self.autosave = autosave
        self.cluster_assignment_threshold = float(cluster_assignment_threshold)
        self.cluster_assignment_top_k = max(int(cluster_assignment_top_k), 1)
        self.key_projector = (
            nn.Identity() if self.hidden_size == self.key_dim else nn.Linear(self.hidden_size, self.key_dim)
        )
        self.cluster_assignment_lsh = RandomHyperplaneLSHIndex(self.key_dim)

    def build_key_vector(self, tensor_data: torch.Tensor) -> torch.Tensor:
        pooled = tensor_data.mean(dim=0)
        key_vector = self.key_projector(pooled)
        return key_vector.detach().float()

    def register_latent(self, latent: LatentTensor) -> str:
        cluster = self.assign_latent_to_cluster(latent)
        return cluster.pointer_id or ""

    def _autosave_latent(self, latent: LatentTensor) -> None:
        if self.persistence is None or not self.autosave:
            return
        self.persistence.save_page_table(self.page_table)
        self.persistence.save_latent(latent)

    def _autosave_cluster(self, cluster: MemoryCluster) -> None:
        if self.persistence is None or not self.autosave:
            return
        self.persistence.save_page_table(self.page_table)
        self.persistence.save_cluster(cluster)

    def _prepare_tensor_data(self, tensor_data: torch.Tensor) -> torch.Tensor:
        if tensor_data.dim() == 3:
            if tensor_data.size(0) != 1:
                raise ValueError("`tensor_data` must be [K, D] or [1, K, D].")
            tensor_data = tensor_data.squeeze(0)
        if tensor_data.dim() != 2:
            raise ValueError(
                "`tensor_data` must have shape [K, D] after squeeze, "
                f"but received {tuple(tensor_data.shape)}."
            )
        if tensor_data.size(-1) != self.hidden_size:
            raise ValueError(
                "Hidden size mismatch while registering memory. "
                f"Expected {self.hidden_size}, got {tensor_data.size(-1)}."
            )
        return tensor_data.detach()

    def _refresh_cluster_metadata(self, cluster: MemoryCluster) -> None:
        cluster.utility_score = (
            sum(memory.utility_score for memory in cluster.memories) / max(len(cluster.memories), 1)
        )
        if cluster.memories:
            key_bank = torch.stack(
                [memory.latent_tensor.key_vector.float() for memory in cluster.memories],
                dim=0,
            )
            cluster.centroid_embedding = key_bank.mean(dim=0)
            if cluster.summary_key_embedding is None:
                cluster.summary_key_embedding = cluster.centroid_embedding.detach().clone()
        if not cluster.summary_key_text:
            topics = []
            for memory in cluster.memories:
                topic = memory.metadata.get("topic") or memory.latent_tensor.metadata.get("topic")
                if topic:
                    topics.append(str(topic))
            if topics:
                cluster.summary_key_text = topics[0]

        if cluster.pointer_id is not None:
            entry = self.page_table.resolve_entry(cluster.pointer_id)
            entry.cluster_size = cluster.cluster_size
            entry.utility_score = cluster.utility_score
            entry.version = cluster.version
            entry.summary_key_text = cluster.summary_key_text
            if cluster.summary_key_embedding is not None:
                entry.summary_key_embedding = cluster.summary_key_embedding.detach().clone()

    def _cluster_representation(self, cluster: MemoryCluster) -> Optional[torch.Tensor]:
        if cluster.summary_key_embedding is not None and cluster.centroid_embedding is not None:
            return 0.5 * cluster.summary_key_embedding.float() + 0.5 * cluster.centroid_embedding.float()
        if cluster.summary_key_embedding is not None:
            return cluster.summary_key_embedding.float()
        if cluster.centroid_embedding is not None:
            return cluster.centroid_embedding.float()
        if cluster.memories:
            key_bank = torch.stack(
                [memory.latent_tensor.key_vector.float() for memory in cluster.memories],
                dim=0,
            )
            return key_bank.mean(dim=0)
        return None

    def _candidate_clusters_for_key(self, key_vector: torch.Tensor) -> list[tuple[MemoryCluster, float]]:
        if not self.memory_clusters:
            return []
        cluster_vectors: dict[str, torch.Tensor] = {}
        active_clusters: dict[str, MemoryCluster] = {}
        for cluster_id, cluster in self.memory_clusters.items():
            vector = self._cluster_representation(cluster)
            if vector is None or cluster.pointer_id is None:
                continue
            cluster_vectors[cluster_id] = vector.to(key_vector.device, dtype=key_vector.dtype)
            active_clusters[cluster_id] = cluster
        if not cluster_vectors:
            return []

        candidate_ids = list(active_clusters.keys())
        lsh_hits = self.cluster_assignment_lsh.query(key_vector, cluster_vectors)
        if lsh_hits:
            candidate_ids = lsh_hits

        candidate_vectors = torch.stack([cluster_vectors[cluster_id] for cluster_id in candidate_ids], dim=0)
        normalized_query = torch.nn.functional.normalize(key_vector.unsqueeze(0), dim=-1)
        normalized_candidates = torch.nn.functional.normalize(candidate_vectors, dim=-1)
        scores = torch.matmul(normalized_candidates, normalized_query.squeeze(0))
        top_scores, top_indices = torch.topk(
            scores,
            k=min(self.cluster_assignment_top_k, scores.numel()),
            dim=0,
        )
        return [
            (active_clusters[candidate_ids[index]], float(score))
            for score, index in zip(top_scores.tolist(), top_indices.tolist())
        ]

    def _append_prebuilt_latent_to_cluster(
        self,
        cluster: MemoryCluster,
        latent: LatentTensor,
        *,
        metadata: Optional[dict] = None,
        memory_summary: str = "",
        source_agent: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> PointerSearchHit:
        if cluster.pointer_id is None:
            raise ValueError(f"Cluster {cluster.cluster_id} is missing a pointer_id.")
        local_index = cluster.next_local_index()
        combined_metadata = {"cluster_id": cluster.cluster_id, "local_index": local_index, **dict(metadata or {})}
        latent.pointer = cluster.pointer_id
        latent.metadata.update(combined_metadata)
        cluster_memory = self._build_cluster_memory(
            latent,
            local_index=local_index,
            memory_summary=memory_summary,
            source_agent=source_agent,
            timestamp=timestamp,
            utility_score=latent.utility_score,
            metadata=combined_metadata,
        )
        cluster.append_memory(cluster_memory)
        cluster.version += 1
        self.memory_store[latent.storage_id] = latent
        self._refresh_cluster_metadata(cluster)
        self._autosave_cluster(cluster)
        self._autosave_latent(latent)
        return PointerSearchHit(
            pointer=cluster.pointer_id,
            score=float("inf"),
            latent=latent,
            cluster_id=cluster.cluster_id,
            local_index=local_index,
        )

    def assign_latent_to_cluster(
        self,
        latent: LatentTensor,
        *,
        metadata: Optional[dict] = None,
        memory_summary: str = "",
        source_agent: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> MemoryCluster:
        candidates = self._candidate_clusters_for_key(latent.key_vector.float())
        if candidates:
            best_cluster, best_score = candidates[0]
            if best_score >= self.cluster_assignment_threshold:
                self._append_prebuilt_latent_to_cluster(
                    best_cluster,
                    latent,
                    metadata=metadata,
                    memory_summary=memory_summary,
                    source_agent=source_agent,
                    timestamp=timestamp,
                )
                return best_cluster
        return self.create_cluster_from_latent(
            latent,
            summary_key_text=str((metadata or {}).get("topic") or latent.metadata.get("topic") or ""),
            summary_key_embedding=latent.key_vector.detach().clone(),
            metadata=metadata,
        )

    def _build_cluster_memory(
        self,
        latent: LatentTensor,
        *,
        local_index: int,
        memory_summary: str = "",
        source_agent: Optional[str] = None,
        timestamp: Optional[str] = None,
        utility_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> ClusterMemory:
        combined_metadata = dict(metadata or {})
        if combined_metadata:
            latent.metadata.update(combined_metadata)
        return ClusterMemory(
            local_index=local_index,
            latent_tensor=latent,
            memory_key_embedding=latent.key_vector.detach().clone(),
            memory_summary=memory_summary or str(latent.metadata.get("summary") or latent.metadata.get("topic") or ""),
            source_agent=source_agent,
            timestamp=timestamp,
            utility_score=float(utility_score if utility_score is not None else latent.utility_score),
            metadata=combined_metadata,
        )

    def create_cluster_from_latent(
        self,
        latent: LatentTensor,
        *,
        summary_key_text: str = "",
        summary_key_embedding: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryCluster:
        entry = self.page_table.allocate_cluster(
            cluster_id=latent.storage_id,
            summary_key_text=summary_key_text or str(latent.metadata.get("topic") or ""),
            summary_key_embedding=summary_key_embedding,
            cluster_size=1,
            utility_score=latent.utility_score,
            metadata=metadata,
        )
        latent.pointer = entry.pointer_id
        latent.metadata.setdefault("cluster_id", entry.cluster_id)
        latent.metadata.setdefault("local_index", 0)
        cluster = MemoryCluster(
            cluster_id=entry.cluster_id,
            pointer_id=entry.pointer_id,
            summary_key_text=entry.summary_key_text,
            summary_key_embedding=summary_key_embedding.detach().clone() if summary_key_embedding is not None else None,
            memories=[
                self._build_cluster_memory(
                    latent,
                    local_index=0,
                    utility_score=latent.utility_score,
                    metadata={"cluster_id": entry.cluster_id, "local_index": 0, **dict(metadata or {})},
                )
            ],
            utility_score=latent.utility_score,
            version=entry.version,
            metadata=dict(metadata or {}),
        )
        self.memory_clusters[cluster.cluster_id] = cluster
        self.memory_store[latent.storage_id] = latent
        self._refresh_cluster_metadata(cluster)
        self._autosave_cluster(cluster)
        self._autosave_latent(latent)
        return cluster

    def get_cluster_by_pointer(self, pointer: str) -> MemoryCluster:
        entry = self.page_table.resolve_entry(pointer)
        if entry.cluster_id in self.memory_clusters:
            return self.memory_clusters[entry.cluster_id]
        if entry.legacy_storage_id is not None and entry.legacy_storage_id in self.memory_store:
            latent = self.memory_store[entry.legacy_storage_id]
            latent.pointer = pointer
            cluster = MemoryCluster(
                cluster_id=entry.cluster_id,
                pointer_id=pointer,
                summary_key_text=entry.summary_key_text,
                summary_key_embedding=entry.summary_key_embedding.detach().clone()
                if entry.summary_key_embedding is not None
                else None,
                memories=[
                    self._build_cluster_memory(
                        latent,
                        local_index=0,
                        utility_score=latent.utility_score,
                        metadata={"cluster_id": entry.cluster_id, "local_index": 0},
                    )
                ],
                utility_score=entry.utility_score,
                version=entry.version,
                metadata=dict(entry.metadata or {}),
            )
            self.memory_clusters[cluster.cluster_id] = cluster
            self._refresh_cluster_metadata(cluster)
            return cluster
        raise KeyError(f"Pointer {pointer} maps to missing cluster {entry.cluster_id}.")

    def append_memory_to_cluster(
        self,
        pointer: str,
        tensor_data: torch.Tensor,
        *,
        utility_score: float = 1.0,
        metadata: Optional[dict] = None,
        memory_summary: str = "",
        source_agent: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> PointerSearchHit:
        cluster = self.get_cluster_by_pointer(pointer)
        prepared_tensor = self._prepare_tensor_data(tensor_data)
        key_vector = self.build_key_vector(prepared_tensor)
        latent = LatentTensor(
            tensor_data=prepared_tensor,
            key_vector=key_vector,
            utility_score=utility_score,
            pointer=pointer,
            metadata=dict(metadata or {}),
        )
        return self._append_prebuilt_latent_to_cluster(
            cluster,
            latent,
            metadata=metadata,
            memory_summary=memory_summary,
            source_agent=source_agent,
            timestamp=timestamp,
        )

    def add_tensor(
        self,
        tensor_data: torch.Tensor,
        *,
        utility_score: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> str:
        prepared_tensor = self._prepare_tensor_data(tensor_data)
        key_vector = self.build_key_vector(prepared_tensor)
        latent = LatentTensor(
            tensor_data=prepared_tensor,
            key_vector=key_vector,
            utility_score=utility_score,
            metadata=dict(metadata or {}),
        )
        return self.register_latent(latent)

    def save_to_disk(self) -> None:
        if self.persistence is None:
            raise ValueError("Persistence backend is not configured for this MemoryAgent.")
        self.persistence.save(self.page_table, self.memory_store, self.memory_clusters)

    def load_from_disk(
        self,
        *,
        map_location: Optional[str | torch.device] = "cpu",
    ) -> None:
        if self.persistence is None:
            raise ValueError("Persistence backend is not configured for this MemoryAgent.")
        page_table, memory_store, memory_clusters = self.persistence.load(map_location=map_location)
        self.page_table = page_table
        self.memory_store = memory_store
        self.memory_clusters = memory_clusters

    def get_latent_by_pointer(self, pointer: str) -> LatentTensor:
        cluster = self.get_cluster_by_pointer(pointer)
        if not cluster.memories:
            raise KeyError(f"Pointer {pointer} maps to empty cluster {cluster.cluster_id}.")
        return cluster.memories[0].latent_tensor

    def get_memory_by_address(self, pointer: str, local_index: int) -> ClusterMemory:
        cluster = self.get_cluster_by_pointer(pointer)
        return cluster.get_memory(local_index)

    def get_latent_by_address(self, pointer: str, local_index: int) -> LatentTensor:
        return self.get_memory_by_address(pointer, local_index).latent_tensor

    def resolve_explicit_pointers(self, pointers: Sequence[str | PointerAddress]) -> list[PointerSearchHit]:
        hits = []
        for pointer_ref in pointers:
            address = pointer_ref if isinstance(pointer_ref, PointerAddress) else PointerAddress(pointer=str(pointer_ref))
            if address.local_index is None:
                cluster = self.get_cluster_by_pointer(address.pointer)
                latent = cluster.memories[0].latent_tensor
                local_index = cluster.memories[0].local_index
                cluster_id = cluster.cluster_id
            else:
                memory = self.get_memory_by_address(address.pointer, address.local_index)
                latent = memory.latent_tensor
                local_index = memory.local_index
                cluster_id = memory.latent_tensor.metadata.get("cluster_id")
            hits.append(
                PointerSearchHit(
                    pointer=address.pointer,
                    score=float("inf"),
                    latent=latent,
                    cluster_id=str(cluster_id) if cluster_id is not None else None,
                    local_index=local_index,
                )
            )
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
        explicit_addresses = message.extract_pointer_addresses()
        explicit_hits = self.memory_agent.resolve_explicit_pointers(explicit_addresses)
        explicit_pointers = [address.pointer for address in explicit_addresses]
        prefetched_hits = self.retriever.search(
            intent_text=message.content,
            memory_store=self.memory_agent.memory_store,
            memory_clusters=self.memory_agent.memory_clusters,
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
