from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from ssamem.data_models import LatentTensor, MemoryCluster, PointerSearchHit


class BaseQueryEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, texts: str | Sequence[str]) -> torch.Tensor:
        raise NotImplementedError


class HashEmbeddingEncoder(BaseQueryEncoder):
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


@dataclass
class ClusterCandidate:
    cluster: MemoryCluster
    score: float


class BaseRetriever(ABC):
    @abstractmethod
    def search(
        self,
        *,
        intent_text: str,
        memory_store: dict[str, LatentTensor],
        memory_clusters: Optional[dict[str, MemoryCluster]] = None,
        top_k: int = 1,
        exclude_pointers: Optional[Sequence[str]] = None,
    ) -> list[PointerSearchHit]:
        raise NotImplementedError


class RandomHyperplaneLSHIndex:
    def __init__(self, dim: int, *, num_tables: int = 4, num_planes: int = 12, seed: int = 7) -> None:
        self.dim = dim
        self.num_tables = num_tables
        self.num_planes = num_planes
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.hyperplanes = torch.randn(num_tables, num_planes, dim, generator=generator)

    def _signature(self, vector: torch.Tensor, table_index: int) -> tuple[int, ...]:
        projections = torch.matmul(self.hyperplanes[table_index].to(vector.device, dtype=vector.dtype), vector)
        return tuple(int(value.item() > 0) for value in projections)

    def build_buckets(self, cluster_vectors: dict[str, torch.Tensor]) -> list[dict[tuple[int, ...], set[str]]]:
        buckets = [dict() for _ in range(self.num_tables)]
        for cluster_id, vector in cluster_vectors.items():
            for table_index in range(self.num_tables):
                signature = self._signature(vector, table_index)
                buckets[table_index].setdefault(signature, set()).add(cluster_id)
        return buckets

    def query(
        self,
        query_vector: torch.Tensor,
        cluster_vectors: dict[str, torch.Tensor],
    ) -> list[str]:
        if not cluster_vectors:
            return []
        buckets = self.build_buckets(cluster_vectors)
        matches: set[str] = set()
        for table_index in range(self.num_tables):
            signature = self._signature(query_vector, table_index)
            matches.update(buckets[table_index].get(signature, set()))
        return list(matches)


class DenseInnerProductRetriever(BaseRetriever):
    def __init__(
        self,
        query_encoder: BaseQueryEncoder,
        *,
        lsh_index: Optional[RandomHyperplaneLSHIndex] = None,
        cluster_lsh_candidate_k: int = 8,
        cluster_rerank_k: int = 3,
    ) -> None:
        self.query_encoder = query_encoder
        self.lsh_index = lsh_index
        self.cluster_lsh_candidate_k = cluster_lsh_candidate_k
        self.cluster_rerank_k = cluster_rerank_k

    def _cluster_representation(self, cluster: MemoryCluster) -> Optional[torch.Tensor]:
        summary = cluster.summary_key_embedding
        centroid = cluster.centroid_embedding
        if summary is not None and centroid is not None:
            return 0.5 * summary.float() + 0.5 * centroid.float()
        if summary is not None:
            return summary.float()
        if centroid is not None:
            return centroid.float()
        if cluster.memories:
            key_bank = torch.stack([memory.latent_tensor.key_vector.float() for memory in cluster.memories], dim=0)
            return key_bank.mean(dim=0)
        return None

    def _candidate_clusters(
        self,
        query: torch.Tensor,
        memory_clusters: dict[str, MemoryCluster],
        *,
        exclude_pointers: set[str],
    ) -> list[ClusterCandidate]:
        cluster_vectors = {}
        active_clusters = {}
        for cluster_id, cluster in memory_clusters.items():
            if cluster.pointer_id in exclude_pointers:
                continue
            vector = self._cluster_representation(cluster)
            if vector is None:
                continue
            cluster_vectors[cluster_id] = vector.to(query.device, dtype=query.dtype)
            active_clusters[cluster_id] = cluster
        if not cluster_vectors:
            return []

        candidate_ids = list(active_clusters.keys())
        if self.lsh_index is not None:
            lsh_hits = self.lsh_index.query(query, cluster_vectors)
            if lsh_hits:
                candidate_ids = lsh_hits

        candidate_vectors = torch.stack([cluster_vectors[cluster_id] for cluster_id in candidate_ids], dim=0)
        scores = torch.matmul(candidate_vectors, query)
        top_scores, top_indices = torch.topk(
            scores,
            k=min(self.cluster_rerank_k, scores.numel()),
            dim=0,
        )
        candidates = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            cluster_id = candidate_ids[index]
            candidates.append(ClusterCandidate(cluster=active_clusters[cluster_id], score=float(score)))
        return candidates

    def _memory_hits_from_clusters(
        self,
        query: torch.Tensor,
        candidates: list[ClusterCandidate],
        *,
        top_k: int,
    ) -> list[PointerSearchHit]:
        memory_hits: list[PointerSearchHit] = []
        for candidate in candidates:
            cluster = candidate.cluster
            if not cluster.memories or cluster.pointer_id is None:
                continue
            key_bank = torch.stack(
                [memory.latent_tensor.key_vector.to(query.device, dtype=query.dtype) for memory in cluster.memories],
                dim=0,
            )
            scores = torch.matmul(key_bank, query)
            local_top_scores, local_top_indices = torch.topk(scores, k=min(top_k, scores.numel()), dim=0)
            for score, index in zip(local_top_scores.tolist(), local_top_indices.tolist()):
                memory = cluster.memories[index]
                memory_hits.append(
                    PointerSearchHit(
                        pointer=cluster.pointer_id,
                        score=float(score),
                        latent=memory.latent_tensor,
                        cluster_id=cluster.cluster_id,
                        local_index=memory.local_index,
                    )
                )

        memory_hits.sort(key=lambda hit: hit.score, reverse=True)
        return memory_hits[:top_k]

    def _fallback_global_search(
        self,
        query: torch.Tensor,
        memory_store: dict[str, LatentTensor],
        *,
        top_k: int,
        exclude_pointers: set[str],
    ) -> list[PointerSearchHit]:
        valid_memories = [latent for latent in memory_store.values() if latent.pointer not in exclude_pointers]
        if not valid_memories:
            return []
        key_bank = torch.stack(
            [latent.key_vector.to(query.device, dtype=query.dtype) for latent in valid_memories],
            dim=0,
        )
        scores = torch.matmul(key_bank, query)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.numel()), dim=0)
        hits = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            latent = valid_memories[index]
            hits.append(
                PointerSearchHit(
                    pointer=latent.pointer or "",
                    score=float(score),
                    latent=latent,
                    cluster_id=latent.metadata.get("cluster_id"),
                    local_index=latent.metadata.get("local_index"),
                )
            )
        return hits

    def search(
        self,
        *,
        intent_text: str,
        memory_store: dict[str, LatentTensor],
        memory_clusters: Optional[dict[str, MemoryCluster]] = None,
        top_k: int = 1,
        exclude_pointers: Optional[Sequence[str]] = None,
    ) -> list[PointerSearchHit]:
        if top_k <= 0:
            return []
        query = self.query_encoder(intent_text).squeeze(0).float()
        exclude = set(exclude_pointers or [])

        if memory_clusters:
            candidates = self._candidate_clusters(query, memory_clusters, exclude_pointers=exclude)
            if candidates:
                return self._memory_hits_from_clusters(query, candidates, top_k=top_k)

        return self._fallback_global_search(query, memory_store, top_k=top_k, exclude_pointers=exclude)


class HashMIPSRetriever(DenseInnerProductRetriever):
    def __init__(self, key_dim: int, vocab_size: int = 4096) -> None:
        super().__init__(
            query_encoder=HashEmbeddingEncoder(key_dim=key_dim, vocab_size=vocab_size),
            lsh_index=RandomHyperplaneLSHIndex(key_dim),
        )


def build_retriever(
    retriever_type: str,
    *,
    key_dim: int,
    hash_vocab_size: int = 4096,
) -> BaseRetriever:
    normalized = retriever_type.strip().lower()
    if normalized == "hash_mips":
        return HashMIPSRetriever(key_dim=key_dim, vocab_size=hash_vocab_size)
    raise ValueError(f"Unsupported retriever type: {retriever_type}")
