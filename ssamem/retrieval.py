from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from torch import nn

from ssamem.data_models import LatentTensor, PointerSearchHit


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


class BaseRetriever(ABC):
    @abstractmethod
    def search(
        self,
        *,
        intent_text: str,
        memory_store: dict[str, LatentTensor],
        top_k: int = 1,
        exclude_pointers: Optional[Sequence[str]] = None,
    ) -> list[PointerSearchHit]:
        raise NotImplementedError


class DenseInnerProductRetriever(BaseRetriever):
    def __init__(self, query_encoder: BaseQueryEncoder) -> None:
        self.query_encoder = query_encoder

    def search(
        self,
        *,
        intent_text: str,
        memory_store: dict[str, LatentTensor],
        top_k: int = 1,
        exclude_pointers: Optional[Sequence[str]] = None,
    ) -> list[PointerSearchHit]:
        if top_k <= 0 or not memory_store:
            return []

        exclude = set(exclude_pointers or [])
        valid_memories = [latent for latent in memory_store.values() if latent.pointer not in exclude]
        if not valid_memories:
            return []

        query = self.query_encoder(intent_text).squeeze(0).float()
        key_bank = torch.stack(
            [latent.key_vector.to(query.device, dtype=query.dtype) for latent in valid_memories],
            dim=0,
        )
        scores = torch.matmul(key_bank, query)

        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.numel()), dim=0)
        hits = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            latent = valid_memories[index]
            hits.append(PointerSearchHit(pointer=latent.pointer or "", score=float(score), latent=latent))
        return hits


class HashMIPSRetriever(DenseInnerProductRetriever):
    def __init__(self, key_dim: int, vocab_size: int = 4096) -> None:
        super().__init__(query_encoder=HashEmbeddingEncoder(key_dim=key_dim, vocab_size=vocab_size))


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
